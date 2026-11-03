"""基于 smolagents 的执行与校验入口。"""
import asyncio
import json
import inspect
import hashlib
from dataclasses import dataclass, field
from typing import Any, Dict, List
from smolagents import ToolCallingAgent


@dataclass
class Task:
    id: str
    type: str
    description: str
    parameters: Dict[str, Any]
    dependencies: List[str] = field(default_factory=list)
    priority: int = 1
    max_retries: int = 3


@dataclass
class TaskResult:
    task_id: str
    success: bool
    result: Any = None
    error: str = None


def json_object(value):
    """兼容 final_answer 返回字典或 JSON 文本；拒绝说明性文字。"""
    if isinstance(value, str):
        value = value.strip()
        if value.startswith('```') and value.endswith('```'):
            value = '\n'.join(value.splitlines()[1:-1])
        try:
            value = json.loads(value)
        except json.JSONDecodeError as exc:
            raise ValueError('Submit the requested JSON object in final_answer.answer, not a status message') from exc
    if not isinstance(value, dict):
        raise ValueError('final_answer must contain a JSON object')
    return value


class BaseAgent(ToolCallingAgent):
    """保留原生工具调用循环；业务校验失败会成为下一步的反馈。"""

    def __init__(self, name, model, memory=None, tools=None, instructions=None):
        # 每个 Agent 使用框架自己的记忆，避免不同任务共享可变会话。
        super().__init__(
            name=name, model=model, tools=tools or [], instructions=instructions,
            max_steps=12, max_tool_threads=1, verbosity_level=0,
        )
        self.observations = []
        for tool in self.tools.values():
            if tool.name == 'final_answer':
                continue
            forward = tool.forward

            def recorded(*args, _forward=forward, _name=tool.name, **kwargs):
                # 限制来源工具只能读取用户给定或真实搜索返回的地址。
                bound = inspect.signature(_forward).bind(*args, **kwargs).arguments
                targets = bound.get('urls', []) or []
                if bound.get('url') or bound.get('pdf_path'):
                    targets = [bound.get('url') or bound.get('pdf_path')]
                allowed = getattr(self, 'allowed_targets', set())
                if targets and any(target not in allowed for target in targets):
                    return {'error': 'Unknown target: discover the URL with search before reading it'}
                result = _forward(*args, **kwargs)
                if _name == 'discover_sources' and isinstance(result, dict):
                    allowed.update(item['url'] for item in result.get('sources', []) if item.get('url'))
                    self.allowed_targets = allowed
                if isinstance(result, dict) and not result.get('error'):
                    items = result.get('extracted_content', [])
                    if result.get('url') and result.get('content'):
                        items = [result]
                    for item in items:
                        if item.get('url') and item.get('content'):
                            item['source_id'] = 'source_' + hashlib.sha256(item['url'].encode()).hexdigest()[:12]
                self.observations.append({'tool': _name, 'result': result})
                return result

            tool.forward = recorded

    async def run_checked(self, prompt, validator):
        """在线程中执行 Agent，并拒绝步数耗尽时框架生成的兜底答案。

        Args:
            prompt: 本次任务、上下文与输出约束。
            validator: 将 final_answer 校验并转换为业务结果的函数。

        Returns:
            dict: validator 返回的已校验结果；校验错误抛给调度器。
        """
        self.observations.clear()

        def check_answer(answer, memory, agent):
            validator(json_object(answer))
            return True

        self.final_answer_checks = [check_answer]
        result = await asyncio.to_thread(self.run, prompt, return_full_result=True)
        if result.state != 'success':
            raise ValueError(f'{self.name}: agent exhausted its step budget without a valid answer')
        return validator(json_object(result.output))

    async def execute_task(self, task):
        """兼容任务对象入口；具体输出约束由任务调用方提供。"""
        try:
            result = await self.run_checked(task.description, json_object)
            return TaskResult(task.id, True, result=result)
        except Exception as exc:
            return TaskResult(task.id, False, error=str(exc))

    def get_capabilities(self):
        return [name for name in self.tools if name != 'final_answer']
