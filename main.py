#!/usr/bin/env python3
"""
DeepReport - AI-powered financial report generation system
Main application with Gradio frontend
"""

import asyncio
import copy
from html import escape
from urllib.parse import urlparse
import gradio as gr
import logging
import json
import os
import re
from datetime import datetime
from typing import Dict, Any, List, Optional

from src.agents import PlanningAgent
from src.agents.sub_agents import DeepResearcherAgent, BrowserAgent, DeepAnalyzeAgent, FinalAnswerAgent
from src.search import SearchManager
from src.utils import ModelAdapter, MCPManager
from src.report import HTMLReportGenerator
from src.report.section_renderer import render_sections
from src.report.research_writer import ResearchWriter
from src.report.research_sources import ResearchTrace, review_sources, search_queries
from config import config, get_model_config
from smolagents import OpenAIModel, LiteLLMModel
from smolagents.memory import AgentMemory
from smolagents.models import ChatMessage

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DeepReportApp:
    """Main application class"""
    
    def __init__(self):
        self.config = config
        self.model_config = get_model_config()
        
        # Initialize components
        self.search_manager = SearchManager({
            "serper_api_key": config.serper_api_key,
            "metaso_api_key": config.metaso_api_key,
            "sogou_api_key": config.sogou_api_key
        })
        
        self.mcp_manager = MCPManager({
            "mcp_server_url": config.mcp_server_url,
            "mcp_api_key": config.mcp_api_key
        })
        
        self.report_generator = HTMLReportGenerator({
            "charts_enabled": config.charts_enabled,
            "data_sources_enabled": config.data_sources_enabled
        })
        
        # Initialize agents
        self.agents = {}
        self._initialize_agents()
        
        # Create output directory
        os.makedirs(config.output_dir, exist_ok=True)
    
    def _initialize_agents(self):
        """Initialize all agents using SmolAgents framework"""
        try:
            # Initialize SmolAgents model
            model_id = self.model_config.get("model", "gpt-4o")
            if not self.model_config.get("api_key"):
                raise ValueError(f"No API key configured for model {model_id}")
            if "claude" in model_id.lower():
                self.model = LiteLLMModel(
                    model_id=f"anthropic/{model_id}",
                    api_key=self.model_config["api_key"]
                )
            else:
                self.model = OpenAIModel(
                    model_id=model_id,
                    api_key=self.model_config["api_key"],
                    api_base=self.model_config.get("base_url", "https://api.chatanywhere.tech/v1"),
                    client_kwargs={"timeout": 90.0, "max_retries": 1}
                )

            # Initialize memory
            memory = AgentMemory(system_prompt="You are DeepReport, an AI-powered financial research assistant.")

            # Initialize planning agent
            self.agents["planning"] = PlanningAgent(self.model, memory)
            
            # Initialize sub-agents
            self.agents["deep_researcher"] = DeepResearcherAgent(self.model, memory)
            self.agents["browser"] = BrowserAgent(self.model, memory)
            self.agents["deep_analyze"] = DeepAnalyzeAgent(self.model, memory)
            self.agents["final_answer"] = FinalAnswerAgent(self.model, memory)
            
            logger.info("All agents initialized successfully with SmolAgents framework")
            
        except Exception as e:
            logger.error(f"Failed to initialize agents: {e}")
            raise
    
    async def generate_report(self, research_topic: str, requirements: str,
                              output_format: str = "html", model: str = "gpt-4o") -> Dict[str, Any]:
        # Each request gets its own model and agents; concurrent requests cannot switch each other.
        try:
            request_app = copy.copy(self)
            request_app.model_config = get_model_config(model)
            request_app.agents = {}
            request_app.trace = ResearchTrace(config.output_dir)
            request_app._initialize_agents()
            result = await request_app._generate_report(research_topic, requirements, output_format, model)
            result['trace_path'] = str(request_app.trace.path)
            request_app.trace.record('result', result)
            return result
        except Exception as exc:
            return {"success": False, "error": str(exc)}

    async def _generate_report(
        self,
        research_topic: str,
        requirements: str,
        output_format: str = "html",
        model: str = "gpt-4o",
    ) -> Dict[str, Any]:
        """规划并执行研究任务，汇总结果后返回报告或结构化数据。

        Args:
            research_topic: 本次报告的研究主题。
                真实示例："2026 中国经济展望及是否看好"。
            requirements: 用户要求，多条要求用换行分隔，空行会被忽略。
                真实示例："洞察力、客观"；解析后为 ["洞察力、客观"]。
            output_format: 输出格式；"html" 会渲染并保存报告，其他值返回数据。
                真实示例："html"，对应下方成功示例中的 HTML 报告。
            model: 写入报告元数据的模型标识。实际模型实例由外层
                generate_report() 初始化，本方法使用 self.model。
                真实示例："gpt-4o"。

        Returns:
            包含 success 标志的结果字典：
            - HTML 成功：report_path、report_html、report_data、generation_time。
            - 非 HTML 成功：report_data、generation_time。
            - 失败：error；计划执行不完整时另含 failed_tasks 和 task_results。

        Examples:
            与下方成功记录对应的调用参数（通过外层入口初始化请求级模型）::

                result = await app.generate_report(
                    research_topic="2026 中国经济展望及是否看好",
                    requirements="洞察力、客观",
                    output_format="html",
                    model="gpt-4o",
                )

            以下摘自真实运行记录；使用 Python 字典形式展示，True/False 为
            布尔值。省略号（...）仅代表未展开的内容，不是实际返回值。

            HTML 生成成功（reports/runs/20261102_205851_d66c0992/009_result.json）::

                {
                    "success": True,
                    "report_path": "./reports/financial_report_20261102_205952.html",
                    "report_html": ...,  # 实际为完整 HTML 字符串。
                    "report_data": {
                        "research_topic": "2026 中国经济展望及是否看好",
                        "model": "gpt-4o",
                        "generated_at": "2026-11-02T20:59:52.583547",
                        # 其余真实字段（plan、task_results、sections 等）省略。
                    },
                    "generation_time": "2026-11-02T20:59:52.586897",
                }

            研究任务失败（reports/runs/20261102_173405_da984c55/016_result.json）::

                {
                    "success": False,
                    "error": "Research plan incomplete",
                    "failed_tasks": {
                        "task_003": (
                            "Research content failed validation after two attempts: "
                            "Evidence ID is not in the supplied source passages; "
                            "select an ID from sources"
                        ),
                        "task_005": "Failed dependencies: ['task_003']",
                    },
                    "task_results": ...,  # 实际包含所有已执行或被阻断任务的结果。
                }

            上例中 task_003 的证据引用校验失败，依赖它的 task_005 因此被阻断。
            运行记录另有 trace_path，由外层 generate_report() 添加，
            不属于本方法直接返回的字段，因此以上示例未列出。

        Notes:
            研究任务全部成功后才组装报告。解析、执行或渲染中的异常在此
            转为失败结果；运行记录的创建与最终结果记录由外层方法负责。
        """
        try:
            # 1. 清理逐行输入，保留非空要求，供计划生成和后续任务共同使用。
            requirements_list = [
                req.strip()
                for req in requirements.split("\n")
                if req.strip()
            ]

            logger.info(f"Starting report generation for topic: {research_topic}")

            # 2. 直接向模型请求结构化计划；此阶段不进入 PlanningAgent 的工具循环。
            # 提示中明确任务依赖、URL 来源和浏览器动作格式，供执行器使用。
            planning_prompt = f"""You are an expert financial research planner. Create a comprehensive research plan as a JSON object.

Research Topic: {research_topic}
Requirements: {', '.join(requirements_list) if requirements_list else 'None specified'}
Output Format: {output_format}

Respond ONLY with a valid JSON object in this exact format:
{{
    "overview": "Brief description of the research approach",
    "tasks": [
        {{
            "task_id": "task_001",
            "task_type": "deep_researcher",
            "description": "What the task should accomplish",
            "parameters": {{}},
            "dependencies": [],
            "priority": 1,
            "estimated_duration": 30
        }}
    ],
    "expected_output": "What the final report should contain",
    "data_sources": ["string"],
    "citations_required": ["string"],
    "estimated_duration": 0
}}

Use task_type from: deep_researcher, browser_use, deep_analyze, final_answer.
Include 3-5 tasks covering research, data collection, analysis, and final summary.
Every analysis task must depend on research/data collection tasks; final_answer must depend on analysis.
Never invent URLs or PDF filenames from memory. Use url/pdf_path only if explicitly supplied
by the user; otherwise provide parameters.query to discover a real source at execution time.
Keep document dates and planning periods relevant to the research topic. Set exact_target=true
only when the user requires that specific document/page rather than research on its subject.
For browser_use, parameters must include browser_task_type
(web_navigation, pdf_analysis, structured_data_extraction, form_interaction).
Browser actions must be structured objects: {{"type": "click|fill|scroll|extract", "target": "CSS selector", "value": "optional"}}.
For ordinary reading/extraction, omit action/actions. Use browser_task_type=web_navigation.
Never invent CSS selectors for pages that have not been inspected.
Browser tasks should depend on the relevant research tasks so they can use the discovered URLs.
exact_target must be a JSON boolean and requires a user-supplied url or pdf_path.
Use concise search queries in parameters.query for research or browser tasks without a URL."""

            messages = [ChatMessage(role="user", content=planning_prompt)]
            # model.generate 是同步调用，放入线程执行，避免阻塞异步事件循环。
            response = await asyncio.to_thread(self.model.generate, messages)
            response_text = (
                response.content if hasattr(response, "content") else str(response)
            )

            # 兼容模型返回的 Markdown 代码围栏，再将正文解析为 JSON。
            plan_text = response_text.strip()
            if plan_text.startswith("```"):
                plan_text = "\n".join(
                    l
                    for l in plan_text.split("\n")[1:]
                    if not l.strip().startswith("```")
                )

            # 在 JSON 解析前保存原始规划响应，解析失败时也有依据可追查。
            if getattr(self, "trace", None):
                self.trace.record(
                    "planning",
                    {"topic": research_topic, "response": response_text},
                )
            plan_dict = json.loads(plan_text)
            if "tasks" not in plan_dict or not isinstance(plan_dict["tasks"], list):
                return {
                    "success": False,
                    "error": "Invalid plan format: missing tasks list",
                }

            # 3. 向计划补充主题与要求，再由执行器校验依赖并逐项执行就绪任务。
            plan_dict["research_topic"] = research_topic
            plan_dict["requirements"] = requirements_list
            task_results = await self._execute_research_plan(plan_dict)
            # 汇总失败任务；不将缺少必要研究结果的运行发布为成功报告。
            failed = {
                key: value.get("error", "Task failed")
                for key, value in task_results.items()
                if not value.get("success")
            }
            if failed or not task_results:
                return {
                    "success": False,
                    "error": "Research plan incomplete",
                    "failed_tasks": failed,
                    "task_results": task_results,
                }

            # 4. 从成功任务中提取展示所需数据；此处组装报告，不再调用模型写作。
            report_data = {
                "title": f"Financial Research Report: {research_topic}",
                "subtitle": "Comprehensive Analysis and Insights",
                "research_topic": research_topic,
                "requirements": requirements_list,
                "plan": plan_dict,
                "task_results": task_results,
                "generated_at": datetime.now().isoformat(),
                "model": model,
                # 保留现有固定值；这不是本次运行计算出的置信度。
                "confidence_score": 85,
                # 沿用现有统计口径：对各结果 data 的 len 求和。
                # data 为字典时统计的是字段数，不代表实际经济观测数量。
                "data_points": sum(
                    len(r.get("data", []))
                    for r in task_results.values()
                    if isinstance(r, dict)
                ),
                "charts_data": self._extract_chart_data(task_results),
                "citations": self._extract_citations(task_results),
                "data_sources": self._extract_data_sources(task_results),
                "sections": self._create_report_sections(task_results),
                "key_metrics": self._extract_key_metrics(task_results),
                "risk_assessment": self._extract_risk_assessment(task_results),
                "recommendations": self._extract_recommendations(task_results),
                "executive_summary": self._generate_executive_summary(
                    plan_dict, task_results
                ),
            }

            # 5. HTML 分支先渲染，再保存文件；非 HTML 分支只返回结构化数据。
            if output_format == "html":
                html_result = await self.report_generator.generate_report(report_data)

                if html_result["success"]:
                    # 用时间戳命名，写入配置的报告输出目录。
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    filename = f"financial_report_{timestamp}.html"
                    filepath = os.path.join(config.output_dir, filename)

                    # save_report 内部会再次渲染；保持现有保存行为。
                    success = await self.report_generator.save_report(
                        report_data, filepath
                    )

                    if success:
                        return {
                            "success": True,
                            "report_path": filepath,
                            "report_html": html_result["html_content"],
                            "report_data": report_data,
                            "generation_time": datetime.now().isoformat(),
                        }
                    else:
                        return {"success": False, "error": "Failed to save report"}
                else:
                    return {
                        "success": False,
                        "error": html_result.get("error", "Failed to generate HTML"),
                    }

            # JSON 等非 HTML 格式由界面层序列化展示，本分支不单独写文件。
            return {
                "success": True,
                "report_data": report_data,
                "generation_time": datetime.now().isoformat(),
            }

        except Exception as e:
            # 将未预期异常统一转为失败结果，供外层记录并由界面展示。
            logger.error(f"Report generation failed: {e}")
            return {"success": False, "error": str(e)}


    async def _execute_research_plan(self, plan: Dict[str, Any]) -> Dict[str, Any]:
        """Execute research tasks from the plan"""
        tasks = plan.get("tasks", [])
        if not tasks:
            raise ValueError("Research plan contains no tasks")
        handlers = {
            "deep_research": self._execute_deep_research_task,
            "deep_researcher": self._execute_deep_research_task,
            "browser": self._execute_browser_task, "browser_use": self._execute_browser_task,
            "deep_analyze": self._execute_analysis_task, "final_answer": self._execute_final_task
        }
        pending = {}
        for task in tasks:
            task_id = task.get("task_id")
            if not task_id or task_id in pending:
                raise ValueError("Task IDs must be present and unique")
            if task.get("task_type") not in handlers:
                raise ValueError(f"Unknown task type: {task.get('task_type')}")
            if not isinstance(task.get("parameters", {}), dict):
                raise ValueError(f"Invalid parameters for {task_id}")
            if not isinstance(task.get("dependencies", []), list):
                raise ValueError(f"Invalid dependencies for {task_id}")
            pending[task_id] = task
        for task in tasks:
            if any(dep not in pending for dep in task.get("dependencies", [])):
                raise ValueError(f"Unknown dependency for {task['task_id']}")
        task_results = {}
        while pending:
            ready = [task for task in pending.values()
                     if all(dep in task_results for dep in task.get("dependencies", []))]
            if not ready:
                raise ValueError("Research plan contains a dependency cycle")
            for task in ready:
                task_id = task["task_id"]
                dependencies = task.get("dependencies", [])
                failed = [dep for dep in dependencies if not task_results[dep].get("success")]
                if failed:
                    task_results[task_id] = {"success": False, "error": f"Failed dependencies: {failed}",
                                             "task_type": task["task_type"]}
                else:
                    prepared = dict(task)
                    # Include completed evidence as well as explicitly declared dependencies.
                    prepared["context"] = {
                        "research_topic": plan.get("research_topic", ""),
                        "requirements": plan.get("requirements", []),
                        "task_results": {key: value for key, value in task_results.items() if value.get("success")}
                    }
                    try:
                        task_results[task_id] = await handlers[task["task_type"]](prepared)
                    except Exception as exc:
                        task_results[task_id] = {"success": False, "error": str(exc)}
                del pending[task_id]
        return task_results

    @staticmethod
    def _normalize_browser_parameters(parameters):
        parameters = dict(parameters)
        exact = parameters.get("exact_target", False)
        if isinstance(exact, str) and exact.lower() in ("true", "false"):
            parameters["exact_target"] = exact.lower() == "true"
        elif not isinstance(exact, bool):
            raise ValueError("exact_target must be true or false")
        aliases = {"extract_content", "content_extraction", "read", "browse", "navigate"}
        if parameters.get("browser_task_type") in aliases:
            parameters["browser_task_type"] = "web_navigation"
        # Planning models sometimes describe reading as an action. It is not a form/click operation.
        for key in ("action", "actions"):
            value = parameters.get(key)
            actions = value if isinstance(value, list) else [value] if value else []
            executable = []
            for action in actions:
                if isinstance(action, str):
                    read_only = re.search(r"read|extract|scrape|collect|brows|navigat|analy[sz]|提取|读取|阅读|收集|浏览|分析", action, re.I)
                    changes_page = re.search(r"click|fill|submit|login|log in|purchase|delete|点击|填写|提交|登录|购买|删除", action, re.I)
                    if read_only and not changes_page:
                        continue
                    raise ValueError("Browser interaction requires a structured action with type and target")
                if not isinstance(action, dict):
                    raise ValueError("Browser actions must be objects")
                if action.get("type") in aliases and not action.get("target"):
                    continue
                if action.get("type") == "extract" and not action.get("target"):
                    continue
                executable.append(action)
            if executable:
                parameters[key] = executable if key == "actions" else executable[0]
            else:
                parameters.pop(key, None)
        return parameters

    async def _execute_browser_task(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """Fetch research sources with bounded rediscovery after a read-only target fails."""
        from src.agents.base_agent import Task
        from src.agents.deep_researcher_agent import SourceDiscoveryTool
        try:
            original = self._normalize_browser_parameters(task.get("parameters", {}))
        except ValueError as exc:
            return {"success": False, "task_type": "browser", "error": str(exc)}
        target = original.get("pdf_path") or original.get("url")
        actions = original.get("actions", []) + ([original["action"]] if original.get("action") else [])
        interactive = any(action.get("type") != "extract" for action in actions) or any(
            original.get(key) for key in ("form_data", "submit_button")) or original.get("browser_task_type") == "form_interaction"
        if not target and (interactive or original.get("exact_target")):
            return {"success": False, "task_type": "browser",
                    "error": "An exact-target or interactive browser task requires an explicit url or pdf_path"}
        can_discover = not interactive and not original.get("exact_target") and (
            not target or urlparse(target).scheme in ("http", "https"))
        attempts = []
        visited = set()

        async def fetch(parameters):
            url = parameters.get("pdf_path") or parameters.get("url")
            visited.add(url)
            try:
                result = await self.agents["browser"].execute_task(Task(
                    id=task["task_id"], type="browser", description=task["description"], parameters=parameters))
                if not result.success:
                    raise ValueError(result.error or "Browser task failed")
                data = result.result
                content = data.get("full_text") or data.get("content") or data.get("result_content")
                if not content and any(data.get("extracted_data", {}).values()):
                    content = json.dumps(data["extracted_data"], ensure_ascii=False)
                if not content or not content.strip():
                    raise ValueError("Browser task returned no usable content")
                source_url = data.get("url") or url
                relevant = await self._review_sources(task, [{"url": source_url,
                    "title": data.get("title", ""), "content": content, "content_length": len(content)}])
                if not relevant:
                    raise ValueError("Browser page does not contain relevant research evidence")
                source = {"url": source_url, "title": data.get("title", source_url), "snippet": content[:300]}
                return {"success": True, "task_type": "browser", "data": {
                    "results": [source], "structured_data": {
                        "browser_result": data, "sources": [source], "source_attempts": list(attempts),
                        "requested_target": target,
                        "extracted_content": [{"url": source_url, "content": content, "content_length": len(content)}]}}}
            except Exception as exc:
                attempts.append({"url": url, "error": str(exc)})
                return None

        if target:
            result = await fetch(original)
            if result:
                return result
        if can_discover:
            query = original.get("query") or task["description"]
            topic = task.get("context", {}).get("research_topic", "")
            if topic:
                query = f"{topic} {query}"
            # Preserve the requested authority and document format when replacing a broken link.
            domain = (urlparse(target).hostname or "").removeprefix("www.") if target else ""
            if domain:
                query += f" site:{domain}"
            pdf_only = original.get("browser_task_type") == "pdf_analysis" or bool(original.get("pdf_path")) or bool(target and urlparse(target).path.lower().endswith(".pdf"))
            if pdf_only:
                query += " filetype:pdf"
            try:
                # A targetless browser task should consume the discovery stage's URLs first.
                if not target:
                    prior = task.get("context", {}).get("task_results", {})
                    candidates = []
                    for task_id in task.get("dependencies") or prior:
                        result = prior.get(task_id, {})
                        if not result.get("success"):
                            continue
                        data = result.get("data", {})
                        candidates.extend(data.get("structured_data", {}).get("extracted_content", []))
                        candidates.extend(data.get("results", []))
                    tried = 0
                    for source in candidates:
                        candidate = source.get("url", "")
                        parsed = urlparse(candidate)
                        if parsed.scheme not in ("http", "https") or candidate in visited:
                            continue
                        if pdf_only and not parsed.path.lower().endswith(".pdf"):
                            continue
                        parameters = dict(original, url=candidate)
                        if parsed.path.lower().endswith(".pdf"):
                            parameters["browser_task_type"] = "pdf_analysis"
                        result = await fetch(parameters)
                        if result:
                            return result
                        tried += 1
                        if tried >= 3:
                            break
                discovery = await asyncio.to_thread(SourceDiscoveryTool().forward, query, max_results=5)
                if discovery.get("error"):
                    raise ValueError(discovery["error"])
                for source in discovery.get("sources", [])[:5]:
                    candidate = source.get("url", "")
                    parsed = urlparse(candidate)
                    host = parsed.hostname or ""
                    if parsed.scheme not in ("http", "https") or candidate in visited:
                        continue
                    if domain and host != domain and not host.endswith("." + domain):
                        continue
                    parameters = dict(original)
                    parameters.pop("pdf_path", None)
                    parameters["url"] = candidate
                    if pdf_only:
                        parameters["browser_task_type"] = "pdf_analysis"
                    elif parsed.path.lower().endswith(".pdf"):
                        parameters["browser_task_type"] = "pdf_analysis"
                    result = await fetch(parameters)
                    if result:
                        return result
            except Exception as exc:
                attempts.append({"query": query, "error": str(exc)})
        detail = "; ".join(item["error"] for item in attempts) or "No usable browser targets found"
        return {"success": False, "task_type": "browser", "error": detail, "source_attempts": attempts}

    async def _review_sources(self, task, items):
        return await review_sources(self.model, task.get('context', {}).get('research_topic') or task['description'],
                                    task['description'], items, getattr(self, 'trace', None))

    async def _execute_deep_research_task(self, task: Dict[str, Any]) -> Dict[str, Any]:
        from src.agents.deep_researcher_agent import SourceDiscoveryTool, ContentExtractionTool
        topic = task.get('context', {}).get('research_topic') or task['description']
        queries = search_queries(task.get('parameters', {}).get('query') or topic, topic)
        seen, accepted, search_attempts = set(), [], []
        try:
            for query in queries:
                discovery = await asyncio.to_thread(SourceDiscoveryTool().forward, query, max_results=10)
                if getattr(self, 'trace', None):
                    self.trace.record('search', {'task_id': task['task_id'], 'query': query, 'result': discovery})
                search_attempts.append({'query': query, 'error': discovery.get('error')})
                sources = discovery.get('sources', [])
                urls = [item['url'] for item in sources if item.get('url') and item['url'] not in seen][:5]
                seen.update(urls)
                if not urls:
                    continue
                extraction = await asyncio.to_thread(ContentExtractionTool().forward, urls, max_content_length=12000)
                contents = extraction.get('extracted_content', [])
                if getattr(self, 'trace', None):
                    self.trace.record('extraction', {'task_id': task['task_id'], 'content': contents})
                accepted.extend(await self._review_sources(task, contents))
                # A second independent source provides cross-checking; otherwise try a different query.
                accepted = list({item['url']: item for item in accepted}.values())
                if len(accepted) >= 2:
                    break
            if not accepted:
                raise ValueError('No relevant research content found after supplementary searches')
            sources = [{'url': item['url'], 'title': item.get('title', item['url']),
                        'domain': urlparse(item['url']).netloc, 'snippet': item['content'][:300]} for item in accepted]
            return {'success': True, 'task_type': 'deep_research', 'data': {
                'agent_response': f'Collected {len(accepted)} relevant research sources', 'results': sources,
                'structured_data': {'sources': sources, 'extracted_content': accepted,
                                    'search_attempts': search_attempts, 'total_sources': len(accepted),
                                    'total_content_length': sum(len(item['content']) for item in accepted)}}}
        except Exception as exc:
            return {'success': False, 'task_type': 'deep_research', 'error': str(exc), 'search_attempts': search_attempts}

    async def _execute_analysis_task(self, task: Dict[str, Any]) -> Dict[str, Any]:
        return await self._write_research_task(task, "analysis")

    async def _execute_final_task(self, task: Dict[str, Any]) -> Dict[str, Any]:
        return await self._write_research_task(task, "final")

    async def _write_research_task(self, task: Dict[str, Any], stage: str) -> Dict[str, Any]:
        try:
            document = await ResearchWriter(self.model, getattr(self, 'trace', None)).write(task, stage)
            content = ResearchWriter.render_text(document)
            data = {"agent_response": content, "research_document": document,
                    "results": [], "structured_data": {}}
            if stage == "final":
                data.update(report_content=content, format="text")
            return {"success": True, "data": data, "task_type": stage,
                    "task_description": task["description"]}
        except Exception as exc:
            return {"success": False, "error": str(exc), "task_type": stage}

    def _extract_chart_data(self, task_results: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Extract chart data from task results"""
        chart_data = []
        
        for task_id, result in task_results.items():
            # Skip if result is not a dict
            if not isinstance(result, dict):
                continue

            if result.get("success") and result.get("data"):
                data = result["data"]
                
                # Look for chart-able data
                if isinstance(data, dict):
                    # Handle search results (from deep_research task)
                    if "results" in data and data["results"]:
                        # Count sources by domain
                        source_counts = {}
                        for r in data["results"]:
                            # Use 'domain' field from SourceDiscoveryTool, fallback to 'source'
                            source = r.get("domain") or r.get("source", "Unknown")
                            source_counts[source] = source_counts.get(source, 0) + 1

                        chart_data.append({
                            "id": f"chart_{task_id}",
                            "type": "bar",
                            "title": "Search Results by Engine",
                            "labels": list(source_counts.keys()),
                            "datasets": [{
                                "label": "Number of Results",
                                "data": list(source_counts.values())
                            }],
                            "description": f"Distribution of {len(data['results'])} search results across {len(source_counts)} domains"
                        })

                    # Handle structured data (from browser task)
                    if "structured_data" in data:
                        structured = data["structured_data"]
                        if structured.get("headings"):
                            chart_data.append({
                                "id": f"chart_{task_id}_headings",
                                "type": "pie",
                                "title": "Content Structure",
                                "labels": [h[:20] + "..." if len(h) > 20 else h for h in structured["headings"]],
                                "values": [1 for _ in structured["headings"]],
                                "description": "Distribution of content headings"
                            })
        
        return chart_data
    
    def _extract_citations(self, task_results: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Extract citations from task results"""
        citations = []
        
        for task_id, result in task_results.items():
            # Skip if result is not a dict
            if not isinstance(result, dict):
                continue

            if result.get("success") and result.get("data"):
                data = result["data"]
                
                if isinstance(data, dict) and "results" in data:
                    for search_result in data["results"]:
                        citations.append({
                            "type": "website",
                            "title": search_result.get("title", ""),
                            "url": search_result.get("url", ""),
                            "source": search_result.get("source", ""),
                            "access_date": datetime.now().strftime("%Y-%m-%d")
                        })
        
        return citations
    
    def _extract_data_sources(self, task_results: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Extract data sources from task results"""
        sources = []
        
        for task_id, result in task_results.items():
            # Skip if result is not a dict
            if not isinstance(result, dict):
                continue

            if result.get("success") and result.get("data"):
                data = result["data"]
                
                if isinstance(data, dict) and "results" in data:
                    sources_used = set()
                    for search_result in data["results"]:
                        source = search_result.get("source", "Unknown")
                        if source not in sources_used:
                            sources_used.add(source)
                            sources.append({
                                "name": source.title(),
                                "description": f"Search results from {source}",
                                "url": search_result.get("url", "")
                            })
        
        return sources
    
    def _create_report_sections(self, task_results: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Create semantic report sections from structured research results."""
        return render_sections(task_results)

    def _extract_key_metrics(self, task_results: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Extract key metrics from task results"""
        metrics = []
        
        # Count total results
        total_results = 0
        total_sources = 0
        
        for task_id, result in task_results.items():
            if result.get("success") and result.get("data"):
                data = result["data"]
                if isinstance(data, dict) and "results" in data:
                    total_results += len(data["results"])
                    sources = set(r.get("source", "Unknown") for r in data["results"])
                    total_sources += len(sources)
        
        metrics.extend([
            {
                "name": "Total Results",
                "value": str(total_results),
                "description": "Number of search results found"
            },
            {
                "name": "Data Sources",
                "value": str(total_sources),
                "description": "Number of unique data sources"
            },
            {
                "name": "Tasks Completed",
                "value": str(len(task_results)),
                "description": "Research tasks executed"
            }
        ])
        
        return metrics
    
    def _extract_risk_assessment(self, task_results: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Extract risk assessment from task results"""
        # This is a simplified risk assessment
        return [
            {
                "category": "Data Quality",
                "level": "low",
                "impact": "High-quality search results from multiple sources",
                "mitigation": "Cross-reference information from multiple sources"
            },
            {
                "category": "Source Reliability",
                "level": "medium",
                "impact": "Some sources may have bias or inaccuracies",
                "mitigation": "Verify critical information with authoritative sources"
            },
            {
                "category": "Timeliness",
                "level": "low",
                "impact": "Current search results provide up-to-date information",
                "mitigation": "Regular updates and monitoring of key sources"
            }
        ]
    
    def _extract_recommendations(self, task_results: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Extract recommendations from task results"""
        return [
            {
                "title": "Continuous Monitoring",
                "description": "Set up ongoing monitoring of key financial indicators and market trends",
                "priority": "High",
                "timeline": "Immediate"
            },
            {
                "title": "Source Diversification",
                "description": "Expand data sources to include additional financial databases and news sources",
                "priority": "Medium",
                "timeline": "1-2 weeks"
            },
            {
                "title": "Regular Updates",
                "description": "Schedule regular report updates to maintain data freshness and accuracy",
                "priority": "Medium",
                "timeline": "Ongoing"
            }
        ]
    
    def _generate_executive_summary(self, plan: Dict[str, Any], task_results: Dict[str, Any]) -> str:
        """Generate executive summary"""
        for result in reversed(list(task_results.values())):
            if result.get("success") and result.get("task_type") == "final":
                return escape(result.get("data", {}).get("research_document", {}).get("summary", ""))
        return ""


# Create global app instance
app = DeepReportApp()

async def generate_report_interface(research_topic, requirements, output_format, model):
    """处理 Gradio 的报告生成请求，并将结果转换为界面组件更新。

    Args:
        research_topic (str): 用户输入的研究主题。
        requirements (str): 研究要求，多条要求用换行分隔。
        output_format (str): 输出格式；界面提供 "html" 和 "json"。
        model (str): 用于本次报告生成的模型标识。

    Returns:
        tuple: 按按钮事件 outputs 的顺序返回三个值：
            HTML 展示组件的更新、状态文本、JSON/错误文本组件的更新。
            每个分支都会清空并隐藏不使用的展示组件，避免残留上次结果。

    Notes:
        生成失败时优先展示失败任务详情；未预期异常转为错误文本，
        不继续向 Gradio 事件处理层抛出。
    """
    try:
        # 等待后端完成生成；本函数只负责将结果映射到界面。
        result = await app.generate_report(research_topic, requirements, output_format, model)
        if not result.get("success"):
            # 优先展示各失败任务的原因，没有任务详情时回退到整体错误。
            detail = result.get("failed_tasks") or result.get("error", "Unknown error")
            return gr.update(value="", visible=False), "Report generation failed", gr.update(
                value=json.dumps(detail, ensure_ascii=False, indent=2), visible=True)
        if output_format == "html":
            # 完整报告在 iframe 内加载，独立执行图表/标签脚本并隔离报告 CSS。
            # srcdoc 按属性值转义；不授予同源权限，避免报告访问外层界面。
            report_preview = (
                '<iframe title="HTML 报告预览" '
                'sandbox="allow-scripts allow-popups allow-popups-to-escape-sandbox allow-downloads" '
                'style="display:block;width:100%;height:85vh;min-height:600px;'
                'border:1px solid #e3e8f0;border-radius:12px;background:white;" '
                f'srcdoc="{escape(result["report_html"], quote=True)}"></iframe>'
            )
            return gr.update(value=report_preview, visible=True), (
                f"Report generated successfully! File saved to: {result.get('report_path', '')}"
            ), gr.update(value="", visible=False)
        # 非 HTML 结果按 JSON 展示；保留中文字符并缩进，方便阅读。
        return gr.update(value="", visible=False), "Report data generated successfully!", gr.update(
            value=json.dumps(result["report_data"], ensure_ascii=False, indent=2), visible=True)
    except Exception as exc:
        # 兜底处理调用、结果字段读取或序列化等异常，仍返回完整的界面更新。
        return gr.update(value="", visible=False), "Report generation failed", gr.update(value=str(exc), visible=True)


def create_gradio_interface():
    """Create Gradio interface"""
    
    # Define interface components
    with gr.Blocks(title="DeepReport - AI Financial Research", theme=gr.themes.Soft()) as demo:
        gr.Markdown("""
        # 📊 DeepReport
        ### AI-Powered Financial Research and Report Generation
        
        Generate comprehensive financial research reports using advanced AI agents and multi-engine search capabilities.
        """)
        
        with gr.Row():
            with gr.Column(scale=2):
                # Input section
                gr.Markdown("## 📝 Research Parameters")
                
                research_topic = gr.Textbox(
                    label="Research Topic",
                    placeholder="Enter your financial research topic...",
                    lines=2
                )
                
                requirements = gr.Textbox(
                    label="Research Requirements",
                    placeholder="Enter specific requirements (one per line)...",
                    lines=4
                )
                
                with gr.Row():
                    output_format = gr.Radio(
                        choices=["html", "json"],
                        value="html",
                        label="Output Format"
                    )
                    
                    model = gr.Dropdown(
                        choices=["gpt-4o", "gpt-4", "claude-3-sonnet", "claude-3-opus"],
                        value="gpt-4o",
                        label="AI Model"
                    )
                
                generate_btn = gr.Button(
                    "🚀 Generate Report",
                    variant="primary",
                    size="lg"
                )
            
            with gr.Column(scale=1):
                # Status section
                gr.Markdown("## 📊 System Status")
                
                status_output = gr.Textbox(
                    label="Status",
                    interactive=False,
                    lines=2
                )
        
        # Results section
        gr.Markdown("## 📄 Generated Report")
        
        with gr.Row():
            # HTML output (hidden by default)
            html_output = gr.HTML(
                visible=False,
                label="HTML Report"
            )
            
            # Text output (for JSON/errors)
            text_output = gr.Textbox(
                label="Report Output",
                interactive=False,
                lines=20,
                visible=False
            )
        
        # Event handlers
        generate_btn.click(
            fn=generate_report_interface,
            inputs=[research_topic, requirements, output_format, model],
            outputs=[html_output, status_output, text_output]
        )
        
        # Example section
        gr.Markdown("""
        ## 💡 Example Topics
        
        - Tesla Inc. (TSLA) Q4 2023 Financial Performance Analysis
        - Cryptocurrency Market Trends and Investment Opportunities
        - Renewable Energy Sector Growth Prospects 2024
        - Federal Reserve Monetary Policy Impact on Tech Stocks
        - Artificial Intelligence in Financial Services Market Analysis
        
        ## 🔧 System Features
        
        - **SmolAgents Framework**: Advanced multi-agent AI orchestration
        - **Specialized Agents**: DeepResearcher, Browser, DeepAnalyze, FinalAnswer
        - **Multi-Engine Search**: Serper, Metaso, Sogou integration
        - **MCP Protocol**: FastMCP support for local/remote tool connections
        - **Rich Reports**: HTML reports with interactive charts and professional citations
        - **Data Visualization**: Chart.js integration for dynamic charts
        - **Citation Management**: APA, MLA, Chicago, Harvard citation styles
        """)
    
    return demo

# Create and launch the interface
if __name__ == "__main__":
    demo = create_gradio_interface()
    
    # Launch the app
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=True,
        debug=True
    )
