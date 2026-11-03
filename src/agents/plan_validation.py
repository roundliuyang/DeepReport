"""校验任务图与来源约束，不替模型生成研究计划。"""
import re

TASK_TYPES = {
    'deep_research': 'deep_researcher', 'deep_researcher': 'deep_researcher',
    'browser': 'browser', 'browser_use': 'browser',
    'deep_analyze': 'analysis', 'final_answer': 'final',
}


def supplied_targets(text):
    return set(re.findall(r'https?://[^\s<>"\']+', text))


def validate_plan(plan, user_text):
    """验证任务字段、依赖图和用户指定 URL；返回标准化类型的计划。"""
    tasks = plan.get('tasks')
    if not isinstance(tasks, list) or not 3 <= len(tasks) <= 12:
        raise ValueError('Plan needs 3-12 tasks covering research, analysis and final writing')
    by_id = {}
    allowed = supplied_targets(user_text)
    for task in tasks:
        if not isinstance(task, dict):
            raise ValueError('Each task must be an object')
        key = task.get('task_id')
        if not isinstance(key, str) or not re.fullmatch(r'[A-Za-z0-9_-]+', key) or key in by_id:
            raise ValueError('Task IDs must be nonempty unique strings')
        if task.get('task_type') not in TASK_TYPES:
            raise ValueError('Unknown task_type; use deep_researcher, browser_use, deep_analyze, final_answer')
        if not isinstance(task.get('description'), str) or not task['description'].strip():
            raise ValueError(f'{key}: description is required')
        params = task.setdefault('parameters', {})
        deps = task.setdefault('dependencies', [])
        if not isinstance(params, dict) or not isinstance(deps, list) or any(not isinstance(d, str) for d in deps):
            raise ValueError(f'{key}: invalid parameters or dependencies')
        for field in ('url', 'pdf_path'):
            if params.get(field) and params[field] not in allowed:
                raise ValueError('Do not invent targets; use parameters.query to discover URLs at execution time')
        exact = params.get('exact_target', False)
        if not isinstance(exact, bool) or (exact and not (params.get('url') or params.get('pdf_path'))):
            raise ValueError('exact_target requires a boolean and a user-supplied target')
        by_id[key] = task
    ancestors = {}

    def visit(key, path):
        if key not in by_id:
            raise ValueError(f'Unknown dependency: {key}')
        if key in path:
            raise ValueError('Research plan contains a dependency cycle')
        if key not in ancestors:
            found = set()
            for dep in by_id[key]['dependencies']:
                found.add(dep)
                found.update(visit(dep, path | {key}))
            ancestors[key] = found
        return ancestors[key]

    for key in by_id:
        visit(key, set())
    analyses = {k for k, t in by_id.items() if TASK_TYPES[t['task_type']] == 'analysis'}
    finals = [k for k, t in by_id.items() if TASK_TYPES[t['task_type']] == 'final']
    if not analyses or len(finals) != 1:
        raise ValueError('Plan requires analysis and exactly one final_answer task')
    for key in analyses:
        if not any(TASK_TYPES[by_id[d]['task_type']] in ('deep_researcher', 'browser') for d in ancestors[key]):
            raise ValueError('Every analysis must depend on research or browser data collection')
    if not analyses <= ancestors[finals[0]]:
        raise ValueError('final_answer must depend on all analysis tasks')
    if set(by_id) - {finals[0]} - ancestors[finals[0]]:
        raise ValueError('Every task must contribute to the final report through dependencies')
    return plan
