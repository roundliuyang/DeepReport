"""研究规划 Agent；规划也通过原生 final_answer 工具提交。"""
from .base_agent import BaseAgent


class PlanningAgent(BaseAgent):
    def __init__(self, model, memory=None):
        super().__init__('PlanningAgent', model, instructions=(
            'Plan research tasks. Return a JSON object via final_answer. '
            'Never invent URLs, selectors, documents or numeric research findings.'
        ))
