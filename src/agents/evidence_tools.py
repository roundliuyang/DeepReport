"""只读证据工具：模型按编号读取正文，不能提交自造的引用正文。"""
from smolagents import Tool


class InspectEvidenceTool(Tool):
    name = 'inspect_evidence'
    description = 'Read original source passages by their evidence IDs before drawing conclusions.'
    inputs = {'evidence_ids': {'type': 'array', 'description': 'IDs from the supplied evidence index'}}
    output_type = 'object'

    def __init__(self, catalog):
        super().__init__()
        self.catalog = catalog

    def forward(self, evidence_ids: list) -> dict:
        """返回每个编号对应的 URL 和原文；未知编号返回错误。"""
        unknown = [key for key in evidence_ids if key not in self.catalog]
        if unknown:
            return {'error': f'Unknown evidence IDs: {unknown}'}
        return {key: self.catalog[key] for key in evidence_ids}
