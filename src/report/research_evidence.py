"""证据编号、引用校验与研究正文格式化，不调用模型。"""
import hashlib
import json
import re


class ResearchEvidence:
    def __init__(self, model, trace=None):
        self.model = model
        self.trace = trace

    @staticmethod
    def subject_type(topic):
        topic = topic.lower()
        if any(word in topic for word in (
            '宏观', '经济展望', '中国经济', '经济增长', 'gdp', 'macroeconomic',
            'economic outlook', 'economy', 'inflation', '货币政策', '财政政策'
        )):
            return 'macroeconomic'
        return 'topic_specific'

    @staticmethod
    def collect_evidence(task):
        evidence = {}
        for result in task.get('context', {}).get('task_results', {}).values():
            if not result.get('success'):
                continue
            items = result.get('data', {}).get('structured_data', {}).get('extracted_content', [])
            for item in items:
                if item.get('url') and item.get('content', '').strip():
                    evidence.setdefault(item['url'], []).append(item['content'])
        return {url: '\n'.join(dict.fromkeys(parts)) for url, parts in evidence.items()}

    @staticmethod
    def index_evidence(evidence):
        """Give source passages stable IDs; the model selects IDs instead of copying quotes."""
        catalog = {}
        for url, content in evidence.items():
            for start in range(0, len(content), 1200):
                passage = content[start:start + 1200]
                if not passage.strip():
                    continue
                digest = hashlib.sha256((url + '\0' + passage).encode('utf-8')).hexdigest()[:20]
                catalog['evidence_' + digest] = {'url': url, 'quote': passage}
        return catalog

    @staticmethod
    def validate(document, evidence, subject, stage, catalog=None):
        if not isinstance(document, dict) or document.get('subject_type') != subject:
            raise ValueError('The research subject classification is missing or incorrect')
        summary = document.get('summary')
        if not isinstance(summary, str) or not summary.strip():
            raise ValueError('summary must be a nonempty string')
        sections = document.get('sections')
        if not isinstance(sections, list) or not sections:
            raise ValueError('sections must be a nonempty list of research sections')
        headings = set()
        for section in sections:
            if not isinstance(section, dict):
                raise ValueError('Research sections must be objects')
            heading, body = section.get('heading'), section.get('body')
            if not isinstance(heading, str) or not heading.strip() or heading in headings:
                raise ValueError('Research sections need distinct, nonempty headings')
            headings.add(heading)
            # Lists of paragraphs are valid model output; normalize without dropping content.
            if isinstance(body, list) and all(isinstance(paragraph, str) for paragraph in body):
                body = '\n\n'.join(body)
                section['body'] = body
            if not isinstance(body, str) or not body.strip():
                raise ValueError(f"Section {heading!r}: body must contain text")
            if re.search(r'报告.{0,12}(已生成|成功生成)|report.{0,25}(successfully generated|has been generated)', body, re.I):
                raise ValueError('Return research conclusions, not report-generation status')
            references = section.get('evidence')
            if not isinstance(references, list) or not references:
                raise ValueError('Each research section needs supporting source evidence')
            for reference in references:
                if not isinstance(reference, dict):
                    raise ValueError('Evidence references must be objects')
                if 'evidence_id' in reference:
                    evidence_id = reference['evidence_id']
                    # Some models omit the display prefix while preserving the entire hash.
                    # Resolve only exact known IDs; never guess a source for an unknown reference.
                    if isinstance(evidence_id, str) and catalog is not None and evidence_id not in catalog:
                        if 'evidence_' + evidence_id in catalog:
                            evidence_id = 'evidence_' + evidence_id
                    if not isinstance(evidence_id, str) or catalog is None or evidence_id not in catalog:
                        raise ValueError('Evidence ID is not in the supplied source passages; select an ID from sources')
                    # Persist only the original passage, never a model-written or translated quote.
                    reference.clear()
                    reference.update(evidence_id=evidence_id, **catalog[evidence_id])
                    continue
                # Compatibility for previously generated documents with explicit quotations.
                url, quote = reference.get('url'), reference.get('quote')
                if not isinstance(url, str) or url not in evidence:
                    raise ValueError('Citation is not in the collected source content')
                normalize = lambda text: ' '.join(text.split())
                if not isinstance(quote, str) or len(quote.strip()) < 12 or normalize(quote) not in normalize(evidence[url]):
                    raise ValueError('Evidence quote must be a verbatim excerpt from the cited source')
        limitations = document.get('limitations')
        if not isinstance(limitations, list) or any(not isinstance(item, str) for item in limitations):
            raise ValueError('State evidence limitations as a list')
        if subject == 'macroeconomic':
            # Source passages can discuss companies; classify the analysis, not quoted evidence.
            text = '\n'.join([summary] + [s['body'] for s in sections] + limitations).lower()
            if any(term in text for term in ('负债权益比', '流动比率', '员工流失率', 'debt-to-equity', 'current ratio', 'employee turnover')):
                raise ValueError('Do not use company solvency or staffing indicators to judge a national economy')
        return document

    @staticmethod
    def render_text(document):
        parts = [document['summary']]
        for section in document['sections']:
            parts.extend([section['heading'], section['body']])
            parts.append('\n'.join(dict.fromkeys(ref['url'] for ref in section['evidence'])))
        if document['limitations']:
            parts.extend(['证据局限 / Evidence limitations', '\n'.join(document['limitations'])])
        return '\n\n'.join(parts)
