"""Render research documents as readable, escaped HTML sections."""
from html import escape
import re
from urllib.parse import urlparse


def source_link(url, label=None):
    url = str(url or '')
    parsed = urlparse(url)
    if parsed.scheme not in ('http', 'https') or not parsed.netloc:
        return ''
    return (f'<a class="source-link" href="{escape(url, quote=True)}" '
            f'target="_blank" rel="noopener noreferrer">'
            f'{escape(str(label or parsed.netloc.removeprefix("www.")))} ↗</a>')


def paragraphs(text):
    if isinstance(text, list):
        text = '\n\n'.join(map(str, text))
    return ''.join(f'<p>{escape(p.strip())}</p>'
                   for p in re.split(r'\n\s*\n', str(text or '')) if p.strip())


def unique_sources(items):
    seen = set()
    for item in items:
        key = (item.get('url'), item.get('content', item.get('snippet', '')))
        if key not in seen:
            seen.add(key)
            yield item


def render_sections(task_results):
    sections = [{
        'id': 'methodology', 'title': 'Research Methodology',
        'content': '''<div class="reading-column"><p class="section-intro">从资料检索到结论整理，以下步骤构成本报告的研究流程。</p>
        <ol class="method-steps">
        <li><h3>检索与筛选</h3><p>围绕研究主题检索资料，并筛选与问题相关的来源。</p></li>
        <li><h3>提取与核对</h3><p>提取网页或文档正文，将研究论述与来源证据关联。</p></li>
        <li><h3>分析与归纳</h3><p>依据已收集材料分主题分析，记录结论及证据局限。</p></li>
        <li><h3>报告与引用</h3><p>汇总分析形成最终报告，保留来源链接供进一步查阅。</p></li>
        </ol></div>'''}]
    search, extracted = [], []
    for result in task_results.values():
        if result.get('success') and result.get('task_type') in ('deep_research', 'browser'):
            data = result.get('data', {})
            search.extend(data.get('results', []))
            extracted.extend(data.get('structured_data', {}).get('extracted_content', []))
    search, extracted = list(unique_sources(search)), list(unique_sources(extracted))
    titles = {item.get('url'): item.get('title') for item in search}
    if extracted:
        cards = []
        for i, item in enumerate(extracted, 1):
            text, url = str(item.get('content', '')), item.get('url', '')
            title = item.get('title') or titles.get(url) or f'来源正文 {i:02d}'
            remainder = ('<details class="source-details"><summary>展开完整提取内容</summary>'
                         + paragraphs(text) + '</details>') if len(text) > 500 else ''
            cards.append(f'<article class="source-card"><div class="source-meta">来源 {i:02d} · {len(text):,} 字符</div>'
                         f'<h3>{escape(title)}</h3>{source_link(url)}'
                         f'<div class="source-preview">{paragraphs(text[:500] + ("…" if len(text)>500 else ""))}</div>{remainder}</article>')
        sections.append({'id': 'extracted_content', 'title': 'Extracted Research Content',
                         'content': f'<p class="section-intro">共 {len(extracted)} 份提取材料；展开卡片可阅读完整提取正文。</p><div class="source-stack">' + ''.join(cards) + '</div>'})
    if search:
        cards = []
        for i, item in enumerate(search, 1):
            cards.append(f'<article class="source-card"><div class="source-meta">检索结果 {i:02d}</div>'
                         f'<h3>{escape(str(item.get("title") or "未命名来源"))}</h3>'
                         f'{paragraphs(item.get("snippet", ""))}{source_link(item.get("url"), "查看来源")}</article>')
        sections.append({'id': 'search_results', 'title': 'Search Results',
                         'content': f'<p class="section-intro">共 {len(search)} 条检索记录，供查阅与交叉核对。</p><div class="source-grid">' + ''.join(cards) + '</div>'})
    for task_type, section_id, title in [('analysis', 'research_analysis', 'Analysis'), ('final', 'final_report', 'Final Report')]:
        results = [r for r in task_results.values() if r.get('success') and r.get('task_type') == task_type]
        blocks = []
        for i, result in enumerate(results, 1):
            data = result.get('data', {})
            document = data.get('research_document') or {}
            content = data.get('report_content') or data.get('agent_response', '')
            if not document and not content:
                continue
            block = '<article class="research-document reading-column">'
            if len(results) > 1:
                block += f'<div class="source-meta">专题 {i:02d}</div><details class="task-scope"><summary>查看本专题研究范围</summary><h3 class="task-heading">{escape(result.get("task_description") or f"{title} {i}")}</h3></details>'
            if document:
                block += '<div class="research-summary"><div class="source-meta">核心摘要</div>' + paragraphs(document.get('summary')) + '</div>'
                for j, section in enumerate(document.get('sections', []), 1):
                    block += f'<section class="research-subsection"><h4><span class="section-number">{j:02d}</span>{escape(section.get("heading", ""))}</h4>' + paragraphs(section.get('body'))
                    urls = dict.fromkeys(e.get('url') for e in section.get('evidence', []) if e.get('url'))
                    block += '<div class="evidence-links"><span>参考来源</span>' + ''.join(source_link(url) for url in urls) + '</div></section>'
                if document.get('limitations'):
                    block += '<aside class="research-limitations"><h4>研究局限</h4><ul>' + ''.join(f'<li>{escape(str(x))}</li>' for x in document['limitations']) + '</ul></aside>'
            else:
                block += paragraphs(content)
            blocks.append(block + '</article>')
        if blocks:
            sections.append({'id': section_id, 'title': title, 'content': ''.join(blocks)})
    return sections
