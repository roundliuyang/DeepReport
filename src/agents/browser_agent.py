"""由 Agent 选择真实网页导航或 PDF 阅读工具。"""
import os
import tempfile
from typing import Dict, Any
import logging
import fitz
from smolagents import Tool
from playwright.sync_api import sync_playwright
from .base_agent import BaseAgent
from .deep_researcher_agent import SourceDiscoveryTool, ContentExtractionTool

logger = logging.getLogger(__name__)


class WebNavigationTool(Tool):
    name = 'navigate_web'
    description = 'Read a discovered web page in a browser, including rendered text and tables.'
    inputs = {'url': {'type': 'string', 'description': 'A user-supplied or discovered page URL'}}
    output_type = 'object'

    def forward(self, url: str) -> Dict[str, Any]:
        """在工具线程中创建并关闭浏览器，不跨事件循环共享会话。"""
        try:
            with sync_playwright() as playwright:
                browser = playwright.chromium.launch(headless=True)
                try:
                    page = browser.new_page()
                    response = page.goto(url, wait_until='domcontentloaded', timeout=30000)
                    if response and response.status >= 400:
                        return {'error': f'HTTP {response.status}', 'url': url}
                    root = page.locator('article, main, [role="main"]')
                    if not root.count():
                        root = page.locator('body')
                    content = root.first.inner_text(timeout=10000)[:18000]
                    if len(content.strip()) < 200:
                        return {'error': 'Page contains too little readable content', 'url': url}
                    return {'url': page.url, 'requested_url': url, 'title': page.title(),
                            'content': content,
                            'tables': page.locator('table').all_inner_texts()[:10]}
                finally:
                    browser.close()
        except Exception as exc:
            return {'error': str(exc), 'url': url}


class PDFAnalysisTool(Tool):
    """Tool for analyzing PDF documents"""
    
    def __init__(self):
        super().__init__()
        self.name = "analyze_pdf"
        self.description = "Analyze PDF documents for financial data and content"
        self.inputs = {
            "pdf_path": {
                "type": "string",
                "description": "Path or URL to PDF document"
            },
            "extract_tables": {
                "type": "boolean",
                "description": "Whether to extract tables from PDF",
                "nullable": True
            },
            "extract_charts": {
                "type": "boolean",
                "description": "Whether to extract chart information",
                "nullable": True
            }
        }
        self.output_type = "object"
    
    def forward(self, pdf_path: str, extract_tables: bool = True, extract_charts: bool = True) -> Dict[str, Any]:
        """Analyze PDF document content (synchronous)"""
        try:
            result = self._analyze_pdf_sync(pdf_path, extract_tables, extract_charts)
            result.update(url=pdf_path, content=result['full_text'][:18000],
                          title=result.get('metadata', {}).get('title') or pdf_path)
            return result
        except Exception as e:
            logger.error(f"PDF analysis failed: {e}")
            return {"error": str(e)}
    
    def _analyze_pdf_sync(self, pdf_path: str, extract_tables: bool = True, extract_charts: bool = True) -> Dict[str, Any]:
        """Synchronous PDF analysis"""
        import requests
        downloaded = None
        try:
            if pdf_path.startswith(("http://", "https://")):
                response = requests.get(pdf_path, timeout=30)
                response.raise_for_status()
                with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as handle:
                    handle.write(response.content)
                    downloaded = handle.name
                pdf_path = downloaded
            with fitz.open(pdf_path) as doc:
                analysis = {"metadata": doc.metadata, "page_count": len(doc), "sections": [],
                            "tables": [], "charts": [], "financial_data": []}
                texts = []
                for index, page in enumerate(doc):
                    text = page.get_text()
                    texts.append(text)
                    analysis["sections"].append({"page": index + 1, "content": text})
                    if extract_tables:
                        for table in page.find_tables().tables:
                            analysis["tables"].append({"page": index + 1, "rows": table.row_count,
                                                       "cols": table.col_count, "data": table.extract()})
                analysis["full_text"] = "\n".join(texts)
                analysis["word_count"] = len(analysis["full_text"].split())
                return analysis
        finally:
            if downloaded:
                os.unlink(downloaded)


class BrowserAgent(BaseAgent):
    def __init__(self, model, memory=None):
        super().__init__('BrowserAgent', model, tools=[
            SourceDiscoveryTool(), WebNavigationTool(), PDFAnalysisTool(), ContentExtractionTool()
        ], instructions=(
            'Read relevant discovered URLs using navigate_web, or analyze_pdf for PDFs. '
            'For ordinary static extraction you may use extract_content. '
            'Do not invent URLs or selectors. If a source is inaccessible, try a discovered '
            'alternative or search again unless the user requires that exact document. '
            'Select only relevant pages actually read by a tool. '
            'Return selected source URLs and relevance reasons through final_answer.'
        ))
