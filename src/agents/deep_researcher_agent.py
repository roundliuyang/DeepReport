"""
Deep Researcher Agent for filtering high-quality data sources using SmolAgents
"""

import logging
import requests
import re
from typing import Dict, Any, List
from urllib.parse import urlparse
from smolagents import Tool
import serpapi
from bs4 import BeautifulSoup

from .base_agent import BaseAgent

logger = logging.getLogger(__name__)

class SourceDiscoveryTool(Tool):
    """Tool for discovering high-quality data sources"""
    
    def __init__(self):
        super().__init__()
        self.name = "discover_sources"
        self.description = "Search for high-quality financial data sources and research materials"
        self.inputs = {
            "query": {
                "type": "string",
                "description": "Search query for financial data sources"
            },
            "source_type": {
                "type": "string",
                "description": "Type of sources to find (news, reports, data, analysis)",
                "nullable": True
            },
            "max_results": {
                "type": "integer",
                "description": "Maximum number of results to return",
                "nullable": True
            }
        }
        self.output_type = "object"
    
    def forward(self, query: str, source_type: str = "all", max_results: int = 10) -> Dict[str, Any]:
        """同步调用 SerpAPI，返回搜索结果的链接、标题和摘要；网页正文由后续步骤提取。"""
        try:
            # 将查询词、来源类型标记和结果数量上限交给实际搜索方法。
            # 直接返回其结果；_run_sync 已处理的错误也会作为字典返回。
            return self._run_sync(query, source_type, max_results)
        except Exception as e:
            # 兜底捕获仍向外抛出的异常，记录日志并返回错误信息。
            logger.error(f"Source discovery failed: {e}")
            return {"error": str(e)}

    def _run_sync(self, query: str, source_type: str = "all", max_results: int = 10) -> Dict[str, Any]:
        """通过 SerpAPI 搜索来源，并按域名规则评分；不读取网页正文。"""
        try:
            # 1. 初始化搜索客户端；密钥为空时返回空结果。
            serper_api_key = "b353005664862657b0582ba09c2d987dfbdae60bea06d34b7eec9090127b90d6"
            if not serper_api_key:
                logger.warning("SERPAPI_API_KEY not set, returning empty results")
                return {"sources": [], "total_found": 0, "query": query, "high_quality_count": 0}

            client = serpapi.Client(api_key=serper_api_key)
            # 请求 Google 搜索；num 是请求数量，实际返回条数可能不同。
            search_result = client.search({
                "engine": "google",  # 搜索引擎。
                "q": query,  # 搜索词。
                "num": max_results,  # 期望返回的结果数量。
                # 查询包含中文时设置中文搜索界面语言，否则使用英文。
                "hl": "zh-cn" if re.search(r"[\u4e00-\u9fff]", query) else "en",
                "gl": "us"  # 搜索地区参数固定为美国，不是来源国家筛选。
            })
            # 真实返回示例（仅展示关键字段及第一条结果，其余省略）：
            # search_result = {
            #     "search_metadata": {
            #         "id": "6ae9c55e602a0c84dee18daf",
            #         "status": "Success"
            #     },
            #     "search_parameters": {
            #         "engine": "google",
            #         "q": "2026 China economic forecast and policy outlook",
            #         "hl": "en",
            #         "gl": "us"
            #     },
            #     "search_information": {
            #         "query_displayed": "2026 China economic forecast and policy outlook",
            #         "total_results": 144
            #     },
            #     "organic_results": [
            #         {
            #             "position": 1,
            #             "title": "China Bulletin: May 5, 2026",
            #             "link": "https://www.uscc.gov/trade-bulletins/china-bulletin-may-5-2026",
            #             "displayed_link": "https://www.uscc.gov › trade-bulletins › china-bulletin-...",
            #             "date": "May 5, 2026",
            #             "snippet": "According to official figures, China registered 5.0 percent growth in Q1 2026. Closer examination reveals still-weak consumption, ..."
            #         }
            #     ]
            # }
            # 本次 organic_results 共 8 条；snippet 中的 ... 是搜索引擎原本返回的截断标记。
            # link 是目标地址，displayed_link 是展示路径；此时尚未读取网页正文。
            # 后续将 organic_results 整理为 sources：link 改名为 url，另添加域名和评分。

            # API 返回错误信息时，转为异常，统一由下面的 except 处理。
            if search_result.get("error"):
                raise RuntimeError(f"SerpAPI search failed: {search_result['error']}")

            # 2. 整理自然搜索结果；当前不按评分过滤或排序。
            sources = []
            # 预设加分域名列表，仅用于启发式评分，不代表已验证正文质量。
            high_quality_domains = [
                "bloomberg.com", "reuters.com", "wsj.com", "ft.com", "economist.com",
                "sec.gov", "yahoo.com", "marketwatch.com", "seekingalpha.com",
                "morningstar.com", "investopedia.com", "cnbc.com"
            ]
            
            # 只读取自然搜索结果，不处理广告等其他结果区块。
            organic_results = search_result.get('organic_results', [])

            for result in organic_results:
                url = result.get('link', '')
                title = result.get('title', '')
                snippet = result.get('snippet', '')  # 搜索引擎摘要，不是网页全文。
                domain = urlparse(url).netloc.lower()

                # 3. 默认评分为 0.5；域名字符串包含任一预设域名时加分至 0.9。
                quality_score = 0.5
                for quality_domain in high_quality_domains:
                    if domain == quality_domain or domain.endswith("." + quality_domain):
                        quality_score = 0.9
                        break

                # 统一字段名；source_type 只作为结果标签，不参与本次搜索过滤。
                sources.append({
                    "url": url,
                    "title": title,
                    "snippet": snippet,
                    "domain": domain,
                    "quality_score": quality_score,
                    "source_type": source_type
                })
            
            logger.info(f"SerpAPI found {len(sources)} sources for query: {query}")

            # 4. 返回来源列表和统计信息；high_quality_count 统计评分达到 0.8 的条数。
            return {
                "sources": sources,
                "total_found": len(sources),
                "query": query,
                "high_quality_count": len([s for s in sources if s["quality_score"] >= 0.8])
            }
            
        except Exception as e:
            # 替换错误文本中的 api_key= 参数值，再记录日志并返回空来源及错误。
            error = re.sub(r'(?i)(api_key=)[^&\s)]+', r'\1[REDACTED]', str(e))
            logger.error(f"Source discovery failed: {error}")
            return {"error": error, "sources": [], "total_found": 0}

class ContentExtractionTool(Tool):
    """Tool for extracting content from web sources"""
    
    def __init__(self):
        super().__init__()
        self.name = "extract_content"
        self.description = "Extract and analyze content from web sources"
        self.inputs = {
            "urls": {
                "type": "array",
                "description": "List of URLs to extract content from"
            },
            "max_content_length": {
                "type": "integer",
                "description": "Maximum content length per source",
                "nullable": True
            }
        }
        self.output_type = "object"

    def forward(self, urls: List[str], max_content_length: int = 12000) -> Dict[str, Any]:
        """Extract content from multiple web sources (synchronous)"""
        try:
            return self._run_sync(urls, max_content_length)
        except Exception as e:
            logger.error(f"Content extraction failed: {e}")
            return {"error": str(e)}
    
    def _run_sync(self, urls: List[str], max_content_length: int = 12000) -> Dict[str, Any]:
        """Extract content from multiple web sources using requests and BeautifulSoup"""
        try:
            import time
            extracted_content = []
            
            headers = {
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
            }

            for url in urls:
                try:
                    # Use requests to fetch content
                    response = requests.get(url, headers=headers, timeout=10)
                    response.raise_for_status()

                    if response.content.lstrip().startswith(b'%PDF'):
                        import fitz
                        with fitz.open(stream=response.content, filetype='pdf') as document:
                            text = '\n'.join(page.get_text() for page in document)
                            title = document.metadata.get('title') or response.url
                    else:
                        soup = BeautifulSoup(response.content, 'html.parser')
                        title = soup.title.get_text(strip=True) if soup.title else ""
                        for node in soup(["script", "style", "nav", "footer", "header", "noscript"]):
                            node.decompose()
                        containers = soup.select('article, main, [role="main"], .TRS_Editor, .article-content, .entry-content')
                        root = max(containers, key=lambda node: len(node.get_text()), default=soup)
                        text = root.get_text(separator='\n', strip=True)

                    # Clean up whitespace
                    lines = [line.strip() for line in text.split('\n') if line.strip()]
                    clean_text = '\n'.join(lines)
                    
                    # Limit content length
                    if len(clean_text) > max_content_length:
                        clean_text = clean_text[:max_content_length]

                    # Skip if content is too short (< 200 chars means extraction failed)
                    if len(clean_text) < 200:
                        logger.warning(f"Content too short for {url}: {len(clean_text)} chars")
                        continue

                    extracted_content.append({
                        "url": response.url,
                        "requested_url": url,
                        "title": title,
                        "content": clean_text,
                        "chunks_count": 1,
                        "content_length": len(clean_text)
                    })

                    logger.info(f"Successfully extracted {len(clean_text)} chars from {url}")
                    
                    # Small delay to avoid overwhelming servers
                    time.sleep(1)
                    
                except Exception as e:
                    logger.warning(f"Failed to extract content from {url}: {e}")
                    continue
            
            return {
                "extracted_content": extracted_content,
                "total_sources": len(urls),
                "successful_extractions": len(extracted_content),
                "total_content_length": sum(c["content_length"] for c in extracted_content)
            }
            
        except Exception as e:
            logger.error(f"Content extraction failed: {e}")
            return {"error": str(e)}

class DeepResearcherAgent(BaseAgent):
    """Agent 自主搜索与提取；工具观察值是后续证据的唯一来源。"""
    def __init__(self, model, memory=None):
        super().__init__('DeepResearcherAgent', model, tools=[
            SourceDiscoveryTool(), ContentExtractionTool()
        ], instructions=(
            'Search concise topic-specific queries, inspect results and extract actual content. '
            'Search snippets are not source evidence. Read before selecting sources. '
            'Reject unrelated pages; retry different queries on poor results or failed extraction. '
            'Prefer primary sources and cross-check with independent sources when available. '
            'Never invent URLs. Use only user-provided or discovered URLs. '
            'Submit selected source URLs and their relevance reasons via final_answer.'
        ))
