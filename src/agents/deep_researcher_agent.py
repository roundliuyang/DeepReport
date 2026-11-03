"""
Deep Researcher Agent for filtering high-quality data sources using SmolAgents
"""

import asyncio
import json
import logging
import requests
import re
from typing import Dict, Any, List, Optional
from datetime import datetime
from urllib.parse import urlparse
from smolagents import Tool
from smolagents.memory import AgentMemory
from smolagents.models import OpenAIModel
import serpapi
from langchain_community.document_loaders import WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
import requests
from bs4 import BeautifulSoup

from .base_agent import BaseAgent, Task, TaskResult

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
                    if quality_domain in domain:
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
    
    def forward(self, urls: List[str], max_content_length: int = 2000) -> Dict[str, Any]:
        """Extract content from multiple web sources (synchronous)"""
        try:
            return self._run_sync(urls, max_content_length)
        except Exception as e:
            logger.error(f"Content extraction failed: {e}")
            return {"error": str(e)}
    
    def _run_sync(self, urls: List[str], max_content_length: int = 2000) -> Dict[str, Any]:
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

class QualityAssessmentTool(Tool):
    """Tool for assessing content quality"""
    
    def __init__(self):
        super().__init__()
        self.name = "assess_quality"
        self.description = "Assess the quality and relevance of extracted content"
        self.inputs = {
            "content_data": {
                "type": "array",
                "description": "Array of content objects to assess"
            },
            "research_topic": {
                "type": "string",
                "description": "The research topic for relevance assessment"
            }
        }
        self.output_type = "object"
    
    def forward(self, content_data: List[Dict[str, Any]], research_topic: str) -> Dict[str, Any]:
        """Assess quality and relevance of content"""
        try:
            quality_assessments = []
            
            for content in content_data:
                content_text = content.get("content", "")
                url = content.get("url", "")
                
                # Quality metrics
                quality_metrics = {
                    "content_length": len(content_text),
                    "readability_score": self._calculate_readability(content_text),
                    "relevance_score": self._calculate_relevance(content_text, research_topic),
                    "source_credibility": self._assess_source_credibility(url),
                    "data_richness": self._assess_data_richness(content_text)
                }
                
                # Overall quality score
                overall_score = (
                    quality_metrics["readability_score"] * 0.2 +
                    quality_metrics["relevance_score"] * 0.3 +
                    quality_metrics["source_credibility"] * 0.3 +
                    quality_metrics["data_richness"] * 0.2
                )
                
                quality_metrics["overall_score"] = overall_score
                
                quality_assessments.append({
                    "url": url,
                    "quality_metrics": quality_metrics,
                    "recommendation": "include" if overall_score >= 0.7 else "review"
                })
            
            return {
                "quality_assessments": quality_assessments,
                "average_quality_score": sum(a["quality_metrics"]["overall_score"] for a in quality_assessments) / len(quality_assessments) if quality_assessments else 0,
                "recommended_sources": len([a for a in quality_assessments if a["recommendation"] == "include"])
            }
            
        except Exception as e:
            logger.error(f"Quality assessment failed: {e}")
            return {"error": str(e)}
    
    def _calculate_readability(self, text: str) -> float:
        """Calculate readability score (simplified)"""
        if not text:
            return 0.0
        
        # Simple metrics
        sentences = text.split('.')
        words = text.split()
        
        if len(sentences) == 0 or len(words) == 0:
            return 0.0
        
        avg_words_per_sentence = len(words) / len(sentences)
        
        # Normalize to 0-1 scale
        if avg_words_per_sentence > 25:
            return 0.3
        elif avg_words_per_sentence < 10:
            return 0.5
        else:
            return 0.8
    
    def _calculate_relevance(self, text: str, topic: str) -> float:
        """Calculate relevance score to research topic"""
        if not text or not topic:
            return 0.0
        
        # Simple keyword matching
        topic_lower = topic.lower()
        text_lower = text.lower()
        
        topic_words = topic_lower.split()
        matches = sum(1 for word in topic_words if word in text_lower)
        
        return min(matches / len(topic_words), 1.0)
    
    def _assess_source_credibility(self, url: str) -> float:
        """Assess source credibility based on URL"""
        high_quality_domains = [
            "bloomberg.com", "reuters.com", "wsj.com", "ft.com", "economist.com",
            "sec.gov", "yahoo.com", "marketwatch.com", "seekingalpha.com",
            "morningstar.com", "investopedia.com", "cnbc.com"
        ]
        
        domain = urlparse(url).netloc.lower()
        
        for quality_domain in high_quality_domains:
            if quality_domain in domain:
                return 0.9
        
        # Default credibility
        return 0.5
    
    def _assess_data_richness(self, text: str) -> float:
        """Assess data richness in content"""
        if not text:
            return 0.0
        
        # Look for data indicators
        data_indicators = ["%", "$", "billion", "million", "data", "statistics", "figures", "chart", "table"]
        indicator_count = sum(1 for indicator in data_indicators if indicator in text.lower())
        
        return min(indicator_count / len(data_indicators), 1.0)

class DeepResearcherAgent(BaseAgent):
    """Agent for filtering high-quality data sources and research using SmolAgents"""
    
    def __init__(self, model: OpenAIModel, memory: AgentMemory = None):
        super().__init__(
            name="DeepResearcherAgent",
            model=model,
            memory=memory or AgentMemory(system_prompt="You are a deep researcher agent that filters and extracts high-quality data sources.")
        )
        
        # Add research tools
        self.add_custom_tool(SourceDiscoveryTool())
        self.add_custom_tool(ContentExtractionTool())
        self.add_custom_tool(QualityAssessmentTool())
        
        self.high_quality_sources = [
            "bloomberg.com", "reuters.com", "wsj.com", "ft.com", "economist.com",
            "sec.gov", "yahoo.com", "marketwatch.com", "seekingalpha.com",
            "morningstar.com", "investopedia.com", "cnbc.com"
        ]
    
    async def execute_task(self, task: Task) -> TaskResult:
        """Execute deep research tasks using SmolAgents"""
        start_time = datetime.now()
        
        try:
            research_topic = task.parameters.get("research_topic")
            query = task.parameters.get("query")
            source_filter = task.parameters.get("source_filter", True)
            
            if not research_topic and not query:
                return TaskResult(
                    task_id=task.id,
                    success=False,
                    error="Research topic or query is required"
                )
            
            # Use SmolAgents to orchestrate the research process
            task_description = f"Research and filter high-quality sources for: {research_topic or query}"
            
            if source_filter:
                task_description += " with strict quality filtering"
            
            result = await self.run_task_with_smolagents(task_description)
            
            if result.get("success"):
                research_result = result.get("result", {})
                
                # Enhance with additional processing
                enhanced_result = await self._enhance_research_result(research_result, research_topic)
                
                execution_time = (datetime.now() - start_time).total_seconds()
                
                return TaskResult(
                    task_id=task.id,
                    success=True,
                    result=enhanced_result,
                    execution_time=execution_time,
                    metadata={
                        "research_topic": research_topic,
                        "source_filter": source_filter,
                        "method": "smolagents",
                        "total_sources": len(enhanced_result.get("sources", []))
                    }
                )
            else:
                # Fallback to traditional method
                fallback_result = await self._execute_traditional_research(task)
                fallback_result.execution_time = (datetime.now() - start_time).total_seconds()
                return fallback_result
                
        except Exception as e:
            execution_time = (datetime.now() - start_time).total_seconds()
            return TaskResult(
                task_id=task.id,
                success=False,
                error=str(e),
                execution_time=execution_time
            )
    
    async def _enhance_research_result(self, smolagents_result: Dict[str, Any], research_topic: str) -> Dict[str, Any]:
        """Enhance SmolAgents result with additional analysis"""
        try:
            # Extract sources from SmolAgents result
            sources = smolagents_result.get("sources", [])
            
            # Perform additional quality filtering
            if sources:
                quality_analysis = await self._perform_quality_analysis(sources, research_topic)
                smolagents_result["quality_analysis"] = quality_analysis
            
            # Add metadata
            smolagents_result["enhanced_at"] = datetime.now().isoformat()
            smolagents_result["enhanced_by"] = "DeepResearcherAgent"
            
            return smolagents_result
            
        except Exception as e:
            logger.error(f"Failed to enhance research result: {e}")
            return smolagents_result
    
    async def _perform_quality_analysis(self, sources: List[Dict[str, Any]], research_topic: str) -> Dict[str, Any]:
        """Perform quality analysis on discovered sources"""
        try:
            # Use the quality assessment tool
            quality_tool = QualityAssessmentTool()
            
            # Prepare content data for quality assessment
            content_data = []
            for source in sources:
                content_data.append({
                    "url": source.get("url", ""),
                    "content": source.get("content", ""),
                    "title": source.get("title", "")
                })
            
            quality_result = await quality_tool.run(content_data, research_topic)
            return quality_result
            
        except Exception as e:
            logger.error(f"Quality analysis failed: {e}")
            return {"error": str(e)}
    
    async def _execute_traditional_research(self, task: Task) -> TaskResult:
        """Fallback traditional research method"""
        try:
            research_topic = task.parameters.get("research_topic")
            query = task.parameters.get("query")
            source_filter = task.parameters.get("source_filter", True)
            
            # Traditional research logic
            search_query = query or research_topic
            
            # Use source discovery tool
            discovery_tool = SourceDiscoveryTool()
            discovery_result = await discovery_tool.run(search_query)
            
            if "error" in discovery_result:
                return TaskResult(
                    task_id=task.id,
                    success=False,
                    error=discovery_result["error"]
                )
            
            sources = discovery_result.get("sources", [])
            
            # Extract content from sources
            if sources:
                urls = [s["url"] for s in sources[:5]]  # Top 5 sources
                extraction_tool = ContentExtractionTool()
                extraction_result = await extraction_tool.run(urls)
                
                if "error" not in extraction_result:
                    discovery_result.update(extraction_result)
            
            return TaskResult(
                task_id=task.id,
                success=True,
                result=discovery_result,
                metadata={
                    "research_topic": research_topic,
                    "source_filter": source_filter,
                    "method": "traditional"
                }
            )
            
        except Exception as e:
            return TaskResult(
                task_id=task.id,
                success=False,
                error=str(e)
            )
    
    def get_capabilities(self) -> List[str]:
        """Return list of capabilities this agent provides"""
        return [
            "source_discovery",
            "quality_filtering",
            "content_extraction",
            "research_analysis",
            "data_curation",
            "smolagents_integration"
        ]
    
    async def research_with_smolagents(self, research_topic: str, requirements: List[str] = None) -> Dict[str, Any]:
        """Perform research using SmolAgents framework directly"""
        try:
            # Create a comprehensive research task
            task_description = f"""
            Conduct comprehensive financial research on: {research_topic}
            
            Research requirements:
            - Discover high-quality data sources
            - Filter sources by credibility and relevance
            - Extract and analyze content
            - Provide quality assessment
            - Return structured research data
            
            Additional requirements: {', '.join(requirements) if requirements else 'None specified'}
            """
            
            result = await self.run_task_with_smolagents(task_description)
            
            if result.get("success"):
                return result.get("result", {})
            else:
                raise Exception(f"SmolAgents research failed: {result.get('error')}")
                
        except Exception as e:
            logger.error(f"SmolAgents research failed: {e}")
            raise
