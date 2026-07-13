"""
Deep Researcher Agent for filtering high-quality data sources using SmolAgents
"""

import asyncio
import json
import logging
import requests
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
        """Discover high-quality sources for financial research (synchronous)"""
        try:
            return self._run_sync(query, source_type, max_results)
        except Exception as e:
            logger.error(f"Source discovery failed: {e}")
            return {"error": str(e)}
    
    def _run_sync(self, query: str, source_type: str = "all", max_results: int = 10) -> Dict[str, Any]:
        """Discover high-quality sources for financial research using SerpAPI"""
        try:
            # Use SerpAPI for search
            serper_api_key = "b353005664862657b0582ba09c2d987dfbdae60bea06d34b7eec9090127b90d6"
            if not serper_api_key:
                logger.warning("SERPAPI_API_KEY not set, returning empty results")
                return {"sources": [], "total_found": 0, "query": query, "high_quality_count": 0}
            
            client = serpapi.Client(api_key=serper_api_key)
            search_result = client.search({
                "q": query,
                "num": max_results,
                "hl": "en",
                "gl": "us"
            })
            
            # Parse and filter results
            sources = []
            high_quality_domains = [
                "bloomberg.com", "reuters.com", "wsj.com", "ft.com", "economist.com",
                "sec.gov", "yahoo.com", "marketwatch.com", "seekingalpha.com",
                "morningstar.com", "investopedia.com", "cnbc.com"
            ]
            
            organic_results = search_result.get('organic_results', [])
            
            for result in organic_results:
                url = result.get('link', '')
                title = result.get('title', '')
                snippet = result.get('snippet', '')
                domain = urlparse(url).netloc.lower()
                
                # Quality assessment
                quality_score = 0.5
                for quality_domain in high_quality_domains:
                    if quality_domain in domain:
                        quality_score = 0.9
                        break
                
                sources.append({
                    "url": url,
                    "title": title,
                    "snippet": snippet,
                    "domain": domain,
                    "quality_score": quality_score,
                    "source_type": source_type
                })
            
            logger.info(f"SerpAPI found {len(sources)} sources for query: {query}")
            
            return {
                "sources": sources,
                "total_found": len(sources),
                "query": query,
                "high_quality_count": len([s for s in sources if s["quality_score"] >= 0.8])
            }
            
        except Exception as e:
            logger.error(f"Source discovery failed: {e}")
            return {"error": str(e), "sources": [], "total_found": 0}

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
                    
                    # Parse with BeautifulSoup
                    soup = BeautifulSoup(response.content, 'html.parser')
                    
                    # Remove script and style elements
                    for script in soup(["script", "style", "nav", "footer", "header"]):
                        script.decompose()
                    
                    # Get text content
                    text = soup.get_text(separator='\n', strip=True)
                    
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
                        "url": url,
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