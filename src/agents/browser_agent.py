"""
Browser Agent for web interaction and PDF analysis using SmolAgents
"""

import asyncio
import json
import logging
import os
import tempfile
from typing import Dict, Any, List, Optional
from datetime import datetime
from urllib.parse import urlparse
import aiohttp
import fitz  # PyMuPDF for PDF processing
from playwright.async_api import async_playwright
from smolagents import Tool
from smolagents.memory import AgentMemory
from smolagents.models import OpenAIModel

from .base_agent import BaseAgent, Task, TaskResult

logger = logging.getLogger(__name__)

# Compatibility wrapper for browser_use library (which removed BrowserUse class in newer versions)
class BrowserUse:
    """Compatibility wrapper using Playwright for web automation"""
    
    def __init__(self):
        self.playwright = None
        self.browser = None
        self.context = None
        self.page = None
    
    async def __aenter__(self):
        await self.start()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.close()
    
    async def start(self):
        """Initialize Playwright browser"""
        self.playwright = await async_playwright().start()
        self.browser = await self.playwright.chromium.launch(headless=True)
        self.context = await self.browser.new_context()
        self.page = await self.context.new_page()
    
    async def close(self):
        """Close browser resources"""
        if self.browser:
            await self.browser.close()
        if self.playwright:
            await self.playwright.stop()
    
    async def goto(self, url: str, timeout: int = 30000):
        """Navigate to URL"""
        await self.page.goto(url, timeout=timeout, wait_until='domcontentloaded')
    
    async def get_title(self) -> str:
        """Get page title"""
        return await self.page.title()
    
    async def get_text_content(self) -> str:
        """Get all text content from the page"""
        return await self.page.inner_text('body')
    
    async def click_element(self, selector: str):
        """Click an element by CSS selector"""
        await self.page.click(selector, timeout=10000)
    
    async def fill_input(self, selector: str, value: str):
        """Fill an input field"""
        await self.page.fill(selector, value, timeout=10000)
    
    async def scroll_to_element(self, selector: str):
        """Scroll to an element"""
        await self.page.wait_for_selector(selector, timeout=10000)
        await self.page.evaluate(f"""
            document.querySelector('{selector}').scrollIntoView({{behavior: 'smooth'}})
        """)
    
    async def extract_element_text(self, selector: str) -> str:
        """Extract text from an element"""
        try:
            element = await self.page.query_selector(selector)
            if element:
                return await element.inner_text()
            return ""
        except Exception:
            return ""
    
    async def get_tables(self) -> List[List[str]]:
        """Extract all tables from the page"""
        try:
            tables = await self.page.evaluate("""
                () => {
                    const tables = Array.from(document.querySelectorAll('table'));
                    return tables.map(table => {
                        const rows = Array.from(table.querySelectorAll('tr'));
                        return rows.map(row => {
                            const cells = Array.from(row.querySelectorAll('td, th'));
                            return cells.map(cell => cell.textContent.trim()).filter(Boolean);
                        }).filter(row => row.length > 0);
                    });
                }
            """)
            return tables
        except Exception:
            return []
    
    async def get_forms(self) -> List[Dict[str, Any]]:
        """Extract all forms from the page"""
        try:
            forms = await self.page.evaluate("""
                () => {
                    const forms = Array.from(document.querySelectorAll('form'));
                    return forms.map(form => {
                        const fields = Array.from(form.querySelectorAll('input, select, textarea'));
                        return {
                            id: form.id || '',
                            action: form.action || '',
                            method: form.method || '',
                            fields: fields.map(field => ({
                                name: field.name || '',
                                type: field.type || field.tagName.toLowerCase(),
                                placeholder: field.placeholder || ''
                            }))
                        };
                    });
                }
            """)
            return forms
        except Exception:
            return []
    
    async def get_lists(self) -> List[List[str]]:
        """Extract all lists from the page"""
        try:
            lists = await self.page.evaluate("""
                () => {
                    const lists = Array.from(document.querySelectorAll('ul, ol'));
                    return lists.map(list => {
                        const items = Array.from(list.querySelectorAll('li'));
                        return items.map(item => item.textContent.trim());
                    });
                }
            """)
            return lists
        except Exception:
            return []
    
    async def get_links(self) -> List[Dict[str, str]]:
        """Extract all links from the page"""
        try:
            links = await self.page.evaluate("""
                () => {
                    const links = Array.from(document.querySelectorAll('a[href]'));
                    return links.map(link => ({
                        text: link.textContent.trim(),
                        href: link.href,
                        title: link.title || ''
                    }));
                }
            """)
            return links
        except Exception:
            return []
    
    async def get_images(self) -> List[Dict[str, str]]:
        """Extract all images from the page"""
        try:
            images = await self.page.evaluate("""
                () => {
                    const images = Array.from(document.querySelectorAll('img'));
                    return images.map(img => ({
                        src: img.src,
                        alt: img.alt || '',
                        title: img.title || ''
                    }));
                }
            """)
            return images
        except Exception:
            return []
    
    async def click_button(self, selector: str):
        """Click a button by CSS selector"""
        await self.page.click(selector, timeout=10000)

class WebNavigationTool(Tool):
    """Tool for web navigation and interaction"""
    
    def __init__(self, browser: BrowserUse):
        super().__init__()
        self.name = "navigate_web"
        self.description = "Navigate and interact with web pages"
        self.inputs = {
            "url": {
                "type": "string",
                "description": "URL to navigate to"
            },
            "actions": {
                "type": "array",
                "description": "List of actions to perform on the page",
                "nullable": True
            }
        }
        self.output_type = "object"
        self.browser = browser
    
    def forward(self, url: str, actions: List[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Navigate to URL and perform actions (synchronous)"""
        try:
            # Note: This tool requires browser initialization
            return {"error": "WebNavigationTool requires async context - use BrowserAgent directly"}
        except Exception as e:
            logger.error(f"Web navigation failed: {e}")
            return {"error": str(e)}
    
    async def _navigate_and_perform(self, url: str, actions: List[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Async helper for navigation"""
        # Navigate to URL
        await self.browser.goto(url)
        
        # Extract basic page information
        title = await self.browser.get_title()
        content = await self.browser.get_text_content()
        
        # Perform actions
        action_results = []
        if actions:
            for action in actions:
                result = await self._perform_action(action)
                action_results.append(result)
        
        return {
            "url": url,
            "title": title,
            "content": content,
            "action_results": action_results,
            "timestamp": datetime.now().isoformat()
        }
    
    async def _perform_action(self, action: Dict[str, Any]) -> Dict[str, Any]:
        """Perform a single action on the page"""
        try:
            action_type = action.get("type")
            target = action.get("target")
            value = action.get("value")
            
            if action_type == "click":
                await self.browser.click_element(target)
                return {"action": "click", "target": target, "success": True}
            elif action_type == "fill":
                await self.browser.fill_input(target, value)
                return {"action": "fill", "target": target, "value": value, "success": True}
            elif action_type == "scroll":
                await self.browser.scroll_to_element(target)
                return {"action": "scroll", "target": target, "success": True}
            elif action_type == "extract":
                content = await self.browser.extract_element_text(target)
                return {"action": "extract", "target": target, "content": content, "success": True}
            else:
                return {"action": action_type, "target": target, "success": False, "error": "Unknown action type"}
                
        except Exception as e:
            return {"action": action.get("type", "unknown"), "target": action.get("target", "unknown"), "success": False, "error": str(e)}

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
            return self._analyze_pdf_sync(pdf_path, extract_tables, extract_charts)
        except Exception as e:
            logger.error(f"PDF analysis failed: {e}")
            return {"error": str(e)}
    
    def _analyze_pdf_sync(self, pdf_path: str, extract_tables: bool = True, extract_charts: bool = True) -> Dict[str, Any]:
        """Synchronous PDF analysis"""
        import requests
        
        # Check if it's a URL or local path
        if pdf_path.startswith("http"):
            # Download PDF from URL
            response = requests.get(pdf_path)
            if response.status_code == 200:
                # Save to temporary file
                with tempfile.NamedTemporaryFile(suffix='.pdf', delete=False) as temp_file:
                    temp_file.write(response.content)
                    pdf_path = temp_file.name
            else:
                return {"error": f"Failed to download PDF: {response.status_code}"}
        
        # Analyze PDF using PyMuPDF
        doc = fitz.open(pdf_path)
        
        analysis = {
            "metadata": doc.metadata,
            "page_count": len(doc),
            "word_count": 0,
            "sections": [],
            "tables": [],
            "charts": [],
            "financial_data": [],
            "full_text": ""
        }
        
        # Extract text and analyze structure
        full_text = ""
        for page_num, page in enumerate(doc):
            text = page.get_text()
            full_text += text
            
            # Basic section detection
            if text.strip():
                analysis["sections"].append({
                    "page": page_num + 1,
                    "content": text[:500] + "..." if len(text) > 500 else text,
                    "word_count": len(text.split())
                })
            
            # Extract tables
            if extract_tables:
                tables = page.find_tables()
                for table in tables:
                    analysis["tables"].append({
                        "page": page_num + 1,
                        "rows": len(table.rows()),
                        "cols": len(table.cols()),
                        "data": table.extract()
                    })
            
            # Look for financial data patterns
            financial_indicators = ["$", "%", "revenue", "profit", "loss", "EBITDA", "EPS", "P/E"]
            found_indicators = [indicator for indicator in financial_indicators if indicator.lower() in text.lower()]
            if found_indicators:
                analysis["financial_data"].append({
                    "page": page_num + 1,
                    "indicators": found_indicators,
                    "context": text[:200]
                })
        
        analysis["full_text"] = full_text
        analysis["word_count"] = len(full_text.split())
        
        # Clean up temporary file if created
        if pdf_path.startswith("/tmp/"):
            os.unlink(pdf_path)
        
        doc.close()
        
        return analysis

class StructuredDataExtractionTool(Tool):
    """Tool for extracting structured data from web pages"""
    
    def __init__(self, browser: BrowserUse):
        super().__init__()
        self.name = "extract_structured_data"
        self.description = "Extract structured data (tables, forms, lists) from web pages"
        self.inputs = {
            "url": {
                "type": "string",
                "description": "URL to extract data from"
            },
            "data_types": {
                "type": "array",
                "description": "Types of data to extract",
                "nullable": True
            }
        }
        self.output_type = "object"
        self.browser = browser
    
    def forward(self, url: str, data_types: List[str] = None) -> Dict[str, Any]:
        """Extract structured data from web page (synchronous)"""
        try:
            # Note: This tool requires browser initialization
            return {"error": "StructuredDataExtractionTool requires async context - use BrowserAgent directly"}
        except Exception as e:
            logger.error(f"Structured data extraction failed: {e}")
            return {"error": str(e)}
    
    async def _extract_data_async(self, url: str, data_types: List[str] = None) -> Dict[str, Any]:
        """Async helper for structured data extraction"""
        if data_types is None:
            data_types = ["tables", "forms", "lists"]
        
        await self.browser.goto(url)
        
        structured_data = {
            "url": url,
            "title": await self.browser.get_title(),
            "extracted_data": {}
        }
        
        # Extract different types of structured data
        if "tables" in data_types:
            tables = await self.browser.get_tables()
            structured_data["extracted_data"]["tables"] = tables
        
        if "forms" in data_types:
            forms = await self.browser.get_forms()
            structured_data["extracted_data"]["forms"] = forms
        
        if "lists" in data_types:
            lists = await self.browser.get_lists()
            structured_data["extracted_data"]["lists"] = lists
        
        if "links" in data_types:
            links = await self.browser.get_links()
            structured_data["extracted_data"]["links"] = links
        
        if "images" in data_types:
            images = await self.browser.get_images()
            structured_data["extracted_data"]["images"] = images
        
        return structured_data

class FormInteractionTool(Tool):
    """Tool for interacting with web forms"""
    
    def __init__(self, browser: BrowserUse):
        super().__init__()
        self.name = "interact_with_form"
        self.description = "Fill and submit web forms"
        self.inputs = {
            "url": {
                "type": "string",
                "description": "URL containing the form"
            },
            "form_data": {
                "type": "object",
                "description": "Form field names and values to fill"
            },
            "submit_button": {
                "type": "string",
                "description": "Selector for the submit button",
                "nullable": True
            }
        }
        self.output_type = "object"
        self.browser = browser
    
    def forward(self, url: str, form_data: Dict[str, Any], submit_button: str = None) -> Dict[str, Any]:
        """Interact with web forms (synchronous)"""
        try:
            # Note: This tool requires browser initialization
            return {"error": "FormInteractionTool requires async context - use BrowserAgent directly"}
        except Exception as e:
            logger.error(f"Form interaction failed: {e}")
            return {"error": str(e)}
    
    async def _interact_form_async(self, url: str, form_data: Dict[str, Any], submit_button: str = None) -> Dict[str, Any]:
        """Async helper for form interaction"""
        await self.browser.goto(url)
        
        # Fill form fields
        for field_name, field_value in form_data.items():
            await self.browser.fill_input(field_name, field_value)
        
        # Submit form if button specified
        if submit_button:
            await self.browser.click_button(submit_button)
        
        # Extract results after form submission
        result_content = await self.browser.get_text_content()
        
        return {
            "url": url,
            "form_data": form_data,
            "submitted": submit_button is not None,
            "result_content": result_content,
            "timestamp": datetime.now().isoformat()
        }

class BrowserAgent(BaseAgent):
    """Agent for fine-grained web interaction using browser-use library and SmolAgents"""
    
    def __init__(self, model: OpenAIModel, memory: AgentMemory = None):
        # Initialize browser wrapper (not the actual browser yet)
        self.browser = BrowserUse()
        self.browser_initialized = False
        
        # Initialize tools dict before calling super().__init__
        # This is needed because ToolCallingAgent.__init__ will access self.tools
        self.tools = {}
        
        super().__init__(
            name="BrowserAgent",
            model=model,
            memory=memory or AgentMemory(system_prompt="You are a browser agent that performs web interaction and PDF analysis.")
        )
        
        # Add PDF tool which doesn't need browser
        self.add_custom_tool(PDFAnalysisTool())
        
    async def start(self):
        """Start the browser agent"""
        await super().start()
        # Initialize browser and add remaining tools
        if not self.browser_initialized:
            await self.browser.start()
            self.browser_initialized = True
            self.add_custom_tool(WebNavigationTool(self.browser))
            self.add_custom_tool(StructuredDataExtractionTool(self.browser))
            self.add_custom_tool(FormInteractionTool(self.browser))
        
    async def _initialize_browser(self):
        """Initialize browser-use instance"""
        try:
            self.browser = BrowserUse()
            self.browser_initialized = True
            logger.info("Browser-use initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize browser-use: {e}")
            raise
    
    def _setup_tools(self, tools, add_base_tools):
        """Setup browser tools - called by parent __init__ synchronously"""
        # Initialize tools dict (required by parent class)
        self.tools = {tool.name: tool for tool in tools} if tools else {}
        
        # Don't add base tools automatically
        # We'll add our custom tools later in start() method
    
    async def execute_task(self, task: Task) -> TaskResult:
        """Execute browser-related tasks using SmolAgents"""
        start_time = datetime.now()
        
        try:
            task_type = task.parameters.get("browser_task_type")
            url = task.parameters.get("url")
            query = task.parameters.get("query")
            action = task.parameters.get("action")
            pdf_path = task.parameters.get("pdf_path")
            
            # Create task description for SmolAgents
            task_description = f"Perform browser task: {task_type or 'web interaction'}"
            
            if url:
                task_description += f" on {url}"
            if query:
                task_description += f" with query: {query}"
            
            # Use SmolAgents to orchestrate the task
            result = await self.run_task_with_smolagents(task_description)
            
            if result.get("success"):
                browser_result = result.get("result", {})
                
                # Enhance with specific browser operations
                enhanced_result = await self._enhance_browser_result(browser_result, task)
                
                execution_time = (datetime.now() - start_time).total_seconds()
                
                return TaskResult(
                    task_id=task.id,
                    success=True,
                    result=enhanced_result,
                    execution_time=execution_time,
                    metadata={
                        "url": url,
                        "query": query,
                        "task_type": task_type,
                        "action": action,
                        "method": "smolagents"
                    }
                )
            else:
                # Fallback to traditional method
                fallback_result = await self._execute_traditional_browser_task(task)
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
    
    async def _enhance_browser_result(self, smolagents_result: Dict[str, Any], task: Task) -> Dict[str, Any]:
        """Enhance SmolAgents result with browser-specific operations"""
        try:
            task_type = task.parameters.get("browser_task_type")
            url = task.parameters.get("url")
            
            if task_type == "web_navigation" and url:
                # Perform actual web navigation
                navigation_tool = WebNavigationTool(self.browser)
                navigation_result = await navigation_tool.run(url)
                smolagents_result.update(navigation_result)
            
            elif task_type == "pdf_analysis":
                pdf_path = task.parameters.get("pdf_path") or url
                if pdf_path:
                    pdf_tool = PDFAnalysisTool()
                    pdf_result = await pdf_tool.run(pdf_path)
                    smolagents_result.update(pdf_result)
            
            elif task_type == "structured_data_extraction" and url:
                extraction_tool = StructuredDataExtractionTool(self.browser)
                extraction_result = await extraction_tool.run(url)
                smolagents_result.update(extraction_result)
            
            # Add metadata
            smolagents_result["enhanced_at"] = datetime.now().isoformat()
            smolagents_result["enhanced_by"] = "BrowserAgent"
            
            return smolagents_result
            
        except Exception as e:
            logger.error(f"Failed to enhance browser result: {e}")
            return smolagents_result
    
    async def _execute_traditional_browser_task(self, task: Task) -> TaskResult:
        """Fallback traditional browser task execution"""
        try:
            task_type = task.parameters.get("browser_task_type")
            url = task.parameters.get("url")
            action = task.parameters.get("action")
            pdf_path = task.parameters.get("pdf_path")
            
            if task_type == "web_navigation":
                result = await self._perform_web_navigation(url, action)
            elif task_type == "pdf_analysis":
                result = await self._analyze_pdf_document(pdf_path or url)
            elif task_type == "form_interaction":
                result = await self._interact_with_form(url, action)
            elif task_type == "structured_data_extraction":
                result = await self._extract_structured_data(url)
            else:
                result = {"error": f"Unknown browser task type: {task_type}"}
            
            return TaskResult(
                task_id=task.id,
                success="error" not in result,
                result=result if "error" not in result else None,
                error=result.get("error") if "error" in result else None,
                metadata={
                    "url": url,
                    "task_type": task_type,
                    "method": "traditional"
                }
            )
            
        except Exception as e:
            return TaskResult(
                task_id=task.id,
                success=False,
                error=str(e)
            )
    
    async def _perform_web_navigation(self, url: str, action: str) -> Dict[str, Any]:
        """Perform web navigation and interaction"""
        try:
            await self.browser.goto(url)
            
            title = await self.browser.get_title()
            content = await self.browser.get_text_content()
            
            action_results = {}
            if action:
                action_results = await self._execute_browser_action(action)
            
            return {
                "url": url,
                "title": title,
                "content": content,
                "action_results": action_results,
                "timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            return {"error": f"Failed to perform web navigation: {str(e)}"}
    
    async def _analyze_pdf_document(self, pdf_path: str) -> Dict[str, Any]:
        """Analyze PDF document content"""
        try:
            pdf_tool = PDFAnalysisTool()
            return await pdf_tool.run(pdf_path)
            
        except Exception as e:
            return {"error": f"Failed to analyze PDF: {str(e)}"}
    
    async def _interact_with_form(self, url: str, action: Dict[str, Any]) -> Dict[str, Any]:
        """Interact with web forms"""
        try:
            form_tool = FormInteractionTool(self.browser)
            form_data = action.get("fields", {})
            submit_button = action.get("submit")
            
            return await form_tool.run(url, form_data, submit_button)
            
        except Exception as e:
            return {"error": f"Failed to interact with form: {str(e)}"}
    
    async def _extract_structured_data(self, url: str) -> Dict[str, Any]:
        """Extract structured data from web pages"""
        try:
            extraction_tool = StructuredDataExtractionTool(self.browser)
            return await extraction_tool.run(url)
            
        except Exception as e:
            return {"error": f"Failed to extract structured data: {str(e)}"}
    
    async def _execute_browser_action(self, action_description: str) -> Dict[str, Any]:
        """Execute browser action based on natural language description"""
        try:
            # Use AI to interpret the action
            prompt = f"""
            Interpret this browser action: "{action_description}"
            
            Return a JSON object with:
            {{
                "action_type": "click" | "fill" | "scroll" | "extract" | "navigate",
                "target": "element description or selector",
                "value": "value to fill (if applicable)",
                "confidence": 0.0 to 1.0
            }}
            """
            
            response = await self.model.generate_structured_response(
                prompt,
                {
                    "action_type": "",
                    "target": "",
                    "value": "",
                    "confidence": 0.0
                }
            )
            
            if response.get("success") and response["content"]["confidence"] > 0.7:
                action = response["content"]
                
                if action["action_type"] == "click":
                    await self.browser.click_element(action["target"])
                elif action["action_type"] == "fill":
                    await self.browser.fill_input(action["target"], action["value"])
                elif action["action_type"] == "scroll":
                    await self.browser.scroll_to_element(action["target"])
                elif action["action_type"] == "extract":
                    content = await self.browser.extract_element_text(action["target"])
                    return {"content": content, "interpreted_action": action}
                
                return {"interpreted_action": action, "executed": True}
            else:
                return {"error": "Could not interpret action with sufficient confidence"}
                
        except Exception as e:
            return {"error": f"Failed to execute browser action: {str(e)}"}
    
    async def stop(self):
        """Stop the browser agent"""
        if self.browser:
            await self.browser.close()
        await super().stop()
    
    def get_capabilities(self) -> List[str]:
        """Return list of capabilities this agent provides"""
        return [
            "web_navigation",
            "pdf_analysis",
            "form_interaction",
            "structured_data_extraction",
            "content_interaction",
            "ai_guided_browsing",
            "smolagents_integration"
        ]
    
    async def browse_with_smolagents(self, url: str, task: str) -> Dict[str, Any]:
        """Browse website using SmolAgents framework directly"""
        try:
            # Create comprehensive browsing task
            task_description = f"""
            Browse and interact with website: {url}
            
            Task: {task}
            
            Please:
            1. Navigate to the URL
            2. Understand the page structure
            3. Perform necessary interactions
            4. Extract relevant information
            5. Provide structured results
            """
            
            result = await self.run_task_with_smolagents(task_description)
            
            if result.get("success"):
                return result.get("result", {})
            else:
                raise Exception(f"SmolAgents browsing failed: {result.get('error')}")
                
        except Exception as e:
            logger.error(f"SmolAgents browsing failed: {e}")
            raise