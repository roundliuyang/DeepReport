from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional, Callable
from dataclasses import dataclass
from enum import Enum
import asyncio
import logging
from smolagents import ToolCallingAgent, Tool
from smolagents.memory import AgentMemory
from smolagents.models import Model

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class AgentStatus(Enum):
    IDLE = "idle"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"

@dataclass
class Task:
    """Task data structure for agents"""
    id: str
    type: str
    description: str
    parameters: Dict[str, Any]
    dependencies: List[str] = None
    priority: int = 1
    max_retries: int = 3
    
    def __post_init__(self):
        if self.dependencies is None:
            self.dependencies = []

@dataclass
class TaskResult:
    """Result of task execution"""
    task_id: str
    success: bool
    result: Any = None
    error: str = None
    execution_time: float = 0.0
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}

class BaseAgent(ToolCallingAgent, ABC):
    """Base class for all agents in the system using SmolAgents framework"""
    
    def __init__(self, name: str, model: Model, memory: AgentMemory = None, tools: List[Tool] = None, system_prompt: str = None):
        # Create default memory if not provided
        if memory is None:
            prompt = system_prompt or f"You are {name}, a specialized AI assistant."
            memory = AgentMemory(system_prompt=prompt)
        
        # Store memory for later use (smolagents 1.26.0 doesn't accept memory parameter)
        self.agent_memory = memory
        
        # Initialize ToolCallingAgent without memory parameter
        super().__init__(tools=tools or [], model=model)
        
        self.name = name
        self.status = AgentStatus.IDLE
        self.task_queue = asyncio.Queue()
        self.completed_tasks = {}
        self.failed_tasks = {}
        self.model_config = model.config if hasattr(model, 'config') else {}
        
    @abstractmethod
    async def execute_task(self, task: Task) -> TaskResult:
        """Execute a single task"""
        pass
    
    @abstractmethod
    def get_capabilities(self) -> List[str]:
        """Return list of capabilities this agent provides"""
        pass
    
    async def start(self):
        """Start the agent"""
        self.status = AgentStatus.RUNNING
        logger.info(f"Agent {self.name} started")
        
    async def stop(self):
        """Stop the agent"""
        self.status = AgentStatus.IDLE
        logger.info(f"Agent {self.name} stopped")
    
    async def submit_task(self, task: Task) -> str:
        """Submit a task for execution"""
        await self.task_queue.put(task)
        logger.info(f"Task {task.id} submitted to {self.name}")
        return task.id
    
    async def get_task_result(self, task_id: str) -> Optional[TaskResult]:
        """Get result of a completed task"""
        return self.completed_tasks.get(task_id)
    
    async def wait_for_task(self, task_id: str, timeout: float = 300.0) -> Optional[TaskResult]:
        """Wait for a task to complete"""
        start_time = asyncio.get_event_loop().time()
        
        while asyncio.get_event_loop().time() - start_time < timeout:
            result = await self.get_task_result(task_id)
            if result:
                return result
            await asyncio.sleep(1)
        
        return None
    
    async def process_tasks(self):
        """Process tasks from the queue"""
        while self.status == AgentStatus.RUNNING:
            try:
                task = await asyncio.wait_for(self.task_queue.get(), timeout=1.0)
                logger.info(f"Processing task {task.id} with {self.name}")
                
                result = await self.execute_task(task)
                
                if result.success:
                    self.completed_tasks[task.id] = result
                    logger.info(f"Task {task.id} completed successfully")
                else:
                    self.failed_tasks[task.id] = result
                    logger.error(f"Task {task.id} failed: {result.error}")
                
                self.task_queue.task_done()
                
            except asyncio.TimeoutError:
                continue
            except Exception as e:
                logger.error(f"Error processing task in {self.name}: {str(e)}")
                await asyncio.sleep(1)
    
    async def run_task_with_smolagents(self, task_description: str, tools: List[Tool] = None) -> Dict[str, Any]:
        """Run a task using SmolAgents framework"""
        try:
            # Use SmolAgents to run the task (run() is synchronous in smolagents 1.26.0)
            result = self.run(task_description, max_steps=5)
            return {"success": True, "result": str(result)}
        except Exception as e:
            logger.error(f"SmolAgents task execution failed: {e}")
            return {"success": False, "error": str(e)}
    
    def add_custom_tool(self, tool: Tool):
        """Add a custom tool to the agent"""
        # In smolagents 1.26.0, self.tools is a dict keyed by tool name
        if hasattr(self, 'tools'):
            if isinstance(self.tools, dict):
                self.tools[tool.name] = tool
            else:
                self.tools.append(tool)
        else:
            self.tools = {tool.name: tool}
        logger.info(f"Added tool {tool.name} to {self.name}")
    
    def get_tools_info(self) -> List[Dict[str, Any]]:
        """Get information about available tools"""
        return [
            {
                "name": tool.name,
                "description": tool.description,
                "parameters": getattr(tool, 'inputs', {})
            }
            for tool in self.tools
        ]