from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, List
from datetime import datetime
import uuid
import asyncio
import logging

from langchain.agents import AgentExecutor
from langchain.memory import ConversationBufferWindowMemory
from langchain.schema import BaseMessage, HumanMessage, SystemMessage
from langchain.callbacks.base import BaseCallbackHandler
from langchain.tools import Tool

logger = logging.getLogger(__name__)

class AgentCallbackHandler(BaseCallbackHandler):
    """
    Custom callback handler for agent execution tracking
    """
    def __init__(self, agent_id: str, websocket_manager=None):
        self.agent_id = agent_id
        self.websocket_manager = websocket_manager
        self.start_time = None
        self.tokens_used = 0
        
    def on_llm_start(self, serialized: Dict[str, Any], prompts: List[str], **kwargs) -> None:
        """Called when LLM starts"""
        self.start_time = datetime.now()
        if self.websocket_manager:
            asyncio.create_task(
                self.websocket_manager.broadcast({
                    "type": "agent_thinking",
                    "agent_id": self.agent_id,
                    "message": "Processing request..."
                })
            )
    
    def on_llm_end(self, response, **kwargs) -> None:
        """Called when LLM ends"""
        if hasattr(response, 'llm_output') and response.llm_output:
            if 'token_usage' in response.llm_output:
                self.tokens_used += response.llm_output['token_usage'].get('total_tokens', 0)
    
    def on_tool_start(self, serialized: Dict[str, Any], input_str: str, **kwargs) -> None:
        """Called when tool starts"""
        tool_name = serialized.get("name", "unknown")
        if self.websocket_manager:
            asyncio.create_task(
                self.websocket_manager.broadcast({
                    "type": "tool_execution",
                    "agent_id": self.agent_id,
                    "tool": tool_name,
                    "message": f"Using tool: {tool_name}"
                })
            )
    
    def on_agent_finish(self, finish, **kwargs) -> None:
        """Called when agent finishes"""
        execution_time = (datetime.now() - self.start_time).total_seconds() if self.start_time else 0
        if self.websocket_manager:
            asyncio.create_task(
                self.websocket_manager.broadcast({
                    "type": "agent_complete",
                    "agent_id": self.agent_id,
                    "execution_time": execution_time,
                    "tokens_used": self.tokens_used
                })
            )

class BaseAgent(ABC):
    """
    Abstract base class for all agents in the IntelliFlow system
    """
    
    def __init__(
        self,
        name: str,
        description: str,
        llm_provider,
        tools: List[Tool] = None,
        memory_window: int = 10,
        websocket_manager=None
    ):
        self.agent_id = str(uuid.uuid4())
        self.name = name
        self.description = description
        self.llm = llm_provider
        self.tools = tools or []
        self.websocket_manager = websocket_manager
        
        # Initialize memory
        self.memory = ConversationBufferWindowMemory(
            k=memory_window,
            return_messages=True,
            memory_key="chat_history"
        )
        
        # Initialize callback handler
        self.callback_handler = AgentCallbackHandler(
            agent_id=self.agent_id,
            websocket_manager=websocket_manager
        )
        
        # Agent executor will be initialized in child classes
        self.agent_executor = None
        
        # Metrics tracking
        self.total_executions = 0
        self.successful_executions = 0
        self.failed_executions = 0
        self.total_tokens = 0
        self.total_execution_time = 0
    
    @abstractmethod
    def initialize_agent(self) -> AgentExecutor:
        """
        Initialize the specific agent executor
        Must be implemented by child classes
        """
        pass
    
    @abstractmethod
    def get_system_prompt(self) -> str:
        """
        Get the system prompt for this agent
        Must be implemented by child classes
        """
        pass
    
    async def execute(
        self,
        input_data: Dict[str, Any],
        context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Execute the agent with given input
        """
        try:
            self.total_executions += 1
            start_time = datetime.now()
            
            # Broadcast agent start
            if self.websocket_manager:
                await self.websocket_manager.broadcast({
                    "type": "agent_start",
                    "agent_id": self.agent_id,
                    "agent_name": self.name,
                    "agent_type": self.__class__.__name__,
                    "message": f"{self.name} started processing"
                })
            
            # Prepare input
            task = input_data.get("task", "")
            additional_context = input_data.get("context", {})
            
            # Add context to the conversation if provided
            if context:
                additional_context.update(context)
            
            # Format the input message
            formatted_input = self._format_input(task, additional_context)
            
            # Execute the agent
            result = await self._run_agent(formatted_input)
            
            # Calculate metrics
            execution_time = (datetime.now() - start_time).total_seconds()
            self.total_execution_time += execution_time
            self.successful_executions += 1
            
            # Prepare response
            response = {
                "agent_id": self.agent_id,
                "agent_name": self.name,
                "status": "success",
                "result": result,
                "execution_time": execution_time,
                "tokens_used": self.callback_handler.tokens_used,
                "timestamp": datetime.now().isoformat()
            }
            
            # Broadcast completion
            if self.websocket_manager:
                await self.websocket_manager.broadcast({
                    "type": "agent_complete",
                    "agent_id": self.agent_id,
                    "agent_name": self.name,
                    "status": "success",
                    "execution_time": execution_time
                })
            
            return response
            
        except Exception as e:
            self.failed_executions += 1
            logger.error(f"Agent {self.name} execution failed: {str(e)}")
            
            # Broadcast error
            if self.websocket_manager:
                await self.websocket_manager.broadcast({
                    "type": "agent_error",
                    "agent_id": self.agent_id,
                    "agent_name": self.name,
                    "error": str(e)
                })
            
            return {
                "agent_id": self.agent_id,
                "agent_name": self.name,
                "status": "error",
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }
    
    def _format_input(self, task: str, context: Dict[str, Any]) -> str:
        """
        Format the input for the agent
        """
        formatted = f"Task: {task}\n"
        
        if context:
            formatted += "\nContext:\n"
            for key, value in context.items():
                formatted += f"- {key}: {value}\n"
        
        return formatted
    
    async def _run_agent(self, input_text: str) -> Any:
        """
        Run the agent executor asynchronously
        """
        if not self.agent_executor:
            self.agent_executor = self.initialize_agent()
        
        # Run in executor to avoid blocking
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(
            None,
            self.agent_executor.run,
            input_text,
            [self.callback_handler]
        )
        
        return result
    
    def add_tool(self, tool: Tool):
        """
        Add a tool to the agent's toolkit
        """
        self.tools.append(tool)
        # Reinitialize agent if already initialized
        if self.agent_executor:
            self.agent_executor = self.initialize_agent()
    
    def remove_tool(self, tool_name: str):
        """
        Remove a tool from the agent's toolkit
        """
        self.tools = [t for t in self.tools if t.name != tool_name]
        # Reinitialize agent if already initialized
        if self.agent_executor:
            self.agent_executor = self.initialize_agent()
    
    def get_metrics(self) -> Dict[str, Any]:
        """
        Get agent performance metrics
        """
        success_rate = (
            self.successful_executions / self.total_executions 
            if self.total_executions > 0 else 0
        )
        avg_execution_time = (
            self.total_execution_time / self.total_executions
            if self.total_executions > 0 else 0
        )
        
        return {
            "agent_id": self.agent_id,
            "agent_name": self.name,
            "total_executions": self.total_executions,
            "successful_executions": self.successful_executions,
            "failed_executions": self.failed_executions,
            "success_rate": success_rate,
            "avg_execution_time": avg_execution_time,
            "total_tokens": self.total_tokens
        }
    
    def reset_memory(self):
        """
        Reset the agent's conversation memory
        """
        self.memory.clear()
    
    def get_status(self) -> Dict[str, Any]:
        """
        Get current agent status
        """
        return {
            "agent_id": self.agent_id,
            "agent_name": self.name,
            "agent_type": self.__class__.__name__,
            "description": self.description,
            "tools_count": len(self.tools),
            "tools": [tool.name for tool in self.tools],
            "memory_messages": len(self.memory.chat_memory.messages),
            "metrics": self.get_metrics()
        }
    
    def __str__(self):
        return f"{self.name} ({self.__class__.__name__})"
    
    def __repr__(self):
        return f"<{self.__class__.__name__}(name='{self.name}', id='{self.agent_id}')>"