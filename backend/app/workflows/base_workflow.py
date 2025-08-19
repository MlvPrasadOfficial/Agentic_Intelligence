from typing import TypedDict, Annotated, Sequence, Dict, Any, Optional, List
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
import uuid
import logging

from langgraph.graph import StateGraph, END
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage, AIMessage
import operator

logger = logging.getLogger(__name__)

# Workflow State Definition
class WorkflowState(TypedDict):
    """
    State schema for workflow execution
    """
    workflow_id: str
    messages: Annotated[Sequence[BaseMessage], operator.add]
    current_agent: str
    workflow_status: str
    step_history: List[Dict[str, Any]]
    results: Dict[str, Any]
    errors: List[str]
    metadata: Dict[str, Any]
    context: Dict[str, Any]
    confidence_scores: Dict[str, float]
    retry_count: int
    max_retries: int

class WorkflowStatus(str, Enum):
    """Workflow execution status"""
    INITIALIZING = "initializing"
    RUNNING = "running"
    WAITING_APPROVAL = "waiting_approval"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"

@dataclass
class WorkflowConfig:
    """Configuration for workflow execution"""
    max_retries: int = 3
    timeout_seconds: int = 300
    require_human_approval: bool = False
    approval_threshold: float = 0.7
    parallel_execution: bool = False
    save_checkpoints: bool = True
    enable_monitoring: bool = True

class BaseWorkflow:
    """
    Base class for all LangGraph workflows
    """
    
    def __init__(
        self,
        workflow_name: str,
        workflow_type: str,
        config: Optional[WorkflowConfig] = None,
        websocket_manager=None,
        langsmith_client=None
    ):
        self.workflow_id = str(uuid.uuid4())
        self.workflow_name = workflow_name
        self.workflow_type = workflow_type
        self.config = config or WorkflowConfig()
        self.websocket_manager = websocket_manager
        self.langsmith_client = langsmith_client
        
        # Initialize the state graph
        self.graph = StateGraph(WorkflowState)
        
        # Metrics
        self.start_time = None
        self.end_time = None
        self.total_steps = 0
        self.completed_steps = 0
        
        # Build the workflow
        self.setup_nodes()
        self.setup_edges()
        self.compiled_graph = self.graph.compile()
    
    def setup_nodes(self):
        """
        Define workflow nodes. Override in subclasses.
        """
        # Add common nodes
        self.graph.add_node("initialize", self.initialize_workflow)
        self.graph.add_node("validate_input", self.validate_input)
        self.graph.add_node("finalize", self.finalize_workflow)
        self.graph.add_node("handle_error", self.handle_error)
    
    def setup_edges(self):
        """
        Define workflow edges and routing. Override in subclasses.
        """
        # Set entry point
        self.graph.set_entry_point("initialize")
        
        # Common edge patterns
        self.graph.add_edge("initialize", "validate_input")
        
        # Add conditional edges for error handling
        self.graph.add_conditional_edges(
            "validate_input",
            self.route_after_validation,
            {
                "continue": "process",  # To be defined in subclass
                "error": "handle_error"
            }
        )
        
        # Error handling leads to END
        self.graph.add_edge("handle_error", END)
        
        # Finalize leads to END
        self.graph.add_edge("finalize", END)
    
    async def initialize_workflow(self, state: WorkflowState) -> WorkflowState:
        """
        Initialize workflow execution
        """
        self.start_time = datetime.now()
        
        # Set initial state values
        state["workflow_id"] = self.workflow_id
        state["workflow_status"] = WorkflowStatus.INITIALIZING.value
        state["step_history"] = []
        state["errors"] = []
        state["retry_count"] = 0
        state["max_retries"] = self.config.max_retries
        
        # Add initialization message
        state["messages"].append(
            SystemMessage(content=f"Workflow '{self.workflow_name}' initialized")
        )
        
        # Log step
        self._log_step(state, "initialize", "Workflow initialized")
        
        # Broadcast status
        await self._broadcast_status(
            "workflow_started",
            {"workflow_id": self.workflow_id, "workflow_name": self.workflow_name}
        )
        
        state["workflow_status"] = WorkflowStatus.RUNNING.value
        return state
    
    async def validate_input(self, state: WorkflowState) -> WorkflowState:
        """
        Validate workflow input
        """
        try:
            # Perform validation (override in subclasses for specific validation)
            if not state.get("context"):
                state["errors"].append("No input context provided")
                state["workflow_status"] = WorkflowStatus.FAILED.value
            else:
                state["messages"].append(
                    SystemMessage(content="Input validation successful")
                )
                self._log_step(state, "validate_input", "Input validated successfully")
            
            return state
            
        except Exception as e:
            state["errors"].append(f"Validation error: {str(e)}")
            state["workflow_status"] = WorkflowStatus.FAILED.value
            return state
    
    async def finalize_workflow(self, state: WorkflowState) -> WorkflowState:
        """
        Finalize workflow execution
        """
        self.end_time = datetime.now()
        execution_time = (self.end_time - self.start_time).total_seconds()
        
        state["workflow_status"] = WorkflowStatus.COMPLETED.value
        state["metadata"]["execution_time"] = execution_time
        state["metadata"]["completed_at"] = self.end_time.isoformat()
        
        # Add completion message
        state["messages"].append(
            SystemMessage(
                content=f"Workflow completed successfully in {execution_time:.2f} seconds"
            )
        )
        
        # Log final step
        self._log_step(state, "finalize", "Workflow completed")
        
        # Broadcast completion
        await self._broadcast_status(
            "workflow_completed",
            {
                "workflow_id": self.workflow_id,
                "execution_time": execution_time,
                "results": state.get("results", {})
            }
        )
        
        return state
    
    async def handle_error(self, state: WorkflowState) -> WorkflowState:
        """
        Handle workflow errors
        """
        state["workflow_status"] = WorkflowStatus.FAILED.value
        
        error_message = "; ".join(state.get("errors", ["Unknown error"]))
        state["messages"].append(
            SystemMessage(content=f"Workflow failed: {error_message}")
        )
        
        # Log error
        self._log_step(state, "handle_error", f"Error: {error_message}")
        
        # Broadcast error
        await self._broadcast_status(
            "workflow_error",
            {
                "workflow_id": self.workflow_id,
                "errors": state.get("errors", [])
            }
        )
        
        return state
    
    def route_after_validation(self, state: WorkflowState) -> str:
        """
        Route after input validation
        """
        if state.get("errors"):
            return "error"
        return "continue"
    
    def route_based_on_confidence(self, state: WorkflowState, threshold: float = 0.7) -> str:
        """
        Route based on confidence score
        """
        confidence = state.get("confidence_scores", {}).get("current", 0)
        
        if confidence < threshold:
            if state["retry_count"] < state["max_retries"]:
                state["retry_count"] += 1
                return "retry"
            else:
                return "human_review"
        
        return "continue"
    
    def _log_step(self, state: WorkflowState, step_name: str, description: str):
        """
        Log workflow step
        """
        step_entry = {
            "step_name": step_name,
            "description": description,
            "timestamp": datetime.now().isoformat(),
            "status": state.get("workflow_status"),
            "agent": state.get("current_agent")
        }
        
        if "step_history" not in state:
            state["step_history"] = []
        
        state["step_history"].append(step_entry)
        self.completed_steps += 1
        
        logger.info(f"Workflow {self.workflow_id} - Step: {step_name} - {description}")
    
    async def _broadcast_status(self, event_type: str, data: Dict[str, Any]):
        """
        Broadcast workflow status via WebSocket
        """
        if self.websocket_manager:
            await self.websocket_manager.broadcast({
                "type": event_type,
                "workflow_id": self.workflow_id,
                "workflow_name": self.workflow_name,
                "timestamp": datetime.now().isoformat(),
                "data": data
            })
    
    async def execute(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute the workflow
        """
        try:
            # Initialize state
            initial_state: WorkflowState = {
                "workflow_id": self.workflow_id,
                "messages": [HumanMessage(content=str(input_data))],
                "current_agent": "system",
                "workflow_status": WorkflowStatus.INITIALIZING.value,
                "step_history": [],
                "results": {},
                "errors": [],
                "metadata": {
                    "workflow_type": self.workflow_type,
                    "started_at": datetime.now().isoformat()
                },
                "context": input_data,
                "confidence_scores": {},
                "retry_count": 0,
                "max_retries": self.config.max_retries
            }
            
            # Execute the compiled graph
            final_state = await self.compiled_graph.ainvoke(initial_state)
            
            # Extract results
            return {
                "workflow_id": self.workflow_id,
                "status": final_state.get("workflow_status"),
                "results": final_state.get("results", {}),
                "errors": final_state.get("errors", []),
                "step_history": final_state.get("step_history", []),
                "metadata": final_state.get("metadata", {}),
                "messages": [msg.content for msg in final_state.get("messages", [])]
            }
            
        except Exception as e:
            logger.error(f"Workflow execution failed: {str(e)}")
            return {
                "workflow_id": self.workflow_id,
                "status": WorkflowStatus.FAILED.value,
                "error": str(e)
            }
    
    def get_status(self) -> Dict[str, Any]:
        """
        Get current workflow status
        """
        return {
            "workflow_id": self.workflow_id,
            "workflow_name": self.workflow_name,
            "workflow_type": self.workflow_type,
            "total_steps": self.total_steps,
            "completed_steps": self.completed_steps,
            "progress": (
                self.completed_steps / self.total_steps 
                if self.total_steps > 0 else 0
            )
        }
    
    def visualize(self) -> str:
        """
        Generate workflow visualization (Mermaid diagram)
        """
        # This would generate a Mermaid diagram of the workflow
        # For now, return a placeholder
        return f"""
        graph TD
            A[Initialize] --> B[Validate Input]
            B --> C{Valid?}
            C -->|Yes| D[Process]
            C -->|No| E[Handle Error]
            D --> F[Finalize]
            E --> G[End]
            F --> G[End]
        """