from pydantic import BaseModel, Field, EmailStr
from typing import Optional, Dict, Any, List
from datetime import datetime
from enum import Enum

# Enums
class WorkflowStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"

class AgentType(str, Enum):
    RESEARCH = "research"
    CODE = "code"
    DATA = "data"
    COMMUNICATION = "communication"
    PLANNING = "planning"

class WorkflowType(str, Enum):
    MARKET_RESEARCH = "market_research"
    CODE_DOCUMENTATION = "code_documentation"
    CUSTOMER_SUPPORT = "customer_support"
    CUSTOM = "custom"

# User Schemas
class UserBase(BaseModel):
    email: EmailStr
    username: str
    full_name: Optional[str] = None

class UserCreate(UserBase):
    password: str

class UserResponse(UserBase):
    id: str
    is_active: bool
    created_at: datetime
    
    class Config:
        from_attributes = True

# Workflow Schemas
class WorkflowExecutionBase(BaseModel):
    workflow_type: WorkflowType
    workflow_name: str
    input_data: Dict[str, Any]

class WorkflowExecutionCreate(WorkflowExecutionBase):
    pass

class WorkflowExecutionUpdate(BaseModel):
    status: Optional[WorkflowStatus] = None
    output_data: Optional[Dict[str, Any]] = None
    error_message: Optional[str] = None

class WorkflowExecutionResponse(WorkflowExecutionBase):
    id: str
    status: WorkflowStatus
    output_data: Optional[Dict[str, Any]] = None
    error_message: Optional[str] = None
    execution_time: Optional[float] = None
    created_at: datetime
    completed_at: Optional[datetime] = None
    
    class Config:
        from_attributes = True

# Agent Schemas
class AgentActivityBase(BaseModel):
    agent_type: AgentType
    agent_name: str
    task: Dict[str, Any]

class AgentActivityCreate(AgentActivityBase):
    workflow_execution_id: str

class AgentActivityResponse(AgentActivityBase):
    id: str
    agent_id: str
    status: str
    result: Optional[Dict[str, Any]] = None
    error_message: Optional[str] = None
    execution_time: Optional[float] = None
    tokens_used: Optional[int] = None
    created_at: datetime
    
    class Config:
        from_attributes = True

# Workflow Template Schemas
class WorkflowTemplateBase(BaseModel):
    name: str
    description: Optional[str] = None
    category: WorkflowType
    template_data: Dict[str, Any]
    is_public: bool = False

class WorkflowTemplateCreate(WorkflowTemplateBase):
    pass

class WorkflowTemplateResponse(WorkflowTemplateBase):
    id: str
    usage_count: int
    created_at: datetime
    updated_at: Optional[datetime] = None
    
    class Config:
        from_attributes = True

# WebSocket Message Schemas
class WSMessage(BaseModel):
    type: str  # status_update, agent_activity, error, completion
    data: Dict[str, Any]
    timestamp: datetime = Field(default_factory=datetime.now)

class AgentStatusUpdate(BaseModel):
    agent_id: str
    agent_type: AgentType
    status: str
    message: Optional[str] = None
    progress: Optional[float] = None  # 0.0 to 1.0

# MCP Tool Schemas
class MCPToolBase(BaseModel):
    name: str
    description: str
    tool_type: str
    configuration: Dict[str, Any]

class MCPToolCreate(MCPToolBase):
    pass

class MCPToolResponse(MCPToolBase):
    id: str
    is_enabled: bool
    usage_count: int
    created_at: datetime
    
    class Config:
        from_attributes = True

# Workflow Step Schemas
class WorkflowStepBase(BaseModel):
    step_name: str
    step_type: str
    input_data: Optional[Dict[str, Any]] = None

class WorkflowStepResponse(WorkflowStepBase):
    id: str
    step_number: int
    status: str
    output_data: Optional[Dict[str, Any]] = None
    created_at: datetime
    completed_at: Optional[datetime] = None
    
    class Config:
        from_attributes = True

# Analytics Schemas
class AgentMetricsResponse(BaseModel):
    agent_type: AgentType
    total_executions: int
    success_rate: float
    avg_execution_time: float
    total_tokens_used: int
    
class WorkflowAnalytics(BaseModel):
    total_workflows: int
    completed_workflows: int
    failed_workflows: int
    avg_execution_time: float
    success_rate: float
    workflows_by_type: Dict[str, int]

# Request/Response Models for API
class WorkflowRequest(BaseModel):
    workflow_type: WorkflowType
    parameters: Dict[str, Any]
    config: Optional[Dict[str, Any]] = None

class WorkflowResponse(BaseModel):
    workflow_id: str
    status: WorkflowStatus
    message: str
    websocket_url: Optional[str] = None

class HealthCheckResponse(BaseModel):
    status: str
    version: str
    timestamp: datetime
    services: Dict[str, bool]  # ollama, redis, postgres status

# Additional schemas for main.py API
class AgentStatus(BaseModel):
    agent_id: str
    agent_name: str
    agent_type: str
    status: str
    description: str
    tools: List[str]
    metrics: Dict[str, Any]

class AgentRequest(BaseModel):
    agent_type: str
    task: str
    parameters: Optional[Dict[str, Any]] = None

class AgentResponse(BaseModel):
    execution_id: str
    agent_type: str
    agent_name: str
    status: str
    result: Dict[str, Any]
    execution_time: float
    timestamp: str

class SystemStatus(BaseModel):
    status: str
    active_agents: int
    active_workflows: int
    active_connections: int
    system_metrics: Dict[str, Any]
    timestamp: str

class UserCreateRequest(BaseModel):
    username: str
    email: str
    password: str

class LoginRequest(BaseModel):
    username: str
    password: str

class TokenResponse(BaseModel):
    access_token: str
    token_type: str
    expires_in: int