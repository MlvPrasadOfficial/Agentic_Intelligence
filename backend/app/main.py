import asyncio
import logging
import uuid
from contextlib import asynccontextmanager
from datetime import datetime
from typing import Dict, Any, List, Optional
import json

from fastapi import FastAPI, HTTPException, Depends, WebSocket, WebSocketDisconnect, BackgroundTasks, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
import uvicorn

# Import application modules
from .config import Settings
from .models.schemas import (
    WorkflowRequest, WorkflowResponse, AgentRequest, AgentResponse,
    AgentStatus, WorkflowStatus, SystemStatus, UserCreateRequest,
    UserResponse, LoginRequest, TokenResponse
)
from .agents import (
    ResearchAgent, CodeAgent, DataAgent, CommunicationAgent, PlanningAgent
)
from .workflows import MarketResearchWorkflow
from .services.llm_service import LLMService
from .services.langsmith_service import LangSmithService
from .services.mcp_service import MCPService
from .api.websocket_manager import websocket_manager, WebSocketMessage, MessageType
from .api.dependencies import get_current_user, get_db
from .models.database import init_db

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Settings
settings = Settings()

# Security
security = HTTPBearer()

# Application state
app_state = {
    "agents": {},
    "workflows": {},
    "active_sessions": {},
    "system_metrics": {}
}

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager"""
    logger.info("Starting IntelliFlow API...")
    
    try:
        # Initialize database
        await init_db()
        logger.info("Database initialized")
        
        # Initialize services
        llm_service = LLMService(settings)
        langsmith_service = LangSmithService(settings) if settings.langsmith_api_key else None
        mcp_service = MCPService(settings) if settings.mcp_server_url else None
        
        # Store services in app state
        app.state.llm_service = llm_service
        app.state.langsmith_service = langsmith_service
        app.state.mcp_service = mcp_service
        
        # Initialize agents
        await initialize_agents(llm_service, langsmith_service)
        
        # Start WebSocket monitoring
        await websocket_manager.start_monitoring()
        
        logger.info("IntelliFlow API started successfully")
        
        yield
        
    except Exception as e:
        logger.error(f"Failed to start application: {str(e)}")
        raise
    finally:
        # Cleanup
        logger.info("Shutting down IntelliFlow API...")
        await websocket_manager.stop_monitoring()
        logger.info("IntelliFlow API shutdown complete")

# Create FastAPI app
app = FastAPI(
    title=settings.api_title,
    version=settings.api_version,
    description="IntelliFlow - Enterprise-Grade Multi-Agent Workflow Automation System",
    lifespan=lifespan
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://127.0.0.1:3000"],  # Frontend URLs
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

async def initialize_agents(llm_service: LLMService, langsmith_service: Optional[LangSmithService]):
    """Initialize all agents"""
    try:
        agents = {
            "research": ResearchAgent(llm_service.get_llm(), websocket_manager),
            "code": CodeAgent(llm_service.get_llm(), websocket_manager),
            "data": DataAgent(llm_service.get_llm(), websocket_manager),
            "communication": CommunicationAgent(llm_service.get_llm(), websocket_manager),
            "planning": PlanningAgent(llm_service.get_llm(), websocket_manager)
        }
        
        app_state["agents"] = agents
        logger.info(f"Initialized {len(agents)} agents")
        
    except Exception as e:
        logger.error(f"Error initializing agents: {str(e)}")
        raise

# Pydantic models for API requests/responses
class HealthResponse(BaseModel):
    status: str
    timestamp: str
    version: str
    agents_status: Dict[str, str]
    services_status: Dict[str, str]

class WorkflowExecuteRequest(BaseModel):
    workflow_type: str = Field(..., description="Type of workflow to execute")
    input_data: Dict[str, Any] = Field(..., description="Input data for the workflow")
    client_id: Optional[str] = Field(None, description="Client ID for WebSocket updates")
    
class AgentExecuteRequest(BaseModel):
    agent_type: str = Field(..., description="Type of agent to execute")
    task: str = Field(..., description="Task for the agent to perform")
    context: Optional[Dict[str, Any]] = Field(None, description="Additional context")
    client_id: Optional[str] = Field(None, description="Client ID for WebSocket updates")

# API Routes

@app.get("/", response_model=Dict[str, str])
async def root():
    """Root endpoint"""
    return {
        "message": "Welcome to IntelliFlow API",
        "version": settings.api_version,
        "docs": "/docs",
        "health": "/health"
    }

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint"""
    agents_status = {}
    for name, agent in app_state["agents"].items():
        try:
            # Simple health check - verify agent is accessible
            agents_status[name] = "healthy" if agent else "unavailable"
        except Exception:
            agents_status[name] = "error"
    
    services_status = {
        "llm_service": "healthy" if hasattr(app.state, 'llm_service') else "unavailable",
        "langsmith_service": "healthy" if hasattr(app.state, 'langsmith_service') and app.state.langsmith_service else "disabled",
        "mcp_service": "healthy" if hasattr(app.state, 'mcp_service') and app.state.mcp_service else "disabled"
    }
    
    return HealthResponse(
        status="healthy",
        timestamp=datetime.now().isoformat(),
        version=settings.api_version,
        agents_status=agents_status,
        services_status=services_status
    )

# Agent Management Endpoints

@app.get("/api/agents", response_model=List[AgentStatus])
async def list_agents():
    """List all available agents and their status"""
    agent_statuses = []
    
    for name, agent in app_state["agents"].items():
        try:
            status_info = agent.get_status()
            agent_statuses.append(AgentStatus(
                agent_id=status_info["agent_id"],
                agent_name=status_info["agent_name"],
                agent_type=status_info["agent_type"],
                status="idle",  # Default status
                description=status_info["description"],
                tools=status_info["tools"],
                metrics=status_info["metrics"]
            ))
        except Exception as e:
            logger.error(f"Error getting status for agent {name}: {str(e)}")
            agent_statuses.append(AgentStatus(
                agent_id=name,
                agent_name=name.title(),
                agent_type=name,
                status="error",
                description=f"Error: {str(e)}",
                tools=[],
                metrics={}
            ))
    
    return agent_statuses

@app.get("/api/agents/{agent_type}", response_model=AgentStatus)
async def get_agent_status(agent_type: str):
    """Get status of a specific agent"""
    if agent_type not in app_state["agents"]:
        raise HTTPException(status_code=404, detail=f"Agent type '{agent_type}' not found")
    
    agent = app_state["agents"][agent_type]
    try:
        status_info = agent.get_status()
        return AgentStatus(
            agent_id=status_info["agent_id"],
            agent_name=status_info["agent_name"],
            agent_type=status_info["agent_type"],
            status="idle",
            description=status_info["description"],
            tools=status_info["tools"],
            metrics=status_info["metrics"]
        )
    except Exception as e:
        logger.error(f"Error getting status for agent {agent_type}: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error getting agent status: {str(e)}")

@app.post("/api/agents/{agent_type}/execute", response_model=AgentResponse)
async def execute_agent(
    agent_type: str,
    request: AgentExecuteRequest,
    background_tasks: BackgroundTasks
):
    """Execute a specific agent"""
    if agent_type not in app_state["agents"]:
        raise HTTPException(status_code=404, detail=f"Agent type '{agent_type}' not found")
    
    agent = app_state["agents"][agent_type]
    execution_id = str(uuid.uuid4())
    
    try:
        # Prepare input data
        input_data = {
            "task": request.task,
            "context": request.context or {}
        }
        
        # Execute agent
        result = await agent.execute(input_data, request.context)
        
        response = AgentResponse(
            execution_id=execution_id,
            agent_type=agent_type,
            agent_name=agent.name,
            status="completed" if result.get("status") == "success" else "failed",
            result=result,
            execution_time=result.get("execution_time", 0),
            timestamp=datetime.now().isoformat()
        )
        
        return response
        
    except Exception as e:
        logger.error(f"Error executing agent {agent_type}: {str(e)}")
        return AgentResponse(
            execution_id=execution_id,
            agent_type=agent_type,
            agent_name=agent.name if agent else agent_type,
            status="failed",
            result={"error": str(e)},
            execution_time=0,
            timestamp=datetime.now().isoformat()
        )

# Workflow Management Endpoints

@app.get("/api/workflows", response_model=List[str])
async def list_workflow_types():
    """List available workflow types"""
    return [
        "market_research",
        "code_documentation", 
        "customer_support",
        "data_analysis",
        "content_creation"
    ]

@app.post("/api/workflows/execute", response_model=WorkflowResponse)
async def execute_workflow(
    request: WorkflowExecuteRequest,
    background_tasks: BackgroundTasks
):
    """Execute a workflow"""
    workflow_id = str(uuid.uuid4())
    
    try:
        # Create workflow based on type
        if request.workflow_type == "market_research":
            workflow = MarketResearchWorkflow(
                workflow_id=workflow_id,
                agents=app_state["agents"],
                websocket_manager=websocket_manager
            )
        else:
            raise HTTPException(
                status_code=400, 
                detail=f"Workflow type '{request.workflow_type}' not implemented yet"
            )
        
        # Store workflow in app state
        app_state["workflows"][workflow_id] = {
            "workflow": workflow,
            "status": "running",
            "created_at": datetime.now(),
            "client_id": request.client_id
        }
        
        # Execute workflow in background
        background_tasks.add_task(
            execute_workflow_background,
            workflow,
            request.input_data,
            workflow_id,
            request.client_id
        )
        
        return WorkflowResponse(
            workflow_id=workflow_id,
            workflow_type=request.workflow_type,
            status="started",
            result=None,
            execution_time=0,
            timestamp=datetime.now().isoformat()
        )
        
    except Exception as e:
        logger.error(f"Error starting workflow {request.workflow_type}: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error starting workflow: {str(e)}")

async def execute_workflow_background(
    workflow, 
    input_data: Dict[str, Any], 
    workflow_id: str, 
    client_id: Optional[str]
):
    """Execute workflow in background"""
    try:
        # Notify workflow start
        await websocket_manager.workflow_started(
            workflow_id, 
            workflow.__class__.__name__, 
            client_id
        )
        
        # Execute workflow
        start_time = datetime.now()
        result = await workflow.execute(input_data)
        execution_time = (datetime.now() - start_time).total_seconds()
        
        # Update workflow state
        if workflow_id in app_state["workflows"]:
            app_state["workflows"][workflow_id]["status"] = "completed"
            app_state["workflows"][workflow_id]["result"] = result
            app_state["workflows"][workflow_id]["execution_time"] = execution_time
        
        # Notify workflow completion
        await websocket_manager.workflow_completed(
            workflow_id,
            result,
            execution_time,
            client_id
        )
        
    except Exception as e:
        logger.error(f"Error executing workflow {workflow_id}: {str(e)}")
        
        # Update workflow state
        if workflow_id in app_state["workflows"]:
            app_state["workflows"][workflow_id]["status"] = "failed"
            app_state["workflows"][workflow_id]["error"] = str(e)
        
        # Notify workflow error
        await websocket_manager.send_system_status(
            "error",
            f"Workflow {workflow_id} failed: {str(e)}",
            client_id
        )

@app.get("/api/workflows/{workflow_id}", response_model=WorkflowResponse)
async def get_workflow_status(workflow_id: str):
    """Get status of a specific workflow"""
    if workflow_id not in app_state["workflows"]:
        raise HTTPException(status_code=404, detail=f"Workflow '{workflow_id}' not found")
    
    workflow_data = app_state["workflows"][workflow_id]
    
    return WorkflowResponse(
        workflow_id=workflow_id,
        workflow_type=workflow_data["workflow"].__class__.__name__,
        status=workflow_data["status"],
        result=workflow_data.get("result"),
        execution_time=workflow_data.get("execution_time", 0),
        timestamp=workflow_data["created_at"].isoformat()
    )

# System Management Endpoints

@app.get("/api/system/status", response_model=SystemStatus)
async def get_system_status():
    """Get overall system status"""
    try:
        # Get connection stats
        connection_stats = await websocket_manager.connection_manager.get_connection_stats()
        
        # Count active workflows
        active_workflows = sum(
            1 for w in app_state["workflows"].values() 
            if w["status"] == "running"
        )
        
        return SystemStatus(
            status="operational",
            active_agents=len(app_state["agents"]),
            active_workflows=active_workflows,
            active_connections=connection_stats["active_connections"],
            system_metrics={
                "uptime": "unknown",  # Would track actual uptime
                "memory_usage": "unknown",
                "cpu_usage": "unknown"
            },
            timestamp=datetime.now().isoformat()
        )
    
    except Exception as e:
        logger.error(f"Error getting system status: {str(e)}")
        return SystemStatus(
            status="error",
            active_agents=0,
            active_workflows=0,
            active_connections=0,
            system_metrics={"error": str(e)},
            timestamp=datetime.now().isoformat()
        )

@app.get("/api/system/metrics")
async def get_system_metrics():
    """Get detailed system metrics"""
    try:
        import psutil
        
        return {
            "timestamp": datetime.now().isoformat(),
            "system": {
                "cpu_percent": psutil.cpu_percent(interval=1),
                "memory": {
                    "total": psutil.virtual_memory().total,
                    "available": psutil.virtual_memory().available,
                    "percent": psutil.virtual_memory().percent
                },
                "disk": {
                    "total": psutil.disk_usage('/').total,
                    "free": psutil.disk_usage('/').free,
                    "percent": psutil.disk_usage('/').percent
                }
            },
            "application": {
                "agents": {
                    name: agent.get_metrics() 
                    for name, agent in app_state["agents"].items()
                },
                "workflows": {
                    "total": len(app_state["workflows"]),
                    "active": sum(
                        1 for w in app_state["workflows"].values() 
                        if w["status"] == "running"
                    ),
                    "completed": sum(
                        1 for w in app_state["workflows"].values() 
                        if w["status"] == "completed"
                    ),
                    "failed": sum(
                        1 for w in app_state["workflows"].values() 
                        if w["status"] == "failed"
                    )
                }
            },
            "websocket": await websocket_manager.connection_manager.get_connection_stats()
        }
        
    except Exception as e:
        logger.error(f"Error getting system metrics: {str(e)}")
        return {"error": str(e), "timestamp": datetime.now().isoformat()}

# WebSocket Endpoint

@app.websocket("/ws/{client_id}")
async def websocket_endpoint(websocket: WebSocket, client_id: str, session_id: Optional[str] = None):
    """WebSocket endpoint for real-time updates"""
    connection_successful = await websocket_manager.connection_manager.connect(
        websocket, client_id, session_id
    )
    
    if not connection_successful:
        await websocket.close(code=status.WS_1003_UNSUPPORTED_DATA)
        return
    
    try:
        while True:
            # Receive message from client
            data = await websocket.receive_text()
            
            try:
                message_data = json.loads(data)
                await websocket_manager.handle_user_message(client_id, message_data)
                
            except json.JSONDecodeError:
                # Send error message
                error_message = WebSocketMessage.create(
                    MessageType.SYSTEM_STATUS,
                    {
                        "status": "error",
                        "message": "Invalid JSON format"
                    },
                    client_id=client_id
                )
                await websocket_manager.connection_manager.send_personal_message(
                    client_id, error_message
                )
            
    except WebSocketDisconnect:
        await websocket_manager.connection_manager.disconnect(client_id)
    except Exception as e:
        logger.error(f"WebSocket error for client {client_id}: {str(e)}")
        await websocket_manager.connection_manager.disconnect(client_id)

# Authentication Endpoints (Basic implementation)

@app.post("/api/auth/register", response_model=UserResponse)
async def register_user(request: UserCreateRequest):
    """Register a new user"""
    # This is a placeholder implementation
    # In production, you would hash passwords and store in database
    return UserResponse(
        user_id=str(uuid.uuid4()),
        username=request.username,
        email=request.email,
        created_at=datetime.now().isoformat()
    )

@app.post("/api/auth/login", response_model=TokenResponse)
async def login_user(request: LoginRequest):
    """Login user and return token"""
    # This is a placeholder implementation
    # In production, you would verify credentials against database
    if request.username == "demo" and request.password == "demo":
        return TokenResponse(
            access_token=f"demo_token_{uuid.uuid4()}",
            token_type="bearer",
            expires_in=3600
        )
    else:
        raise HTTPException(
            status_code=401,
            detail="Invalid credentials"
        )

@app.get("/api/auth/me", response_model=UserResponse)
async def get_current_user_info(
    credentials: HTTPAuthorizationCredentials = Depends(security)
):
    """Get current user information"""
    # This is a placeholder implementation
    return UserResponse(
        user_id="demo_user",
        username="demo",
        email="demo@intelliflow.com",
        created_at=datetime.now().isoformat()
    )

# Error Handlers

@app.exception_handler(HTTPException)
async def http_exception_handler(request, exc):
    """Handle HTTP exceptions"""
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "error": True,
            "message": exc.detail,
            "timestamp": datetime.now().isoformat()
        }
    )

@app.exception_handler(Exception)
async def general_exception_handler(request, exc):
    """Handle general exceptions"""
    logger.error(f"Unhandled exception: {str(exc)}")
    return JSONResponse(
        status_code=500,
        content={
            "error": True,
            "message": "Internal server error",
            "timestamp": datetime.now().isoformat()
        }
    )

# Development server
if __name__ == "__main__":
    uvicorn.run(
        "app.main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )