from sqlalchemy import create_engine, Column, String, JSON, DateTime, Float, Boolean, ForeignKey, Text, Integer
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, relationship
from sqlalchemy.sql import func
from datetime import datetime
import uuid

Base = declarative_base()

def generate_uuid():
    return str(uuid.uuid4())

class WorkflowExecution(Base):
    __tablename__ = "workflow_executions"
    
    id = Column(String, primary_key=True, default=generate_uuid)
    workflow_type = Column(String, nullable=False)
    workflow_name = Column(String, nullable=False)
    status = Column(String, default="pending")  # pending, running, completed, failed
    input_data = Column(JSON)
    output_data = Column(JSON)
    error_message = Column(Text)
    langsmith_run_id = Column(String)
    execution_time = Column(Float)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())
    completed_at = Column(DateTime(timezone=True))
    user_id = Column(String, ForeignKey("users.id"))
    
    # Relationships
    user = relationship("User", back_populates="workflow_executions")
    agent_activities = relationship("AgentActivity", back_populates="workflow_execution")
    workflow_steps = relationship("WorkflowStep", back_populates="workflow_execution")

class AgentActivity(Base):
    __tablename__ = "agent_activities"
    
    id = Column(String, primary_key=True, default=generate_uuid)
    agent_id = Column(String, nullable=False)
    agent_type = Column(String, nullable=False)  # research, code, data, communication, planning
    agent_name = Column(String, nullable=False)
    task = Column(JSON)
    input_data = Column(JSON)
    result = Column(JSON)
    status = Column(String, default="pending")  # pending, running, completed, failed
    error_message = Column(Text)
    execution_time = Column(Float)
    tokens_used = Column(Integer)
    workflow_execution_id = Column(String, ForeignKey("workflow_executions.id"))
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    completed_at = Column(DateTime(timezone=True))
    
    # Relationships
    workflow_execution = relationship("WorkflowExecution", back_populates="agent_activities")

class WorkflowStep(Base):
    __tablename__ = "workflow_steps"
    
    id = Column(String, primary_key=True, default=generate_uuid)
    workflow_execution_id = Column(String, ForeignKey("workflow_executions.id"))
    step_number = Column(Integer, nullable=False)
    step_name = Column(String, nullable=False)
    step_type = Column(String, nullable=False)  # agent, decision, parallel, sequential
    status = Column(String, default="pending")
    input_data = Column(JSON)
    output_data = Column(JSON)
    metadata = Column(JSON)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    completed_at = Column(DateTime(timezone=True))
    
    # Relationships
    workflow_execution = relationship("WorkflowExecution", back_populates="workflow_steps")

class User(Base):
    __tablename__ = "users"
    
    id = Column(String, primary_key=True, default=generate_uuid)
    email = Column(String, unique=True, nullable=False)
    username = Column(String, unique=True, nullable=False)
    full_name = Column(String)
    hashed_password = Column(String, nullable=False)
    is_active = Column(Boolean, default=True)
    is_superuser = Column(Boolean, default=False)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())
    
    # Relationships
    workflow_executions = relationship("WorkflowExecution", back_populates="user")
    workflow_templates = relationship("WorkflowTemplate", back_populates="user")

class WorkflowTemplate(Base):
    __tablename__ = "workflow_templates"
    
    id = Column(String, primary_key=True, default=generate_uuid)
    name = Column(String, nullable=False)
    description = Column(Text)
    category = Column(String)  # market_research, code_documentation, customer_support, custom
    template_data = Column(JSON)  # Stores the workflow configuration
    is_public = Column(Boolean, default=False)
    usage_count = Column(Integer, default=0)
    user_id = Column(String, ForeignKey("users.id"))
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())
    
    # Relationships
    user = relationship("User", back_populates="workflow_templates")

class MCPTool(Base):
    __tablename__ = "mcp_tools"
    
    id = Column(String, primary_key=True, default=generate_uuid)
    name = Column(String, unique=True, nullable=False)
    description = Column(Text)
    tool_type = Column(String)  # web_search, database, file_system, api, custom
    configuration = Column(JSON)
    is_enabled = Column(Boolean, default=True)
    usage_count = Column(Integer, default=0)
    created_at = Column(DateTime(timezone=True), server_default=func.now())
    updated_at = Column(DateTime(timezone=True), onupdate=func.now())

class AgentMetrics(Base):
    __tablename__ = "agent_metrics"
    
    id = Column(String, primary_key=True, default=generate_uuid)
    agent_type = Column(String, nullable=False)
    metric_type = Column(String)  # performance, accuracy, cost, tokens
    metric_value = Column(Float)
    metadata = Column(JSON)
    recorded_at = Column(DateTime(timezone=True), server_default=func.now())

# Database session management
def get_engine(database_url: str):
    return create_engine(database_url, echo=True)

def get_session_factory(engine):
    return sessionmaker(autocommit=False, autoflush=False, bind=engine)

def init_db(database_url: str):
    engine = get_engine(database_url)
    Base.metadata.create_all(bind=engine)
    return engine