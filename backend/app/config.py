from pydantic_settings import BaseSettings
from typing import Optional
import os

class Settings(BaseSettings):
    # API Settings
    api_title: str = "IntelliFlow API"
    api_version: str = "1.0.0"
    api_description: str = "Enterprise-Grade Multi-Agent Workflow Automation System"
    
    # Server Settings
    host: str = "0.0.0.0"
    port: int = 8000
    reload: bool = True
    
    # Database
    database_url: str = "postgresql://admin:password@localhost/intelliflow"
    redis_url: str = "redis://localhost:6379"
    
    # LLM Settings
    llama_model: str = "llama3.1"
    ollama_base_url: str = "http://localhost:11434"
    
    # LangSmith
    langsmith_api_key: Optional[str] = None
    langsmith_project: str = "intelliflow"
    langsmith_enabled: bool = False
    
    # MCP Settings
    mcp_server_url: Optional[str] = None
    mcp_enabled: bool = False
    
    # Security
    secret_key: str = "your-secret-key-change-in-production"
    algorithm: str = "HS256"
    access_token_expire_minutes: int = 30
    
    # CORS
    cors_origins: list = ["http://localhost:3000", "http://localhost:3001"]
    
    # Celery
    celery_broker_url: str = "redis://localhost:6379/0"
    celery_result_backend: str = "redis://localhost:6379/0"
    
    # Logging
    log_level: str = "INFO"
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"

settings = Settings()