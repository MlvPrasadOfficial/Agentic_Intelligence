from typing import Generator, Optional
from sqlalchemy.orm import Session
from fastapi import Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer
from jose import JWTError, jwt
from datetime import datetime, timedelta

from ..config import settings
from ..models.database import get_engine, get_session_factory
from ..models.schemas import UserResponse

# OAuth2 scheme
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="api/auth/token", auto_error=False)

# Database session dependency
engine = get_engine(settings.database_url)
SessionLocal = get_session_factory(engine)

def get_db() -> Generator:
    """
    Database session dependency
    """
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

# Authentication dependencies
def create_access_token(data: dict, expires_delta: Optional[timedelta] = None):
    """
    Create JWT access token
    """
    to_encode = data.copy()
    if expires_delta:
        expire = datetime.utcnow() + expires_delta
    else:
        expire = datetime.utcnow() + timedelta(minutes=settings.access_token_expire_minutes)
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, settings.secret_key, algorithm=settings.algorithm)
    return encoded_jwt

async def get_current_user(
    db: Session = Depends(get_db),
    token: str = Depends(oauth2_scheme)
) -> Optional[UserResponse]:
    """
    Get current authenticated user from JWT token
    """
    if not token:
        return None
        
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    
    try:
        payload = jwt.decode(token, settings.secret_key, algorithms=[settings.algorithm])
        user_id: str = payload.get("sub")
        if user_id is None:
            raise credentials_exception
    except JWTError:
        raise credentials_exception
    
    # Here you would fetch the user from database
    # For now, returning a mock user
    return UserResponse(
        id=user_id,
        email=payload.get("email", ""),
        username=payload.get("username", ""),
        full_name=payload.get("full_name", ""),
        is_active=True,
        created_at=datetime.now()
    )

async def get_current_active_user(
    current_user: UserResponse = Depends(get_current_user)
) -> UserResponse:
    """
    Get current active user
    """
    if not current_user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Not authenticated",
            headers={"WWW-Authenticate": "Bearer"},
        )
    if not current_user.is_active:
        raise HTTPException(status_code=400, detail="Inactive user")
    return current_user

# Service dependencies
class ServiceDependencies:
    """
    Container for service dependencies
    """
    def __init__(self):
        self.langsmith_client = None
        self.mcp_client = None
        self.ollama_client = None
        
    async def initialize(self):
        """
        Initialize all service connections
        """
        # Initialize services here
        pass
    
    async def shutdown(self):
        """
        Cleanup service connections
        """
        # Cleanup services here
        pass

# Global service dependencies instance
services = ServiceDependencies()