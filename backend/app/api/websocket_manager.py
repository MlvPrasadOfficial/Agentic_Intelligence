import asyncio
import json
import logging
from typing import Dict, List, Set, Any, Optional
from datetime import datetime
from dataclasses import dataclass, asdict
from enum import Enum
import uuid

from fastapi import WebSocket, WebSocketDisconnect
from starlette.websockets import WebSocketState

logger = logging.getLogger(__name__)

class MessageType(Enum):
    """WebSocket message types"""
    # Agent events
    AGENT_START = "agent_start"
    AGENT_THINKING = "agent_thinking"
    AGENT_COMPLETE = "agent_complete" 
    AGENT_ERROR = "agent_error"
    AGENT_PROGRESS = "agent_progress"
    
    # Workflow events
    WORKFLOW_START = "workflow_start"
    WORKFLOW_PROGRESS = "workflow_progress"
    WORKFLOW_COMPLETE = "workflow_complete"
    WORKFLOW_ERROR = "workflow_error"
    WORKFLOW_PAUSE = "workflow_pause"
    WORKFLOW_RESUME = "workflow_resume"
    
    # Tool events
    TOOL_START = "tool_start"
    TOOL_COMPLETE = "tool_complete"
    TOOL_ERROR = "tool_error"
    
    # System events
    SYSTEM_STATUS = "system_status"
    SYSTEM_METRICS = "system_metrics"
    
    # User events
    USER_MESSAGE = "user_message"
    USER_ACTION = "user_action"
    
    # Monitoring events
    RESOURCE_USAGE = "resource_usage"
    PERFORMANCE_METRICS = "performance_metrics"

@dataclass
class WebSocketMessage:
    """WebSocket message structure"""
    type: str
    data: Dict[str, Any]
    timestamp: str
    client_id: Optional[str] = None
    session_id: Optional[str] = None
    
    @classmethod
    def create(cls, message_type: MessageType, data: Dict[str, Any], 
               client_id: Optional[str] = None, session_id: Optional[str] = None):
        """Create a new WebSocket message"""
        return cls(
            type=message_type.value,
            data=data,
            timestamp=datetime.now().isoformat(),
            client_id=client_id,
            session_id=session_id
        )
    
    def to_json(self) -> str:
        """Convert message to JSON string"""
        return json.dumps(asdict(self), default=str)

class ConnectionManager:
    """Manages WebSocket connections"""
    
    def __init__(self):
        # Active connections by client ID
        self.active_connections: Dict[str, WebSocket] = {}
        
        # Client sessions
        self.client_sessions: Dict[str, Dict[str, Any]] = {}
        
        # Room-based connections (for group messaging)
        self.rooms: Dict[str, Set[str]] = {}
        
        # Message history for reconnection
        self.message_history: Dict[str, List[WebSocketMessage]] = {}
        
        # Connection metadata
        self.connection_metadata: Dict[str, Dict[str, Any]] = {}
        
        # Event subscriptions
        self.subscriptions: Dict[str, Set[MessageType]] = {}
    
    async def connect(self, websocket: WebSocket, client_id: str, 
                     session_id: Optional[str] = None) -> bool:
        """Connect a client"""
        try:
            await websocket.accept()
            
            # Store connection
            self.active_connections[client_id] = websocket
            
            # Initialize session
            if session_id:
                self.client_sessions[client_id] = {
                    "session_id": session_id,
                    "connected_at": datetime.now(),
                    "last_activity": datetime.now()
                }
            
            # Initialize message history
            if client_id not in self.message_history:
                self.message_history[client_id] = []
            
            # Initialize subscriptions (subscribe to all by default)
            self.subscriptions[client_id] = set(MessageType)
            
            # Store connection metadata
            self.connection_metadata[client_id] = {
                "connected_at": datetime.now(),
                "user_agent": websocket.headers.get("user-agent"),
                "ip_address": websocket.client.host if websocket.client else None
            }
            
            logger.info(f"Client {client_id} connected")
            
            # Send connection confirmation
            await self.send_personal_message(
                client_id,
                WebSocketMessage.create(
                    MessageType.SYSTEM_STATUS,
                    {
                        "status": "connected",
                        "client_id": client_id,
                        "server_time": datetime.now().isoformat()
                    },
                    client_id=client_id
                )
            )
            
            return True
            
        except Exception as e:
            logger.error(f"Error connecting client {client_id}: {str(e)}")
            return False
    
    async def disconnect(self, client_id: str):
        """Disconnect a client"""
        if client_id in self.active_connections:
            websocket = self.active_connections[client_id]
            
            # Close WebSocket if still open
            if websocket.client_state == WebSocketState.CONNECTED:
                try:
                    await websocket.close()
                except Exception as e:
                    logger.warning(f"Error closing WebSocket for {client_id}: {str(e)}")
            
            # Clean up
            del self.active_connections[client_id]
            
            # Remove from rooms
            for room_id in list(self.rooms.keys()):
                if client_id in self.rooms[room_id]:
                    self.rooms[room_id].remove(client_id)
                    if not self.rooms[room_id]:  # Remove empty rooms
                        del self.rooms[room_id]
            
            # Update session
            if client_id in self.client_sessions:
                self.client_sessions[client_id]["disconnected_at"] = datetime.now()
            
            logger.info(f"Client {client_id} disconnected")
    
    async def send_personal_message(self, client_id: str, message: WebSocketMessage):
        """Send message to specific client"""
        if client_id in self.active_connections:
            websocket = self.active_connections[client_id]
            
            try:
                # Check if client is subscribed to this message type
                message_type = MessageType(message.type)
                if message_type not in self.subscriptions.get(client_id, set()):
                    return
                
                await websocket.send_text(message.to_json())
                
                # Add to message history
                self.message_history[client_id].append(message)
                
                # Limit message history size
                if len(self.message_history[client_id]) > 1000:
                    self.message_history[client_id] = self.message_history[client_id][-500:]
                
                # Update last activity
                if client_id in self.client_sessions:
                    self.client_sessions[client_id]["last_activity"] = datetime.now()
                
            except WebSocketDisconnect:
                await self.disconnect(client_id)
            except Exception as e:
                logger.error(f"Error sending message to {client_id}: {str(e)}")
                await self.disconnect(client_id)
    
    async def broadcast(self, message: WebSocketMessage, exclude: Optional[List[str]] = None):
        """Broadcast message to all connected clients"""
        exclude = exclude or []
        
        for client_id in list(self.active_connections.keys()):
            if client_id not in exclude:
                await self.send_personal_message(client_id, message)
    
    async def broadcast_to_room(self, room_id: str, message: WebSocketMessage):
        """Broadcast message to all clients in a room"""
        if room_id in self.rooms:
            for client_id in list(self.rooms[room_id]):
                await self.send_personal_message(client_id, message)
    
    def join_room(self, client_id: str, room_id: str):
        """Add client to a room"""
        if room_id not in self.rooms:
            self.rooms[room_id] = set()
        self.rooms[room_id].add(client_id)
    
    def leave_room(self, client_id: str, room_id: str):
        """Remove client from a room"""
        if room_id in self.rooms and client_id in self.rooms[room_id]:
            self.rooms[room_id].remove(client_id)
            if not self.rooms[room_id]:
                del self.rooms[room_id]
    
    def subscribe(self, client_id: str, message_types: List[MessageType]):
        """Subscribe client to specific message types"""
        if client_id not in self.subscriptions:
            self.subscriptions[client_id] = set()
        self.subscriptions[client_id].update(message_types)
    
    def unsubscribe(self, client_id: str, message_types: List[MessageType]):
        """Unsubscribe client from specific message types"""
        if client_id in self.subscriptions:
            self.subscriptions[client_id].difference_update(message_types)
    
    def get_client_count(self) -> int:
        """Get number of active connections"""
        return len(self.active_connections)
    
    def get_room_clients(self, room_id: str) -> List[str]:
        """Get list of clients in a room"""
        return list(self.rooms.get(room_id, set()))
    
    def get_client_info(self, client_id: str) -> Optional[Dict[str, Any]]:
        """Get client information"""
        if client_id not in self.active_connections:
            return None
        
        return {
            "client_id": client_id,
            "connected": True,
            "session": self.client_sessions.get(client_id, {}),
            "metadata": self.connection_metadata.get(client_id, {}),
            "subscriptions": [mt.value for mt in self.subscriptions.get(client_id, set())],
            "message_count": len(self.message_history.get(client_id, []))
        }
    
    async def get_connection_stats(self) -> Dict[str, Any]:
        """Get connection statistics"""
        return {
            "active_connections": len(self.active_connections),
            "total_rooms": len(self.rooms),
            "total_messages": sum(len(history) for history in self.message_history.values()),
            "rooms": {
                room_id: len(clients) 
                for room_id, clients in self.rooms.items()
            },
            "clients": [
                self.get_client_info(client_id) 
                for client_id in self.active_connections.keys()
            ]
        }

class WebSocketManager:
    """Main WebSocket manager for the application"""
    
    def __init__(self):
        self.connection_manager = ConnectionManager()
        
        # Agent event handlers
        self.agent_handlers: Dict[str, callable] = {}
        
        # Workflow event handlers  
        self.workflow_handlers: Dict[str, callable] = {}
        
        # System monitoring
        self.monitoring_enabled = True
        self.metrics_interval = 30  # seconds
        
        # Start background tasks
        self._monitoring_task = None
    
    async def start_monitoring(self):
        """Start background monitoring tasks"""
        if self.monitoring_enabled and not self._monitoring_task:
            self._monitoring_task = asyncio.create_task(self._monitoring_loop())
    
    async def stop_monitoring(self):
        """Stop background monitoring tasks"""
        if self._monitoring_task:
            self._monitoring_task.cancel()
            try:
                await self._monitoring_task
            except asyncio.CancelledError:
                pass
            self._monitoring_task = None
    
    async def _monitoring_loop(self):
        """Background monitoring loop"""
        while True:
            try:
                await asyncio.sleep(self.metrics_interval)
                
                # Collect system metrics
                metrics = await self._collect_metrics()
                
                # Broadcast metrics to subscribed clients
                await self.connection_manager.broadcast(
                    WebSocketMessage.create(
                        MessageType.SYSTEM_METRICS,
                        metrics
                    )
                )
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in monitoring loop: {str(e)}")
                await asyncio.sleep(5)
    
    async def _collect_metrics(self) -> Dict[str, Any]:
        """Collect system metrics"""
        import psutil
        
        return {
            "timestamp": datetime.now().isoformat(),
            "system": {
                "cpu_percent": psutil.cpu_percent(),
                "memory_percent": psutil.virtual_memory().percent,
                "disk_usage": psutil.disk_usage('/').percent
            },
            "websocket": {
                "active_connections": self.connection_manager.get_client_count(),
                "total_rooms": len(self.connection_manager.rooms),
                "total_messages": sum(
                    len(history) 
                    for history in self.connection_manager.message_history.values()
                )
            }
        }
    
    # Agent event methods
    async def agent_started(self, agent_id: str, agent_name: str, agent_type: str, 
                           client_id: Optional[str] = None):
        """Notify that an agent has started"""
        message = WebSocketMessage.create(
            MessageType.AGENT_START,
            {
                "agent_id": agent_id,
                "agent_name": agent_name,
                "agent_type": agent_type,
                "message": f"{agent_name} started processing"
            },
            client_id=client_id
        )
        
        if client_id:
            await self.connection_manager.send_personal_message(client_id, message)
        else:
            await self.connection_manager.broadcast(message)
    
    async def agent_thinking(self, agent_id: str, message: str, 
                            client_id: Optional[str] = None):
        """Notify that an agent is thinking/processing"""
        ws_message = WebSocketMessage.create(
            MessageType.AGENT_THINKING,
            {
                "agent_id": agent_id,
                "message": message,
                "status": "thinking"
            },
            client_id=client_id
        )
        
        if client_id:
            await self.connection_manager.send_personal_message(client_id, ws_message)
        else:
            await self.connection_manager.broadcast(ws_message)
    
    async def agent_completed(self, agent_id: str, agent_name: str, result: Any,
                             execution_time: float, client_id: Optional[str] = None):
        """Notify that an agent has completed"""
        message = WebSocketMessage.create(
            MessageType.AGENT_COMPLETE,
            {
                "agent_id": agent_id,
                "agent_name": agent_name,
                "result": str(result)[:500] if result else None,  # Limit result size
                "execution_time": execution_time,
                "status": "completed"
            },
            client_id=client_id
        )
        
        if client_id:
            await self.connection_manager.send_personal_message(client_id, message)
        else:
            await self.connection_manager.broadcast(message)
    
    async def agent_error(self, agent_id: str, agent_name: str, error: str,
                         client_id: Optional[str] = None):
        """Notify that an agent encountered an error"""
        message = WebSocketMessage.create(
            MessageType.AGENT_ERROR,
            {
                "agent_id": agent_id,
                "agent_name": agent_name,
                "error": error,
                "status": "error"
            },
            client_id=client_id
        )
        
        if client_id:
            await self.connection_manager.send_personal_message(client_id, message)
        else:
            await self.connection_manager.broadcast(message)
    
    # Workflow event methods
    async def workflow_started(self, workflow_id: str, workflow_type: str, 
                              client_id: Optional[str] = None):
        """Notify that a workflow has started"""
        message = WebSocketMessage.create(
            MessageType.WORKFLOW_START,
            {
                "workflow_id": workflow_id,
                "workflow_type": workflow_type,
                "status": "started",
                "message": f"Workflow '{workflow_type}' started"
            },
            client_id=client_id
        )
        
        if client_id:
            await self.connection_manager.send_personal_message(client_id, message)
        else:
            await self.connection_manager.broadcast(message)
    
    async def workflow_progress(self, workflow_id: str, progress: float, 
                               current_step: str, client_id: Optional[str] = None):
        """Notify workflow progress"""
        message = WebSocketMessage.create(
            MessageType.WORKFLOW_PROGRESS,
            {
                "workflow_id": workflow_id,
                "progress": progress,
                "current_step": current_step,
                "status": "in_progress"
            },
            client_id=client_id
        )
        
        if client_id:
            await self.connection_manager.send_personal_message(client_id, message)
        else:
            await self.connection_manager.broadcast(message)
    
    async def workflow_completed(self, workflow_id: str, result: Any,
                                execution_time: float, client_id: Optional[str] = None):
        """Notify that a workflow has completed"""
        message = WebSocketMessage.create(
            MessageType.WORKFLOW_COMPLETE,
            {
                "workflow_id": workflow_id,
                "result": result,
                "execution_time": execution_time,
                "status": "completed"
            },
            client_id=client_id
        )
        
        if client_id:
            await self.connection_manager.send_personal_message(client_id, message)
        else:
            await self.connection_manager.broadcast(message)
    
    # Tool event methods
    async def tool_started(self, tool_name: str, agent_id: str, 
                          client_id: Optional[str] = None):
        """Notify that a tool execution has started"""
        message = WebSocketMessage.create(
            MessageType.TOOL_START,
            {
                "tool_name": tool_name,
                "agent_id": agent_id,
                "message": f"Using tool: {tool_name}"
            },
            client_id=client_id
        )
        
        if client_id:
            await self.connection_manager.send_personal_message(client_id, message)
        else:
            await self.connection_manager.broadcast(message)
    
    async def tool_completed(self, tool_name: str, agent_id: str, result: Any,
                            client_id: Optional[str] = None):
        """Notify that a tool execution has completed"""
        message = WebSocketMessage.create(
            MessageType.TOOL_COMPLETE,
            {
                "tool_name": tool_name,
                "agent_id": agent_id,
                "result": str(result)[:200] if result else None  # Limit result size
            },
            client_id=client_id
        )
        
        if client_id:
            await self.connection_manager.send_personal_message(client_id, message)
        else:
            await self.connection_manager.broadcast(message)
    
    # Utility methods
    async def send_system_status(self, status: str, message: str, 
                                client_id: Optional[str] = None):
        """Send system status update"""
        ws_message = WebSocketMessage.create(
            MessageType.SYSTEM_STATUS,
            {
                "status": status,
                "message": message,
                "timestamp": datetime.now().isoformat()
            },
            client_id=client_id
        )
        
        if client_id:
            await self.connection_manager.send_personal_message(client_id, ws_message)
        else:
            await self.connection_manager.broadcast(ws_message)
    
    async def handle_user_message(self, client_id: str, message_data: Dict[str, Any]):
        """Handle incoming user message"""
        # Process user message and potentially trigger agents/workflows
        message_type = message_data.get("type", "message")
        
        if message_type == "subscribe":
            # Handle subscription changes
            types = [MessageType(t) for t in message_data.get("message_types", [])]
            self.connection_manager.subscribe(client_id, types)
        
        elif message_type == "unsubscribe":
            # Handle unsubscription
            types = [MessageType(t) for t in message_data.get("message_types", [])]
            self.connection_manager.unsubscribe(client_id, types)
        
        elif message_type == "join_room":
            # Handle room joining
            room_id = message_data.get("room_id")
            if room_id:
                self.connection_manager.join_room(client_id, room_id)
        
        elif message_type == "leave_room":
            # Handle room leaving
            room_id = message_data.get("room_id")
            if room_id:
                self.connection_manager.leave_room(client_id, room_id)

# Global WebSocket manager instance
websocket_manager = WebSocketManager()