"""
MCP (Model Context Protocol) Service for tool integration
"""
from typing import Dict, Any, List, Optional, Callable
from dataclasses import dataclass
import json
import asyncio
import logging
from datetime import datetime
import httpx

logger = logging.getLogger(__name__)

@dataclass
class MCPTool:
    """MCP Tool definition"""
    name: str
    description: str
    parameters: Dict[str, Any]
    function: Optional[Callable] = None
    tool_type: str = "custom"
    enabled: bool = True
    usage_count: int = 0

class MCPService:
    """
    Model Context Protocol service for managing and executing tools
    """
    
    def __init__(self, server_url: Optional[str] = None):
        self.server_url = server_url
        self.tools_registry: Dict[str, MCPTool] = {}
        self.execution_history: List[Dict[str, Any]] = []
        self.is_connected = False
        self._client = None
        
        # Initialize default tools
        self._register_default_tools()
    
    async def connect(self) -> bool:
        """
        Connect to MCP server
        """
        try:
            if self.server_url:
                self._client = httpx.AsyncClient(base_url=self.server_url)
                # Test connection
                response = await self._client.get("/health")
                if response.status_code == 200:
                    self.is_connected = True
                    logger.info(f"Connected to MCP server at {self.server_url}")
                    return True
            else:
                # Run in standalone mode without server
                self.is_connected = True
                logger.info("MCP service running in standalone mode")
                return True
        except Exception as e:
            logger.error(f"Failed to connect to MCP server: {str(e)}")
            self.is_connected = False
            return False
    
    async def disconnect(self):
        """
        Disconnect from MCP server
        """
        if self._client:
            await self._client.aclose()
            self._client = None
        self.is_connected = False
    
    def _register_default_tools(self):
        """
        Register default MCP tools
        """
        # Web Search Tool
        self.register_tool(
            name="web_search",
            description="Search the web for information",
            parameters={
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Search query"},
                    "max_results": {"type": "integer", "default": 5}
                },
                "required": ["query"]
            },
            function=self._web_search_function,
            tool_type="web"
        )
        
        # Database Query Tool
        self.register_tool(
            name="database_query",
            description="Query database for information",
            parameters={
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "SQL query"},
                    "database": {"type": "string", "default": "default"}
                },
                "required": ["query"]
            },
            function=self._database_query_function,
            tool_type="database"
        )
        
        # File System Tool
        self.register_tool(
            name="file_system",
            description="Perform file system operations",
            parameters={
                "type": "object",
                "properties": {
                    "operation": {"type": "string", "enum": ["read", "write", "list", "delete"]},
                    "path": {"type": "string", "description": "File path"},
                    "content": {"type": "string", "description": "Content for write operation"}
                },
                "required": ["operation", "path"]
            },
            function=self._file_system_function,
            tool_type="file_system"
        )
        
        # API Connector Tool
        self.register_tool(
            name="api_connector",
            description="Connect to external APIs",
            parameters={
                "type": "object",
                "properties": {
                    "url": {"type": "string", "description": "API endpoint URL"},
                    "method": {"type": "string", "enum": ["GET", "POST", "PUT", "DELETE"]},
                    "headers": {"type": "object", "description": "Request headers"},
                    "body": {"type": "object", "description": "Request body"}
                },
                "required": ["url", "method"]
            },
            function=self._api_connector_function,
            tool_type="api"
        )
        
        # Calculator Tool
        self.register_tool(
            name="calculator",
            description="Perform mathematical calculations",
            parameters={
                "type": "object",
                "properties": {
                    "expression": {"type": "string", "description": "Mathematical expression"}
                },
                "required": ["expression"]
            },
            function=self._calculator_function,
            tool_type="utility"
        )
    
    def register_tool(
        self,
        name: str,
        description: str,
        parameters: Dict[str, Any],
        function: Optional[Callable] = None,
        tool_type: str = "custom",
        enabled: bool = True
    ) -> bool:
        """
        Register a new tool with MCP
        """
        try:
            tool = MCPTool(
                name=name,
                description=description,
                parameters=parameters,
                function=function,
                tool_type=tool_type,
                enabled=enabled
            )
            
            self.tools_registry[name] = tool
            logger.info(f"Registered MCP tool: {name}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to register tool {name}: {str(e)}")
            return False
    
    def unregister_tool(self, name: str) -> bool:
        """
        Unregister a tool from MCP
        """
        if name in self.tools_registry:
            del self.tools_registry[name]
            logger.info(f"Unregistered MCP tool: {name}")
            return True
        return False
    
    async def execute_tool(
        self,
        tool_name: str,
        parameters: Dict[str, Any],
        context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Execute a registered tool
        """
        start_time = datetime.now()
        
        try:
            # Check if tool exists
            if tool_name not in self.tools_registry:
                raise ValueError(f"Tool '{tool_name}' not found in registry")
            
            tool = self.tools_registry[tool_name]
            
            # Check if tool is enabled
            if not tool.enabled:
                raise ValueError(f"Tool '{tool_name}' is disabled")
            
            # Execute tool function
            if tool.function:
                if asyncio.iscoroutinefunction(tool.function):
                    result = await tool.function(parameters, context)
                else:
                    result = tool.function(parameters, context)
            else:
                # If no local function, try to execute via MCP server
                if self._client and self.server_url:
                    response = await self._client.post(
                        "/tools/execute",
                        json={
                            "tool": tool_name,
                            "parameters": parameters,
                            "context": context
                        }
                    )
                    result = response.json()
                else:
                    raise ValueError(f"No implementation found for tool '{tool_name}'")
            
            # Update usage count
            tool.usage_count += 1
            
            # Record execution
            execution_time = (datetime.now() - start_time).total_seconds()
            execution_record = {
                "tool_name": tool_name,
                "parameters": parameters,
                "result": result,
                "execution_time": execution_time,
                "timestamp": datetime.now().isoformat(),
                "success": True
            }
            self.execution_history.append(execution_record)
            
            return {
                "success": True,
                "result": result,
                "execution_time": execution_time
            }
            
        except Exception as e:
            logger.error(f"Tool execution failed for '{tool_name}': {str(e)}")
            
            # Record failed execution
            execution_record = {
                "tool_name": tool_name,
                "parameters": parameters,
                "error": str(e),
                "timestamp": datetime.now().isoformat(),
                "success": False
            }
            self.execution_history.append(execution_record)
            
            return {
                "success": False,
                "error": str(e)
            }
    
    def get_available_tools(self, tool_type: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Get list of available tools
        """
        tools = []
        for name, tool in self.tools_registry.items():
            if tool_type is None or tool.tool_type == tool_type:
                tools.append({
                    "name": name,
                    "description": tool.description,
                    "parameters": tool.parameters,
                    "type": tool.tool_type,
                    "enabled": tool.enabled,
                    "usage_count": tool.usage_count
                })
        return tools
    
    def enable_tool(self, tool_name: str) -> bool:
        """
        Enable a tool
        """
        if tool_name in self.tools_registry:
            self.tools_registry[tool_name].enabled = True
            return True
        return False
    
    def disable_tool(self, tool_name: str) -> bool:
        """
        Disable a tool
        """
        if tool_name in self.tools_registry:
            self.tools_registry[tool_name].enabled = False
            return True
        return False
    
    def get_execution_history(self, limit: int = 100) -> List[Dict[str, Any]]:
        """
        Get tool execution history
        """
        return self.execution_history[-limit:]
    
    def get_tool_statistics(self) -> Dict[str, Any]:
        """
        Get tool usage statistics
        """
        total_executions = len(self.execution_history)
        successful_executions = sum(1 for e in self.execution_history if e.get("success"))
        failed_executions = total_executions - successful_executions
        
        tool_usage = {}
        for record in self.execution_history:
            tool_name = record.get("tool_name")
            if tool_name:
                if tool_name not in tool_usage:
                    tool_usage[tool_name] = {"success": 0, "failure": 0}
                if record.get("success"):
                    tool_usage[tool_name]["success"] += 1
                else:
                    tool_usage[tool_name]["failure"] += 1
        
        return {
            "total_tools": len(self.tools_registry),
            "enabled_tools": sum(1 for t in self.tools_registry.values() if t.enabled),
            "total_executions": total_executions,
            "successful_executions": successful_executions,
            "failed_executions": failed_executions,
            "success_rate": successful_executions / total_executions if total_executions > 0 else 0,
            "tool_usage": tool_usage
        }
    
    # Tool implementation functions
    async def _web_search_function(self, parameters: Dict[str, Any], context: Optional[Dict[str, Any]]) -> Any:
        """Web search implementation"""
        query = parameters.get("query")
        max_results = parameters.get("max_results", 5)
        
        # Mock implementation - replace with actual search API
        return {
            "query": query,
            "results": [
                {"title": f"Result {i+1}", "url": f"https://example.com/{i}", "snippet": f"Snippet for {query}"}
                for i in range(max_results)
            ]
        }
    
    async def _database_query_function(self, parameters: Dict[str, Any], context: Optional[Dict[str, Any]]) -> Any:
        """Database query implementation"""
        query = parameters.get("query")
        database = parameters.get("database", "default")
        
        # Mock implementation - replace with actual database connection
        return {
            "query": query,
            "database": database,
            "rows": [],
            "row_count": 0
        }
    
    async def _file_system_function(self, parameters: Dict[str, Any], context: Optional[Dict[str, Any]]) -> Any:
        """File system operations implementation"""
        import os
        
        operation = parameters.get("operation")
        path = parameters.get("path")
        
        if operation == "read":
            if os.path.exists(path):
                with open(path, 'r') as f:
                    return {"content": f.read()}
            else:
                raise FileNotFoundError(f"File not found: {path}")
        
        elif operation == "write":
            content = parameters.get("content", "")
            with open(path, 'w') as f:
                f.write(content)
            return {"message": f"File written: {path}"}
        
        elif operation == "list":
            if os.path.isdir(path):
                files = os.listdir(path)
                return {"files": files}
            else:
                raise NotADirectoryError(f"Not a directory: {path}")
        
        elif operation == "delete":
            if os.path.exists(path):
                os.remove(path)
                return {"message": f"File deleted: {path}"}
            else:
                raise FileNotFoundError(f"File not found: {path}")
        
        else:
            raise ValueError(f"Unknown operation: {operation}")
    
    async def _api_connector_function(self, parameters: Dict[str, Any], context: Optional[Dict[str, Any]]) -> Any:
        """API connector implementation"""
        url = parameters.get("url")
        method = parameters.get("method", "GET")
        headers = parameters.get("headers", {})
        body = parameters.get("body")
        
        async with httpx.AsyncClient() as client:
            response = await client.request(
                method=method,
                url=url,
                headers=headers,
                json=body if body else None
            )
            
            return {
                "status_code": response.status_code,
                "headers": dict(response.headers),
                "body": response.json() if response.headers.get("content-type", "").startswith("application/json") else response.text
            }
    
    def _calculator_function(self, parameters: Dict[str, Any], context: Optional[Dict[str, Any]]) -> Any:
        """Calculator implementation"""
        expression = parameters.get("expression")
        
        # Safe evaluation of mathematical expressions
        import ast
        import operator as op
        
        operators = {
            ast.Add: op.add,
            ast.Sub: op.sub,
            ast.Mult: op.mul,
            ast.Div: op.truediv,
            ast.Pow: op.pow,
            ast.USub: op.neg,
        }
        
        def eval_expr(expr):
            if isinstance(expr, ast.Num):
                return expr.n
            elif isinstance(expr, ast.BinOp):
                return operators[type(expr.op)](
                    eval_expr(expr.left),
                    eval_expr(expr.right)
                )
            elif isinstance(expr, ast.UnaryOp):
                return operators[type(expr.op)](eval_expr(expr.operand))
            else:
                raise TypeError(f"Unsupported expression type: {type(expr)}")
        
        tree = ast.parse(expression, mode='eval')
        result = eval_expr(tree.body)
        
        return {"expression": expression, "result": result}