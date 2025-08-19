"""
LangSmith monitoring and observability service
"""
from typing import Dict, Any, Optional, List, Callable
from functools import wraps
from datetime import datetime
import logging
import traceback
import uuid
import json
import asyncio

logger = logging.getLogger(__name__)

class LangSmithMonitor:
    """
    LangSmith monitoring service for tracking agent and workflow execution
    """
    
    def __init__(self, api_key: Optional[str] = None, project: str = "intelliflow"):
        self.api_key = api_key
        self.project = project
        self.enabled = bool(api_key)
        self.client = None
        
        # Local storage for runs when LangSmith is not available
        self.local_runs: List[Dict[str, Any]] = []
        self.active_runs: Dict[str, Dict[str, Any]] = {}
        
        if self.enabled:
            try:
                from langsmith import Client
                self.client = Client(api_key=api_key)
                logger.info(f"LangSmith monitoring enabled for project: {project}")
            except Exception as e:
                logger.warning(f"Failed to initialize LangSmith client: {str(e)}")
                self.enabled = False
    
    def create_run(
        self,
        name: str,
        run_type: str,
        inputs: Dict[str, Any],
        parent_run_id: Optional[str] = None,
        tags: Optional[List[str]] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Create a new monitoring run
        """
        run_id = str(uuid.uuid4())
        run_data = {
            "id": run_id,
            "name": name,
            "run_type": run_type,
            "inputs": inputs,
            "parent_run_id": parent_run_id,
            "tags": tags or [],
            "metadata": metadata or {},
            "start_time": datetime.now().isoformat(),
            "project_name": self.project
        }
        
        if self.enabled and self.client:
            try:
                # Create run in LangSmith
                langsmith_run = self.client.create_run(
                    name=name,
                    run_type=run_type,
                    project_name=self.project,
                    inputs=inputs,
                    parent_run_id=parent_run_id,
                    tags=tags,
                    extra=metadata
                )
                run_data["langsmith_id"] = str(langsmith_run.id)
            except Exception as e:
                logger.error(f"Failed to create LangSmith run: {str(e)}")
        
        # Store run locally
        self.active_runs[run_id] = run_data
        
        return run_id
    
    def update_run(
        self,
        run_id: str,
        outputs: Optional[Dict[str, Any]] = None,
        error: Optional[str] = None,
        end_time: Optional[datetime] = None,
        events: Optional[List[Dict[str, Any]]] = None
    ):
        """
        Update an existing run
        """
        if run_id not in self.active_runs:
            logger.warning(f"Run {run_id} not found in active runs")
            return
        
        run_data = self.active_runs[run_id]
        
        # Update local data
        if outputs:
            run_data["outputs"] = outputs
        if error:
            run_data["error"] = error
            run_data["status"] = "error"
        else:
            run_data["status"] = "success"
        if end_time:
            run_data["end_time"] = end_time.isoformat()
            # Calculate duration
            start_time = datetime.fromisoformat(run_data["start_time"])
            run_data["duration_seconds"] = (end_time - start_time).total_seconds()
        if events:
            run_data.setdefault("events", []).extend(events)
        
        # Update in LangSmith if available
        if self.enabled and self.client and "langsmith_id" in run_data:
            try:
                self.client.update_run(
                    run_data["langsmith_id"],
                    outputs=outputs,
                    error=error,
                    end_time=end_time
                )
            except Exception as e:
                logger.error(f"Failed to update LangSmith run: {str(e)}")
        
        # Move to completed runs if finished
        if "end_time" in run_data:
            self.local_runs.append(run_data)
            del self.active_runs[run_id]
    
    def log_event(self, run_id: str, event_type: str, data: Dict[str, Any]):
        """
        Log an event within a run
        """
        event = {
            "type": event_type,
            "data": data,
            "timestamp": datetime.now().isoformat()
        }
        
        if run_id in self.active_runs:
            self.active_runs[run_id].setdefault("events", []).append(event)
    
    def trace_agent(self, agent_name: str, agent_type: str = "agent"):
        """
        Decorator to trace agent execution
        """
        def decorator(func):
            @wraps(func)
            async def async_wrapper(*args, **kwargs):
                run_id = self.create_run(
                    name=f"{agent_name}_execution",
                    run_type=agent_type,
                    inputs={"args": str(args), "kwargs": kwargs},
                    tags=[agent_name, agent_type]
                )
                
                try:
                    result = await func(*args, **kwargs)
                    self.update_run(
                        run_id,
                        outputs={"result": result},
                        end_time=datetime.now()
                    )
                    return result
                except Exception as e:
                    self.update_run(
                        run_id,
                        error=str(e),
                        end_time=datetime.now()
                    )
                    raise
            
            @wraps(func)
            def sync_wrapper(*args, **kwargs):
                run_id = self.create_run(
                    name=f"{agent_name}_execution",
                    run_type=agent_type,
                    inputs={"args": str(args), "kwargs": kwargs},
                    tags=[agent_name, agent_type]
                )
                
                try:
                    result = func(*args, **kwargs)
                    self.update_run(
                        run_id,
                        outputs={"result": result},
                        end_time=datetime.now()
                    )
                    return result
                except Exception as e:
                    self.update_run(
                        run_id,
                        error=str(e),
                        end_time=datetime.now()
                    )
                    raise
            
            if asyncio.iscoroutinefunction(func):
                return async_wrapper
            else:
                return sync_wrapper
        
        return decorator
    
    def trace_workflow(self, workflow_name: str):
        """
        Decorator to trace workflow execution
        """
        return self.trace_agent(workflow_name, "workflow")
    
    def trace_tool(self, tool_name: str):
        """
        Decorator to trace tool execution
        """
        return self.trace_agent(tool_name, "tool")
    
    def get_run_history(
        self,
        limit: int = 100,
        run_type: Optional[str] = None,
        status: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Get run history with optional filters
        """
        runs = self.local_runs[-limit:]
        
        if run_type:
            runs = [r for r in runs if r.get("run_type") == run_type]
        
        if status:
            runs = [r for r in runs if r.get("status") == status]
        
        return runs
    
    def get_metrics(self) -> Dict[str, Any]:
        """
        Get monitoring metrics
        """
        all_runs = self.local_runs + list(self.active_runs.values())
        
        if not all_runs:
            return {
                "total_runs": 0,
                "success_rate": 0,
                "avg_duration": 0,
                "runs_by_type": {},
                "runs_by_status": {}
            }
        
        # Calculate metrics
        total_runs = len(all_runs)
        successful_runs = sum(1 for r in all_runs if r.get("status") == "success")
        
        # Average duration for completed runs
        completed_runs = [r for r in all_runs if "duration_seconds" in r]
        avg_duration = (
            sum(r["duration_seconds"] for r in completed_runs) / len(completed_runs)
            if completed_runs else 0
        )
        
        # Runs by type
        runs_by_type = {}
        for run in all_runs:
            run_type = run.get("run_type", "unknown")
            runs_by_type[run_type] = runs_by_type.get(run_type, 0) + 1
        
        # Runs by status
        runs_by_status = {}
        for run in all_runs:
            status = run.get("status", "active")
            runs_by_status[status] = runs_by_status.get(status, 0) + 1
        
        return {
            "total_runs": total_runs,
            "success_rate": successful_runs / total_runs if total_runs > 0 else 0,
            "avg_duration": avg_duration,
            "runs_by_type": runs_by_type,
            "runs_by_status": runs_by_status,
            "active_runs": len(self.active_runs)
        }
    
    def get_agent_performance(self, agent_name: str) -> Dict[str, Any]:
        """
        Get performance metrics for a specific agent
        """
        agent_runs = [
            r for r in self.local_runs
            if agent_name in r.get("tags", [])
        ]
        
        if not agent_runs:
            return {
                "agent_name": agent_name,
                "total_executions": 0,
                "success_rate": 0,
                "avg_duration": 0
            }
        
        successful = sum(1 for r in agent_runs if r.get("status") == "success")
        durations = [r["duration_seconds"] for r in agent_runs if "duration_seconds" in r]
        
        return {
            "agent_name": agent_name,
            "total_executions": len(agent_runs),
            "success_rate": successful / len(agent_runs),
            "avg_duration": sum(durations) / len(durations) if durations else 0,
            "min_duration": min(durations) if durations else 0,
            "max_duration": max(durations) if durations else 0
        }
    
    def export_traces(self, format: str = "json") -> str:
        """
        Export traces in specified format
        """
        all_runs = self.local_runs + list(self.active_runs.values())
        
        if format == "json":
            return json.dumps(all_runs, indent=2, default=str)
        elif format == "csv":
            import csv
            import io
            
            output = io.StringIO()
            if all_runs:
                writer = csv.DictWriter(
                    output,
                    fieldnames=["id", "name", "run_type", "status", "start_time", "end_time", "duration_seconds"]
                )
                writer.writeheader()
                for run in all_runs:
                    writer.writerow({
                        "id": run.get("id"),
                        "name": run.get("name"),
                        "run_type": run.get("run_type"),
                        "status": run.get("status", "active"),
                        "start_time": run.get("start_time"),
                        "end_time": run.get("end_time"),
                        "duration_seconds": run.get("duration_seconds")
                    })
            
            return output.getvalue()
        else:
            raise ValueError(f"Unsupported export format: {format}")
    
    def clear_history(self):
        """
        Clear local run history
        """
        self.local_runs.clear()
        self.active_runs.clear()
    
    async def get_langsmith_url(self, run_id: str) -> Optional[str]:
        """
        Get LangSmith UI URL for a run
        """
        if run_id in self.active_runs:
            run_data = self.active_runs[run_id]
        else:
            run_data = next((r for r in self.local_runs if r["id"] == run_id), None)
        
        if run_data and "langsmith_id" in run_data:
            return f"https://smith.langchain.com/public/{self.project}/r/{run_data['langsmith_id']}"
        
        return None