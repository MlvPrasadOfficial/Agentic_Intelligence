from typing import Dict, Any, Optional
from datetime import datetime
import logging

from langgraph.graph import END
from langchain_core.messages import SystemMessage, HumanMessage

from .base_workflow import BaseWorkflow, WorkflowState, WorkflowConfig, WorkflowStatus

logger = logging.getLogger(__name__)

class MarketResearchWorkflow(BaseWorkflow):
    """
    Market Research Workflow Implementation
    
    Flow:
    1. Initialize → Validate Input
    2. Research Agent → Gather competitor data
    3. Data Analysis Agent → Process findings
    4. Communication Agent → Generate report
    5. Planning Agent → Create action items
    6. Quality Check → Validate results
    7. Finalize → Return results
    
    Conditional routing:
    - If validation fails → END with error
    - If research confidence < 0.7 → Loop back to Research (max 3 times)
    - If quality check fails → Route to human review
    """
    
    def __init__(
        self,
        config: Optional[WorkflowConfig] = None,
        agents: Optional[Dict[str, Any]] = None,
        websocket_manager=None,
        langsmith_client=None
    ):
        # Store agents
        self.agents = agents or {}
        
        # Initialize base workflow
        super().__init__(
            workflow_name="Market Research Workflow",
            workflow_type="market_research",
            config=config,
            websocket_manager=websocket_manager,
            langsmith_client=langsmith_client
        )
        
        self.total_steps = 7
    
    def setup_nodes(self):
        """
        Define workflow nodes for market research
        """
        # Call parent setup first
        super().setup_nodes()
        
        # Add market research specific nodes
        self.graph.add_node("research_agent", self.run_research_agent)
        self.graph.add_node("data_analysis", self.run_data_analysis)
        self.graph.add_node("generate_report", self.run_communication_agent)
        self.graph.add_node("create_action_items", self.run_planning_agent)
        self.graph.add_node("quality_check", self.quality_check)
        self.graph.add_node("human_review", self.human_review)
    
    def setup_edges(self):
        """
        Define workflow edges and routing for market research
        """
        # Entry point
        self.graph.set_entry_point("initialize")
        
        # Main flow
        self.graph.add_edge("initialize", "validate_input")
        
        # Validation routing
        self.graph.add_conditional_edges(
            "validate_input",
            self.route_after_validation,
            {
                "continue": "research_agent",
                "error": "handle_error"
            }
        )
        
        # Research agent routing (with retry logic)
        self.graph.add_conditional_edges(
            "research_agent",
            self.route_after_research,
            {
                "continue": "data_analysis",
                "retry": "research_agent",
                "error": "handle_error"
            }
        )
        
        # Linear flow through analysis and reporting
        self.graph.add_edge("data_analysis", "generate_report")
        self.graph.add_edge("generate_report", "create_action_items")
        self.graph.add_edge("create_action_items", "quality_check")
        
        # Quality check routing
        self.graph.add_conditional_edges(
            "quality_check",
            self.route_after_quality_check,
            {
                "approve": "finalize",
                "review": "human_review",
                "reject": "research_agent"
            }
        )
        
        # Human review routing
        self.graph.add_conditional_edges(
            "human_review",
            self.route_after_human_review,
            {
                "approve": "finalize",
                "reject": "handle_error"
            }
        )
        
        # Terminal nodes
        self.graph.add_edge("handle_error", END)
        self.graph.add_edge("finalize", END)
    
    async def validate_input(self, state: WorkflowState) -> WorkflowState:
        """
        Validate market research input
        """
        try:
            context = state.get("context", {})
            
            # Check required fields
            required_fields = ["company_name", "industry"]
            missing_fields = [f for f in required_fields if f not in context]
            
            if missing_fields:
                state["errors"].append(f"Missing required fields: {missing_fields}")
                state["workflow_status"] = WorkflowStatus.FAILED.value
            else:
                # Set default values
                if "research_depth" not in context:
                    context["research_depth"] = "standard"
                if "competitors_count" not in context:
                    context["competitors_count"] = 5
                
                state["context"] = context
                state["messages"].append(
                    SystemMessage(content=f"Input validated for {context['company_name']} in {context['industry']}")
                )
                self._log_step(state, "validate_input", "Input validation successful")
            
            return state
            
        except Exception as e:
            state["errors"].append(f"Validation error: {str(e)}")
            state["workflow_status"] = WorkflowStatus.FAILED.value
            return state
    
    async def run_research_agent(self, state: WorkflowState) -> WorkflowState:
        """
        Execute research agent to gather competitor data
        """
        try:
            state["current_agent"] = "research_agent"
            context = state.get("context", {})
            
            # Broadcast agent start
            await self._broadcast_status(
                "agent_started",
                {"agent": "research_agent", "task": "Gathering competitor data"}
            )
            
            # Simulate research agent execution
            # In production, this would call the actual research agent
            research_task = {
                "task": f"Research competitors for {context['company_name']} in {context['industry']}",
                "parameters": {
                    "company": context["company_name"],
                    "industry": context["industry"],
                    "depth": context.get("research_depth", "standard"),
                    "max_competitors": context.get("competitors_count", 5)
                }
            }
            
            # Execute agent (placeholder for actual agent execution)
            if self.agents.get("research"):
                result = await self.agents["research"].execute(research_task)
            else:
                # Mock result for demonstration
                result = {
                    "competitors": [
                        {"name": "Competitor 1", "market_share": 0.25},
                        {"name": "Competitor 2", "market_share": 0.20},
                        {"name": "Competitor 3", "market_share": 0.15}
                    ],
                    "market_trends": ["AI adoption", "Cloud migration", "Sustainability focus"],
                    "confidence": 0.85
                }
            
            # Store results
            if "results" not in state:
                state["results"] = {}
            state["results"]["research"] = result
            state["confidence_scores"]["research"] = result.get("confidence", 0.5)
            
            # Add message
            state["messages"].append(
                SystemMessage(content=f"Research completed. Found {len(result.get('competitors', []))} competitors")
            )
            
            # Log step
            self._log_step(state, "research_agent", "Competitor research completed")
            
            return state
            
        except Exception as e:
            state["errors"].append(f"Research agent error: {str(e)}")
            return state
    
    async def run_data_analysis(self, state: WorkflowState) -> WorkflowState:
        """
        Execute data analysis agent to process findings
        """
        try:
            state["current_agent"] = "data_agent"
            research_data = state.get("results", {}).get("research", {})
            
            # Broadcast agent start
            await self._broadcast_status(
                "agent_started",
                {"agent": "data_agent", "task": "Analyzing market data"}
            )
            
            # Execute data analysis agent
            if self.agents.get("data"):
                analysis_task = {
                    "task": "Analyze competitor data and market trends",
                    "data": research_data
                }
                result = await self.agents["data"].execute(analysis_task)
            else:
                # Mock result
                result = {
                    "market_size": "$10B",
                    "growth_rate": "15% YoY",
                    "key_insights": [
                        "Market is highly competitive",
                        "AI integration is a key differentiator",
                        "Customer retention is crucial"
                    ],
                    "swot": {
                        "strengths": ["Brand recognition", "Technology"],
                        "weaknesses": ["Limited geographic presence"],
                        "opportunities": ["Emerging markets", "New segments"],
                        "threats": ["New entrants", "Regulatory changes"]
                    }
                }
            
            # Store results
            state["results"]["analysis"] = result
            state["confidence_scores"]["analysis"] = 0.9
            
            # Add message
            state["messages"].append(
                SystemMessage(content="Data analysis completed. Key insights generated.")
            )
            
            # Log step
            self._log_step(state, "data_analysis", "Market data analysis completed")
            
            return state
            
        except Exception as e:
            state["errors"].append(f"Data analysis error: {str(e)}")
            return state
    
    async def run_communication_agent(self, state: WorkflowState) -> WorkflowState:
        """
        Execute communication agent to generate report
        """
        try:
            state["current_agent"] = "communication_agent"
            
            # Broadcast agent start
            await self._broadcast_status(
                "agent_started",
                {"agent": "communication_agent", "task": "Generating report"}
            )
            
            # Prepare report data
            research_data = state.get("results", {}).get("research", {})
            analysis_data = state.get("results", {}).get("analysis", {})
            
            # Execute communication agent
            if self.agents.get("communication"):
                report_task = {
                    "task": "Generate comprehensive market research report",
                    "research_data": research_data,
                    "analysis_data": analysis_data
                }
                result = await self.agents["communication"].execute(report_task)
            else:
                # Mock report
                result = {
                    "executive_summary": "Market research indicates strong growth potential...",
                    "detailed_findings": "Comprehensive analysis reveals...",
                    "visualizations": ["market_share_chart", "growth_trend_graph"],
                    "recommendations": [
                        "Focus on AI integration",
                        "Expand to emerging markets",
                        "Improve customer retention"
                    ]
                }
            
            # Store results
            state["results"]["report"] = result
            
            # Add message
            state["messages"].append(
                SystemMessage(content="Report generated successfully")
            )
            
            # Log step
            self._log_step(state, "communication_agent", "Report generation completed")
            
            return state
            
        except Exception as e:
            state["errors"].append(f"Report generation error: {str(e)}")
            return state
    
    async def run_planning_agent(self, state: WorkflowState) -> WorkflowState:
        """
        Execute planning agent to create action items
        """
        try:
            state["current_agent"] = "planning_agent"
            
            # Broadcast agent start
            await self._broadcast_status(
                "agent_started",
                {"agent": "planning_agent", "task": "Creating action plan"}
            )
            
            # Execute planning agent
            if self.agents.get("planning"):
                planning_task = {
                    "task": "Create actionable business plan",
                    "report": state.get("results", {}).get("report", {})
                }
                result = await self.agents["planning"].execute(planning_task)
            else:
                # Mock action items
                result = {
                    "action_items": [
                        {
                            "priority": "high",
                            "task": "Develop AI integration roadmap",
                            "timeline": "Q1 2024",
                            "resources": ["Tech team", "Budget: $500k"]
                        },
                        {
                            "priority": "medium",
                            "task": "Market expansion strategy",
                            "timeline": "Q2 2024",
                            "resources": ["Marketing team", "Budget: $200k"]
                        }
                    ],
                    "milestones": [
                        {"date": "2024-03-31", "deliverable": "AI prototype"},
                        {"date": "2024-06-30", "deliverable": "Market entry plan"}
                    ]
                }
            
            # Store results
            state["results"]["action_plan"] = result
            
            # Add message
            state["messages"].append(
                SystemMessage(content=f"Action plan created with {len(result['action_items'])} items")
            )
            
            # Log step
            self._log_step(state, "planning_agent", "Action plan creation completed")
            
            return state
            
        except Exception as e:
            state["errors"].append(f"Planning agent error: {str(e)}")
            return state
    
    async def quality_check(self, state: WorkflowState) -> WorkflowState:
        """
        Perform quality check on the results
        """
        try:
            # Calculate overall quality score
            results = state.get("results", {})
            quality_score = 0
            checks_passed = []
            checks_failed = []
            
            # Check if all required sections are present
            required_sections = ["research", "analysis", "report", "action_plan"]
            for section in required_sections:
                if section in results and results[section]:
                    checks_passed.append(f"{section} present")
                    quality_score += 0.25
                else:
                    checks_failed.append(f"{section} missing")
            
            # Store quality score
            state["confidence_scores"]["quality"] = quality_score
            
            # Add quality check message
            state["messages"].append(
                SystemMessage(
                    content=f"Quality check: Score {quality_score:.2f}. "
                    f"Passed: {len(checks_passed)}, Failed: {len(checks_failed)}"
                )
            )
            
            # Log step
            self._log_step(state, "quality_check", f"Quality check completed (score: {quality_score:.2f})")
            
            return state
            
        except Exception as e:
            state["errors"].append(f"Quality check error: {str(e)}")
            return state
    
    async def human_review(self, state: WorkflowState) -> WorkflowState:
        """
        Route to human review for approval
        """
        state["workflow_status"] = WorkflowStatus.WAITING_APPROVAL.value
        
        # Broadcast human review request
        await self._broadcast_status(
            "human_review_required",
            {
                "reason": "Quality score below threshold",
                "score": state.get("confidence_scores", {}).get("quality", 0)
            }
        )
        
        # In production, this would wait for human input
        # For now, auto-approve after logging
        state["messages"].append(
            SystemMessage(content="Human review requested. Auto-approving for demonstration.")
        )
        
        state["human_approved"] = True
        
        # Log step
        self._log_step(state, "human_review", "Human review completed")
        
        return state
    
    def route_after_validation(self, state: WorkflowState) -> str:
        """Route after input validation"""
        if state.get("errors"):
            return "error"
        return "continue"
    
    def route_after_research(self, state: WorkflowState) -> str:
        """Route after research agent execution"""
        if state.get("errors"):
            return "error"
        
        confidence = state.get("confidence_scores", {}).get("research", 0)
        retry_count = state.get("retry_count", 0)
        
        if confidence < 0.7:
            if retry_count < state.get("max_retries", 3):
                state["retry_count"] = retry_count + 1
                return "retry"
            else:
                return "error"
        
        return "continue"
    
    def route_after_quality_check(self, state: WorkflowState) -> str:
        """Route after quality check"""
        quality_score = state.get("confidence_scores", {}).get("quality", 0)
        
        if quality_score >= 0.8:
            return "approve"
        elif quality_score >= 0.6:
            return "review"
        else:
            return "reject"
    
    def route_after_human_review(self, state: WorkflowState) -> str:
        """Route after human review"""
        if state.get("human_approved"):
            return "approve"
        return "reject"