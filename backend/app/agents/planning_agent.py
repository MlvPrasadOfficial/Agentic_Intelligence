from typing import Dict, Any, List, Optional, Tuple
from langchain.agents import initialize_agent, AgentType
from langchain.tools import Tool
from langchain.schema import SystemMessage
import json
from datetime import datetime, timedelta
from dataclasses import dataclass
import uuid
import math

from .base_agent import BaseAgent

@dataclass
class Task:
    """Task data structure"""
    id: str
    name: str
    description: str
    priority: str
    estimated_hours: float
    dependencies: List[str]
    resources: List[str]
    status: str = "planned"
    progress: float = 0.0
    assigned_agent: Optional[str] = None
    start_date: Optional[datetime] = None
    end_date: Optional[datetime] = None

@dataclass
class Resource:
    """Resource data structure"""
    id: str
    name: str
    type: str
    availability: float
    cost_per_hour: float
    skills: List[str]

@dataclass
class Project:
    """Project data structure"""
    id: str
    name: str
    description: str
    start_date: datetime
    target_end_date: datetime
    budget: float
    tasks: List[Task]
    resources: List[Resource]
    status: str = "planning"

class PlanningAgent(BaseAgent):
    """
    Specialized agent for task planning, project management, resource allocation,
    and timeline estimation
    """
    
    def __init__(self, llm_provider, websocket_manager=None):
        # Planning-specific tools
        tools = [
            Tool(
                name="task_breakdown",
                description="Break down complex tasks into smaller, manageable subtasks",
                func=self._breakdown_tasks
            ),
            Tool(
                name="resource_allocator",
                description="Allocate resources efficiently across tasks and projects",
                func=self._allocate_resources
            ),
            Tool(
                name="timeline_estimator",
                description="Estimate realistic timelines for tasks and projects",
                func=self._estimate_timeline
            ),
            Tool(
                name="priority_analyzer",
                description="Analyze and prioritize tasks based on various criteria",
                func=self._analyze_priorities
            ),
            Tool(
                name="risk_assessor",
                description="Assess risks and create mitigation strategies",
                func=self._assess_risks
            ),
            Tool(
                name="dependency_mapper",
                description="Map task dependencies and identify critical paths",
                func=self._map_dependencies
            ),
            Tool(
                name="schedule_optimizer",
                description="Optimize project schedules for efficiency",
                func=self._optimize_schedule
            ),
            Tool(
                name="milestone_planner",
                description="Create project milestones and checkpoints",
                func=self._plan_milestones
            ),
            Tool(
                name="capacity_planner",
                description="Plan team capacity and workload distribution",
                func=self._plan_capacity
            ),
            Tool(
                name="budget_estimator",
                description="Estimate project budgets and costs",
                func=self._estimate_budget
            )
        ]
        
        super().__init__(
            name="Task Planning Agent",
            description="Specialized agent for project planning, resource allocation, and timeline management",
            llm_provider=llm_provider,
            tools=tools,
            websocket_manager=websocket_manager
        )
        
        # Initialize planning data structures
        self.projects: Dict[str, Project] = {}
        self.global_resources: List[Resource] = []
        self.planning_templates = self._initialize_templates()
    
    def get_system_prompt(self) -> str:
        return """You are a Task Planning Agent specialized in:
        1. Breaking down complex tasks into manageable subtasks
        2. Resource allocation and capacity planning
        3. Timeline estimation and scheduling
        4. Priority analysis and task sequencing
        5. Risk assessment and mitigation planning
        6. Dependency mapping and critical path analysis
        7. Schedule optimization and efficiency improvement
        8. Milestone planning and progress tracking
        9. Budget estimation and cost planning
        10. Team capacity and workload management
        
        Planning principles you follow:
        - Start with clear objectives and success criteria
        - Break large tasks into smaller, actionable items
        - Consider resource constraints and availability
        - Account for dependencies and blockers
        - Build in buffer time for uncertainties
        - Plan for regular reviews and adjustments
        - Document assumptions and constraints
        - Focus on value delivery and outcomes
        
        Use the available tools to create comprehensive, realistic project plans."""
    
    def initialize_agent(self):
        """Initialize the planning agent executor"""
        return initialize_agent(
            tools=self.tools,
            llm=self.llm,
            agent=AgentType.STRUCTURED_CHAT_ZERO_SHOT_REACT_DESCRIPTION,
            memory=self.memory,
            verbose=True,
            handle_parsing_errors=True,
            callbacks=[self.callback_handler]
        )
    
    def _breakdown_tasks(self, task_input: str) -> str:
        """
        Break down complex tasks into smaller, manageable subtasks
        """
        try:
            request = self._parse_task_request(task_input)
            main_task = request.get('task', task_input)
            project_type = request.get('project_type', 'general')
            complexity = request.get('complexity', 'medium')
            
            # Analyze the task and break it down
            breakdown = self._analyze_and_breakdown_task(main_task, project_type, complexity)
            
            # Create task hierarchy
            task_hierarchy = self._create_task_hierarchy(breakdown)
            
            # Estimate effort for each subtask
            estimated_tasks = self._estimate_subtask_effort(task_hierarchy)
            
            # Add dependencies
            tasks_with_dependencies = self._add_task_dependencies(estimated_tasks)
            
            result = {
                "main_task": main_task,
                "project_type": project_type,
                "complexity": complexity,
                "total_subtasks": len(tasks_with_dependencies),
                "estimated_total_hours": sum(task['estimated_hours'] for task in tasks_with_dependencies),
                "task_breakdown": tasks_with_dependencies,
                "work_breakdown_structure": self._create_wbs(tasks_with_dependencies),
                "recommended_approach": self._recommend_approach(project_type, complexity)
            }
            
            return json.dumps(result, indent=2, default=str)
            
        except Exception as e:
            return f"Error breaking down tasks: {str(e)}"
    
    def _allocate_resources(self, allocation_request: str) -> str:
        """
        Allocate resources efficiently across tasks and projects
        """
        try:
            request = self._parse_allocation_request(allocation_request)
            tasks = request.get('tasks', [])
            available_resources = request.get('resources', [])
            constraints = request.get('constraints', {})
            
            # Parse resources
            resources = [self._parse_resource(r) for r in available_resources]
            
            # Analyze resource requirements for tasks
            task_requirements = self._analyze_resource_requirements(tasks)
            
            # Perform resource allocation
            allocation_plan = self._optimize_resource_allocation(task_requirements, resources, constraints)
            
            # Calculate utilization metrics
            utilization = self._calculate_resource_utilization(allocation_plan, resources)
            
            # Identify potential conflicts
            conflicts = self._identify_resource_conflicts(allocation_plan)
            
            result = {
                "allocation_summary": {
                    "total_tasks": len(tasks),
                    "total_resources": len(resources),
                    "allocation_efficiency": utilization['overall_efficiency']
                },
                "resource_allocation": allocation_plan,
                "utilization_metrics": utilization,
                "potential_conflicts": conflicts,
                "recommendations": self._generate_allocation_recommendations(allocation_plan, conflicts)
            }
            
            return json.dumps(result, indent=2, default=str)
            
        except Exception as e:
            return f"Error allocating resources: {str(e)}"
    
    def _estimate_timeline(self, timeline_request: str) -> str:
        """
        Estimate realistic timelines for tasks and projects
        """
        try:
            request = self._parse_timeline_request(timeline_request)
            tasks = request.get('tasks', [])
            working_hours_per_day = request.get('working_hours_per_day', 8)
            working_days_per_week = request.get('working_days_per_week', 5)
            buffer_percentage = request.get('buffer_percentage', 20)
            
            # Calculate base timeline
            base_timeline = self._calculate_base_timeline(tasks, working_hours_per_day)
            
            # Apply buffer for uncertainties
            buffered_timeline = self._apply_timeline_buffer(base_timeline, buffer_percentage)
            
            # Consider dependencies and critical path
            critical_path = self._calculate_critical_path(tasks)
            
            # Generate timeline breakdown
            timeline_breakdown = self._create_timeline_breakdown(buffered_timeline, critical_path)
            
            # Identify potential bottlenecks
            bottlenecks = self._identify_bottlenecks(timeline_breakdown)
            
            result = {
                "timeline_summary": {
                    "total_estimated_hours": sum(task.get('estimated_hours', 0) for task in tasks),
                    "estimated_duration_days": timeline_breakdown['total_duration_days'],
                    "working_days": timeline_breakdown['working_days'],
                    "calendar_days": timeline_breakdown['calendar_days'],
                    "buffer_applied": f"{buffer_percentage}%"
                },
                "critical_path": critical_path,
                "timeline_breakdown": timeline_breakdown,
                "bottlenecks": bottlenecks,
                "milestones": self._generate_timeline_milestones(timeline_breakdown),
                "risk_factors": self._identify_timeline_risks(tasks, timeline_breakdown)
            }
            
            return json.dumps(result, indent=2, default=str)
            
        except Exception as e:
            return f"Error estimating timeline: {str(e)}"
    
    def _analyze_priorities(self, priority_request: str) -> str:
        """
        Analyze and prioritize tasks based on various criteria
        """
        try:
            request = self._parse_priority_request(priority_request)
            tasks = request.get('tasks', [])
            criteria = request.get('criteria', ['business_value', 'urgency', 'effort', 'risk'])
            weights = request.get('weights', {})
            
            # Default weights if not provided
            default_weights = {
                'business_value': 0.3,
                'urgency': 0.25,
                'effort': 0.2,
                'risk': 0.15,
                'dependencies': 0.1
            }
            weights = {**default_weights, **weights}
            
            # Score tasks based on criteria
            scored_tasks = []
            for task in tasks:
                scores = self._calculate_priority_scores(task, criteria)
                weighted_score = self._calculate_weighted_score(scores, weights)
                
                scored_tasks.append({
                    **task,
                    'priority_scores': scores,
                    'weighted_score': weighted_score,
                    'priority_rank': 0  # Will be set after sorting
                })
            
            # Sort by priority score
            scored_tasks.sort(key=lambda x: x['weighted_score'], reverse=True)
            
            # Assign priority ranks
            for i, task in enumerate(scored_tasks):
                task['priority_rank'] = i + 1
                task['priority_level'] = self._get_priority_level(i + 1, len(scored_tasks))
            
            # Generate priority matrix
            priority_matrix = self._create_priority_matrix(scored_tasks)
            
            result = {
                "prioritization_summary": {
                    "total_tasks": len(tasks),
                    "criteria_used": criteria,
                    "weights_applied": weights
                },
                "prioritized_tasks": scored_tasks,
                "priority_matrix": priority_matrix,
                "quick_wins": [task for task in scored_tasks if task.get('effort', 'high') == 'low' and task['weighted_score'] > 0.7],
                "high_impact_tasks": [task for task in scored_tasks[:5]],
                "recommendations": self._generate_priority_recommendations(scored_tasks)
            }
            
            return json.dumps(result, indent=2, default=str)
            
        except Exception as e:
            return f"Error analyzing priorities: {str(e)}"
    
    def _assess_risks(self, risk_request: str) -> str:
        """
        Assess risks and create mitigation strategies
        """
        try:
            request = self._parse_risk_request(risk_request)
            project_data = request.get('project', {})
            tasks = request.get('tasks', [])
            context = request.get('context', {})
            
            # Identify potential risks
            identified_risks = self._identify_project_risks(project_data, tasks, context)
            
            # Assess risk impact and probability
            assessed_risks = []
            for risk in identified_risks:
                assessment = self._assess_risk_impact_probability(risk)
                mitigation_strategies = self._generate_mitigation_strategies(risk)
                
                assessed_risks.append({
                    **risk,
                    'impact_score': assessment['impact'],
                    'probability_score': assessment['probability'],
                    'risk_score': assessment['impact'] * assessment['probability'],
                    'risk_level': assessment['level'],
                    'mitigation_strategies': mitigation_strategies,
                    'contingency_plans': self._create_contingency_plans(risk)
                })
            
            # Sort by risk score
            assessed_risks.sort(key=lambda x: x['risk_score'], reverse=True)
            
            # Create risk register
            risk_register = self._create_risk_register(assessed_risks)
            
            # Generate monitoring plan
            monitoring_plan = self._create_risk_monitoring_plan(assessed_risks)
            
            result = {
                "risk_assessment_summary": {
                    "total_risks_identified": len(assessed_risks),
                    "high_risk_count": len([r for r in assessed_risks if r['risk_level'] == 'High']),
                    "medium_risk_count": len([r for r in assessed_risks if r['risk_level'] == 'Medium']),
                    "low_risk_count": len([r for r in assessed_risks if r['risk_level'] == 'Low'])
                },
                "risk_register": risk_register,
                "top_risks": assessed_risks[:5],
                "mitigation_summary": self._summarize_mitigations(assessed_risks),
                "monitoring_plan": monitoring_plan,
                "risk_response_strategies": self._generate_response_strategies()
            }
            
            return json.dumps(result, indent=2, default=str)
            
        except Exception as e:
            return f"Error assessing risks: {str(e)}"
    
    def _map_dependencies(self, dependency_request: str) -> str:
        """
        Map task dependencies and identify critical paths
        """
        try:
            request = self._parse_dependency_request(dependency_request)
            tasks = request.get('tasks', [])
            
            # Create dependency graph
            dependency_graph = self._create_dependency_graph(tasks)
            
            # Identify critical path
            critical_path = self._find_critical_path(dependency_graph)
            
            # Calculate early/late start and finish times
            schedule = self._calculate_project_schedule(dependency_graph)
            
            # Identify dependency conflicts
            conflicts = self._find_dependency_conflicts(dependency_graph)
            
            # Calculate float/slack for non-critical tasks
            float_analysis = self._calculate_task_float(schedule)
            
            result = {
                "dependency_summary": {
                    "total_tasks": len(tasks),
                    "total_dependencies": sum(len(task.get('dependencies', [])) for task in tasks),
                    "critical_path_tasks": len(critical_path),
                    "project_duration": schedule['project_duration']
                },
                "dependency_graph": dependency_graph,
                "critical_path": critical_path,
                "project_schedule": schedule,
                "dependency_conflicts": conflicts,
                "float_analysis": float_analysis,
                "dependency_recommendations": self._generate_dependency_recommendations(dependency_graph, conflicts)
            }
            
            return json.dumps(result, indent=2, default=str)
            
        except Exception as e:
            return f"Error mapping dependencies: {str(e)}"
    
    def _optimize_schedule(self, schedule_request: str) -> str:
        """
        Optimize project schedules for efficiency
        """
        try:
            request = self._parse_schedule_request(schedule_request)
            tasks = request.get('tasks', [])
            resources = request.get('resources', [])
            constraints = request.get('constraints', {})
            optimization_goals = request.get('goals', ['minimize_duration', 'balance_resources'])
            
            # Create initial schedule
            initial_schedule = self._create_initial_schedule(tasks, resources)
            
            # Apply optimization techniques
            optimized_schedule = self._apply_schedule_optimization(
                initial_schedule, 
                resources, 
                constraints, 
                optimization_goals
            )
            
            # Calculate improvements
            improvements = self._calculate_schedule_improvements(initial_schedule, optimized_schedule)
            
            # Validate feasibility
            feasibility_check = self._validate_schedule_feasibility(optimized_schedule, resources, constraints)
            
            result = {
                "optimization_summary": {
                    "optimization_goals": optimization_goals,
                    "duration_improvement": improvements['duration_improvement'],
                    "resource_utilization_improvement": improvements['resource_improvement'],
                    "feasibility": feasibility_check['is_feasible']
                },
                "initial_schedule": initial_schedule,
                "optimized_schedule": optimized_schedule,
                "improvements": improvements,
                "feasibility_analysis": feasibility_check,
                "implementation_recommendations": self._generate_schedule_recommendations(optimized_schedule)
            }
            
            return json.dumps(result, indent=2, default=str)
            
        except Exception as e:
            return f"Error optimizing schedule: {str(e)}"
    
    def _plan_milestones(self, milestone_request: str) -> str:
        """
        Create project milestones and checkpoints
        """
        try:
            request = self._parse_milestone_request(milestone_request)
            project_data = request.get('project', {})
            tasks = request.get('tasks', [])
            timeline = request.get('timeline', {})
            milestone_frequency = request.get('frequency', 'weekly')
            
            # Identify natural milestone points
            natural_milestones = self._identify_natural_milestones(tasks)
            
            # Create time-based milestones
            time_milestones = self._create_time_based_milestones(timeline, milestone_frequency)
            
            # Create deliverable-based milestones
            deliverable_milestones = self._create_deliverable_milestones(tasks)
            
            # Combine and prioritize milestones
            all_milestones = self._combine_milestones(natural_milestones, time_milestones, deliverable_milestones)
            
            # Create milestone schedule
            milestone_schedule = self._create_milestone_schedule(all_milestones, timeline)
            
            # Define success criteria for each milestone
            milestone_criteria = self._define_milestone_criteria(milestone_schedule)
            
            result = {
                "milestone_summary": {
                    "total_milestones": len(milestone_schedule),
                    "milestone_frequency": milestone_frequency,
                    "project_duration": timeline.get('duration_days', 'TBD')
                },
                "milestone_schedule": milestone_schedule,
                "milestone_criteria": milestone_criteria,
                "milestone_types": {
                    "natural": len(natural_milestones),
                    "time_based": len(time_milestones),
                    "deliverable_based": len(deliverable_milestones)
                },
                "tracking_recommendations": self._generate_milestone_tracking_recommendations()
            }
            
            return json.dumps(result, indent=2, default=str)
            
        except Exception as e:
            return f"Error planning milestones: {str(e)}"
    
    def _plan_capacity(self, capacity_request: str) -> str:
        """
        Plan team capacity and workload distribution
        """
        try:
            request = self._parse_capacity_request(capacity_request)
            team_members = request.get('team_members', [])
            tasks = request.get('tasks', [])
            time_period = request.get('time_period', 'month')
            
            # Analyze current capacity
            current_capacity = self._analyze_current_capacity(team_members)
            
            # Calculate demand from tasks
            demand_analysis = self._calculate_capacity_demand(tasks, time_period)
            
            # Perform capacity vs demand analysis
            capacity_gap_analysis = self._analyze_capacity_gaps(current_capacity, demand_analysis)
            
            # Optimize workload distribution
            optimized_distribution = self._optimize_workload_distribution(team_members, tasks)
            
            # Identify capacity constraints and bottlenecks
            constraints = self._identify_capacity_constraints(current_capacity, demand_analysis)
            
            result = {
                "capacity_summary": {
                    "total_team_members": len(team_members),
                    "total_available_hours": current_capacity['total_hours'],
                    "total_demand_hours": demand_analysis['total_hours'],
                    "capacity_utilization": demand_analysis['total_hours'] / current_capacity['total_hours'] * 100 if current_capacity['total_hours'] > 0 else 0
                },
                "current_capacity": current_capacity,
                "demand_analysis": demand_analysis,
                "capacity_gaps": capacity_gap_analysis,
                "optimized_distribution": optimized_distribution,
                "capacity_constraints": constraints,
                "scaling_recommendations": self._generate_capacity_recommendations(capacity_gap_analysis)
            }
            
            return json.dumps(result, indent=2, default=str)
            
        except Exception as e:
            return f"Error planning capacity: {str(e)}"
    
    def _estimate_budget(self, budget_request: str) -> str:
        """
        Estimate project budgets and costs
        """
        try:
            request = self._parse_budget_request(budget_request)
            tasks = request.get('tasks', [])
            resources = request.get('resources', [])
            project_data = request.get('project', {})
            cost_model = request.get('cost_model', 'resource_based')
            
            # Calculate resource costs
            resource_costs = self._calculate_resource_costs(tasks, resources)
            
            # Calculate operational costs
            operational_costs = self._calculate_operational_costs(project_data)
            
            # Calculate overhead costs
            overhead_costs = self._calculate_overhead_costs(resource_costs, project_data)
            
            # Add contingency and risk buffers
            risk_costs = self._calculate_risk_costs(tasks, project_data)
            
            # Create detailed budget breakdown
            budget_breakdown = self._create_budget_breakdown(
                resource_costs, operational_costs, overhead_costs, risk_costs
            )
            
            # Perform cost optimization analysis
            optimization_opportunities = self._identify_cost_optimization_opportunities(budget_breakdown)
            
            result = {
                "budget_summary": {
                    "total_budget": budget_breakdown['total'],
                    "resource_costs": budget_breakdown['resource_costs'],
                    "operational_costs": budget_breakdown['operational_costs'],
                    "overhead_costs": budget_breakdown['overhead_costs'],
                    "contingency": budget_breakdown['contingency']
                },
                "detailed_breakdown": budget_breakdown,
                "cost_by_category": self._categorize_costs(budget_breakdown),
                "cost_by_timeframe": self._calculate_costs_by_timeframe(budget_breakdown, tasks),
                "optimization_opportunities": optimization_opportunities,
                "budget_recommendations": self._generate_budget_recommendations(budget_breakdown, optimization_opportunities)
            }
            
            return json.dumps(result, indent=2, default=str)
            
        except Exception as e:
            return f"Error estimating budget: {str(e)}"
    
    # Helper methods for initialization
    def _initialize_templates(self) -> Dict[str, Any]:
        """Initialize planning templates"""
        return {
            "project_types": {
                "software_development": {
                    "phases": ["planning", "analysis", "design", "development", "testing", "deployment"],
                    "typical_buffer": 25,
                    "common_risks": ["scope_creep", "technical_debt", "resource_availability"]
                },
                "marketing_campaign": {
                    "phases": ["research", "planning", "creative", "production", "launch", "optimization"],
                    "typical_buffer": 20,
                    "common_risks": ["market_changes", "creative_approval", "budget_constraints"]
                },
                "data_analysis": {
                    "phases": ["data_collection", "cleaning", "analysis", "visualization", "reporting"],
                    "typical_buffer": 30,
                    "common_risks": ["data_quality", "scope_expansion", "tool_limitations"]
                }
            }
        }
    
    # Task breakdown helper methods
    def _parse_task_request(self, request: str) -> Dict[str, Any]:
        """Parse task breakdown request"""
        try:
            return json.loads(request)
        except:
            return {"task": request}
    
    def _analyze_and_breakdown_task(self, task: str, project_type: str, complexity: str) -> List[Dict[str, Any]]:
        """Analyze and break down a complex task"""
        # This would use more sophisticated analysis in a real implementation
        if project_type == "software_development":
            return self._breakdown_software_task(task, complexity)
        elif project_type == "marketing_campaign":
            return self._breakdown_marketing_task(task, complexity)
        elif project_type == "data_analysis":
            return self._breakdown_data_task(task, complexity)
        else:
            return self._breakdown_general_task(task, complexity)
    
    def _breakdown_software_task(self, task: str, complexity: str) -> List[Dict[str, Any]]:
        """Break down software development task"""
        base_tasks = [
            {"name": "Requirements Analysis", "type": "analysis"},
            {"name": "Technical Design", "type": "design"},
            {"name": "Implementation", "type": "development"},
            {"name": "Testing", "type": "testing"},
            {"name": "Documentation", "type": "documentation"},
            {"name": "Code Review", "type": "review"}
        ]
        
        if complexity == "high":
            base_tasks.extend([
                {"name": "Architecture Planning", "type": "planning"},
                {"name": "Performance Testing", "type": "testing"},
                {"name": "Security Review", "type": "review"}
            ])
        
        return base_tasks
    
    def _breakdown_marketing_task(self, task: str, complexity: str) -> List[Dict[str, Any]]:
        """Break down marketing task"""
        return [
            {"name": "Market Research", "type": "research"},
            {"name": "Target Audience Analysis", "type": "analysis"},
            {"name": "Campaign Strategy", "type": "planning"},
            {"name": "Content Creation", "type": "creation"},
            {"name": "Campaign Launch", "type": "execution"},
            {"name": "Performance Monitoring", "type": "monitoring"}
        ]
    
    def _breakdown_data_task(self, task: str, complexity: str) -> List[Dict[str, Any]]:
        """Break down data analysis task"""
        return [
            {"name": "Data Collection", "type": "collection"},
            {"name": "Data Cleaning", "type": "preprocessing"},
            {"name": "Exploratory Analysis", "type": "analysis"},
            {"name": "Statistical Analysis", "type": "analysis"},
            {"name": "Visualization", "type": "visualization"},
            {"name": "Report Generation", "type": "reporting"}
        ]
    
    def _breakdown_general_task(self, task: str, complexity: str) -> List[Dict[str, Any]]:
        """Break down general task"""
        return [
            {"name": "Planning", "type": "planning"},
            {"name": "Research", "type": "research"},
            {"name": "Execution", "type": "execution"},
            {"name": "Review", "type": "review"},
            {"name": "Documentation", "type": "documentation"}
        ]
    
    def _create_task_hierarchy(self, breakdown: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Create hierarchical task structure"""
        hierarchy = []
        for i, task in enumerate(breakdown):
            hierarchy.append({
                "id": str(uuid.uuid4()),
                "name": task["name"],
                "type": task["type"],
                "level": 1,
                "parent_id": None,
                "order": i + 1
            })
        return hierarchy
    
    def _estimate_subtask_effort(self, tasks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Estimate effort for subtasks"""
        effort_multipliers = {
            "planning": 1.0,
            "research": 1.5,
            "analysis": 2.0,
            "design": 2.5,
            "development": 3.0,
            "testing": 2.0,
            "documentation": 1.0,
            "review": 0.5
        }
        
        for task in tasks:
            base_effort = 8  # 8 hours base
            task_type = task.get("type", "general")
            multiplier = effort_multipliers.get(task_type, 1.0)
            task["estimated_hours"] = base_effort * multiplier
        
        return tasks
    
    def _add_task_dependencies(self, tasks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Add logical dependencies between tasks"""
        # Simple sequential dependencies for now
        for i, task in enumerate(tasks):
            if i > 0:
                task["dependencies"] = [tasks[i-1]["id"]]
            else:
                task["dependencies"] = []
        
        return tasks
    
    def _create_wbs(self, tasks: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Create Work Breakdown Structure"""
        return {
            "structure": "hierarchical",
            "total_tasks": len(tasks),
            "levels": 2,  # Simplified for this example
            "breakdown_criteria": "functional"
        }
    
    def _recommend_approach(self, project_type: str, complexity: str) -> List[str]:
        """Recommend project approach"""
        recommendations = []
        
        if complexity == "high":
            recommendations.extend([
                "Use iterative approach with regular checkpoints",
                "Implement risk mitigation strategies early",
                "Plan for more detailed requirements analysis"
            ])
        
        if project_type == "software_development":
            recommendations.append("Consider agile methodology")
        
        return recommendations
    
    # Resource allocation helper methods
    def _parse_allocation_request(self, request: str) -> Dict[str, Any]:
        """Parse resource allocation request"""
        try:
            return json.loads(request)
        except:
            return {"tasks": [], "resources": []}
    
    def _parse_resource(self, resource_data: Dict[str, Any]) -> Resource:
        """Parse resource data into Resource object"""
        return Resource(
            id=resource_data.get('id', str(uuid.uuid4())),
            name=resource_data.get('name', 'Unknown'),
            type=resource_data.get('type', 'general'),
            availability=resource_data.get('availability', 1.0),
            cost_per_hour=resource_data.get('cost_per_hour', 50.0),
            skills=resource_data.get('skills', [])
        )
    
    def _analyze_resource_requirements(self, tasks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Analyze resource requirements for tasks"""
        requirements = []
        for task in tasks:
            requirements.append({
                "task_id": task.get("id", str(uuid.uuid4())),
                "task_name": task.get("name", "Unknown"),
                "required_skills": task.get("required_skills", []),
                "effort_hours": task.get("estimated_hours", 8),
                "priority": task.get("priority", "medium")
            })
        return requirements
    
    def _optimize_resource_allocation(self, requirements: List[Dict[str, Any]], resources: List[Resource], constraints: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Optimize resource allocation using simple heuristics"""
        allocation = []
        
        for req in requirements:
            best_resource = None
            best_score = -1
            
            for resource in resources:
                score = self._calculate_resource_match_score(req, resource)
                if score > best_score:
                    best_score = score
                    best_resource = resource
            
            if best_resource:
                allocation.append({
                    "task_id": req["task_id"],
                    "task_name": req["task_name"],
                    "assigned_resource": best_resource.name,
                    "resource_id": best_resource.id,
                    "match_score": best_score,
                    "estimated_cost": req["effort_hours"] * best_resource.cost_per_hour
                })
        
        return allocation
    
    def _calculate_resource_match_score(self, requirement: Dict[str, Any], resource: Resource) -> float:
        """Calculate how well a resource matches a requirement"""
        score = 0.5  # Base score
        
        # Skill matching
        required_skills = set(requirement.get("required_skills", []))
        resource_skills = set(resource.skills)
        if required_skills:
            skill_match = len(required_skills.intersection(resource_skills)) / len(required_skills)
            score += skill_match * 0.4
        
        # Availability
        score += resource.availability * 0.1
        
        return min(score, 1.0)
    
    def _calculate_resource_utilization(self, allocation: List[Dict[str, Any]], resources: List[Resource]) -> Dict[str, Any]:
        """Calculate resource utilization metrics"""
        total_hours = sum(alloc.get("estimated_cost", 0) / 50 for alloc in allocation)  # Assuming $50/hour average
        total_capacity = sum(r.availability * 40 for r in resources)  # 40 hours per week
        
        return {
            "total_allocated_hours": total_hours,
            "total_capacity_hours": total_capacity,
            "overall_efficiency": (total_hours / total_capacity * 100) if total_capacity > 0 else 0,
            "resource_utilization": [
                {
                    "resource_name": r.name,
                    "utilization_percentage": min(100, (r.availability * 40) / 40 * 100)
                }
                for r in resources
            ]
        }
    
    def _identify_resource_conflicts(self, allocation: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Identify potential resource conflicts"""
        conflicts = []
        resource_assignments = {}
        
        for alloc in allocation:
            resource_id = alloc.get("resource_id")
            if resource_id in resource_assignments:
                conflicts.append({
                    "type": "over_allocation",
                    "resource_id": resource_id,
                    "conflicting_tasks": [resource_assignments[resource_id], alloc["task_name"]],
                    "severity": "medium"
                })
            else:
                resource_assignments[resource_id] = alloc["task_name"]
        
        return conflicts
    
    def _generate_allocation_recommendations(self, allocation: List[Dict[str, Any]], conflicts: List[Dict[str, Any]]) -> List[str]:
        """Generate resource allocation recommendations"""
        recommendations = []
        
        if conflicts:
            recommendations.append("Address resource conflicts before project start")
        
        low_match_tasks = [a for a in allocation if a.get("match_score", 0) < 0.6]
        if low_match_tasks:
            recommendations.append("Consider additional training or hiring for low-match assignments")
        
        recommendations.append("Regularly review and adjust resource allocation as project progresses")
        
        return recommendations
    
    # Additional helper methods would continue here for timeline estimation, 
    # priority analysis, risk assessment, etc. Due to length constraints,
    # I'll provide the key framework and a few more critical methods
    
    def _parse_timeline_request(self, request: str) -> Dict[str, Any]:
        """Parse timeline estimation request"""
        try:
            return json.loads(request)
        except:
            return {"tasks": []}
    
    def _calculate_base_timeline(self, tasks: List[Dict[str, Any]], working_hours_per_day: int) -> Dict[str, Any]:
        """Calculate base timeline without buffers"""
        total_hours = sum(task.get('estimated_hours', 0) for task in tasks)
        total_days = math.ceil(total_hours / working_hours_per_day)
        
        return {
            "total_hours": total_hours,
            "total_days": total_days,
            "working_hours_per_day": working_hours_per_day
        }
    
    def _apply_timeline_buffer(self, base_timeline: Dict[str, Any], buffer_percentage: int) -> Dict[str, Any]:
        """Apply buffer to timeline for uncertainties"""
        buffered_hours = base_timeline["total_hours"] * (1 + buffer_percentage / 100)
        buffered_days = math.ceil(buffered_hours / base_timeline["working_hours_per_day"])
        
        return {
            **base_timeline,
            "buffered_hours": buffered_hours,
            "buffered_days": buffered_days,
            "buffer_percentage": buffer_percentage
        }
    
    def _calculate_critical_path(self, tasks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Calculate critical path through tasks"""
        # Simplified critical path calculation
        # In a real implementation, this would use more sophisticated algorithms
        sorted_tasks = sorted(tasks, key=lambda x: len(x.get('dependencies', [])))
        
        critical_path = []
        total_duration = 0
        
        for task in sorted_tasks:
            if not task.get('dependencies') or all(
                dep in [cp['id'] for cp in critical_path] 
                for dep in task.get('dependencies', [])
            ):
                critical_path.append({
                    "id": task.get('id', str(uuid.uuid4())),
                    "name": task.get('name', 'Unknown'),
                    "duration": task.get('estimated_hours', 0),
                    "start_time": total_duration,
                    "end_time": total_duration + task.get('estimated_hours', 0)
                })
                total_duration += task.get('estimated_hours', 0)
        
        return critical_path
    
    # Continue with other helper methods as needed...
    # The class would continue with implementations for all the remaining methods
    # referenced in the tool functions above.
    
    def _create_timeline_breakdown(self, timeline: Dict[str, Any], critical_path: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Create detailed timeline breakdown"""
        return {
            "total_duration_days": timeline.get("buffered_days", 0),
            "working_days": timeline.get("buffered_days", 0),
            "calendar_days": math.ceil(timeline.get("buffered_days", 0) * 7 / 5),  # Assuming 5 working days per week
            "critical_path_duration": sum(task["duration"] for task in critical_path),
            "phases": ["Phase 1", "Phase 2", "Phase 3"]  # Simplified
        }
    
    def _identify_bottlenecks(self, timeline_breakdown: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Identify potential timeline bottlenecks"""
        return [
            {
                "type": "resource_constraint",
                "description": "Limited availability of specialized resources",
                "impact": "medium",
                "mitigation": "Cross-train team members or hire additional resources"
            }
        ]
    
    def _generate_timeline_milestones(self, timeline_breakdown: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Generate timeline milestones"""
        total_days = timeline_breakdown.get("total_duration_days", 30)
        milestone_interval = max(5, total_days // 4)  # Create 4 milestones minimum
        
        milestones = []
        for i in range(1, 5):
            milestones.append({
                "milestone": f"Milestone {i}",
                "day": min(i * milestone_interval, total_days),
                "deliverable": f"Phase {i} completion",
                "success_criteria": f"All Phase {i} tasks completed and reviewed"
            })
        
        return milestones
    
    def _identify_timeline_risks(self, tasks: List[Dict[str, Any]], timeline_breakdown: Dict[str, Any]) -> List[str]:
        """Identify timeline-related risks"""
        return [
            "Scope creep may extend timeline",
            "Resource availability may impact schedule",
            "External dependencies could cause delays",
            "Quality issues may require rework"
        ]
    
    # Implement remaining helper methods with similar patterns...
    # This provides a comprehensive framework for the Planning Agent