// Core API Types
export interface ApiResponse<T = any> {
  data?: T
  error?: string
  message?: string
  timestamp: string
}

// Agent Types
export type AgentType = 'research' | 'code' | 'data' | 'communication' | 'planning'

export type AgentStatus = 'idle' | 'working' | 'complete' | 'error'

export interface Agent {
  agent_id: string
  agent_name: string
  agent_type: AgentType
  status: AgentStatus
  description: string
  tools: string[]
  metrics: AgentMetrics
}

export interface AgentMetrics {
  total_executions: number
  successful_executions: number
  failed_executions: number
  success_rate: number
  avg_execution_time: number
  total_tokens: number
}

export interface AgentExecution {
  execution_id: string
  agent_type: AgentType
  agent_name: string
  status: 'completed' | 'failed' | 'running'
  result: any
  execution_time: number
  timestamp: string
}

// Workflow Types
export type WorkflowType = 'market_research' | 'code_documentation' | 'customer_support' | 'data_analysis' | 'content_creation'

export type WorkflowStatus = 'pending' | 'running' | 'completed' | 'failed' | 'cancelled'

export interface Workflow {
  workflow_id: string
  workflow_type: WorkflowType
  status: WorkflowStatus
  result?: any
  execution_time: number
  timestamp: string
  progress?: number
  current_step?: string
}

export interface WorkflowStep {
  id: string
  name: string
  agent_type: AgentType
  status: AgentStatus
  input: any
  output?: any
  error?: string
  execution_time?: number
}

export interface WorkflowTemplate {
  id: string
  name: string
  description: string
  workflow_type: WorkflowType
  steps: WorkflowStep[]
  estimated_time: number
  complexity: 'low' | 'medium' | 'high'
}

// WebSocket Types
export type WebSocketMessageType = 
  | 'agent_start' 
  | 'agent_thinking' 
  | 'agent_complete' 
  | 'agent_error'
  | 'workflow_start'
  | 'workflow_progress'
  | 'workflow_complete'
  | 'workflow_error'
  | 'tool_start'
  | 'tool_complete'
  | 'system_status'
  | 'system_metrics'
  | 'user_message'

export interface WebSocketMessage {
  type: WebSocketMessageType
  data: any
  timestamp: string
  client_id?: string
  session_id?: string
}

export interface AgentStatusUpdate {
  agent_id: string
  agent_name: string
  agent_type: AgentType
  status: AgentStatus
  message?: string
  progress?: number
}

export interface WorkflowStatusUpdate {
  workflow_id: string
  workflow_type: WorkflowType
  status: WorkflowStatus
  progress?: number
  current_step?: string
  message?: string
}

// System Types
export interface SystemStatus {
  status: 'operational' | 'degraded' | 'error'
  active_agents: number
  active_workflows: number
  active_connections: number
  system_metrics: SystemMetrics
  timestamp: string
}

export interface SystemMetrics {
  cpu_percent?: number
  memory_percent?: number
  disk_usage?: number
  uptime?: string
  [key: string]: any
}

// User Types
export interface User {
  user_id: string
  username: string
  email: string
  created_at: string
}

export interface AuthToken {
  access_token: string
  token_type: string
  expires_in: number
}

// UI Types
export interface ToastMessage {
  id: string
  title: string
  description?: string
  type: 'success' | 'error' | 'warning' | 'info'
  duration?: number
}

export type ViewType = 'home' | 'dashboard' | 'workflows' | 'agents' | 'analytics'

// Workflow Builder Types
export interface WorkflowNode {
  id: string
  type: 'agent' | 'condition' | 'input' | 'output'
  position: { x: number; y: number }
  data: {
    label: string
    agent_type?: AgentType
    parameters?: any
    [key: string]: any
  }
}

export interface WorkflowEdge {
  id: string
  source: string
  target: string
  type?: string
  data?: any
}

export interface WorkflowGraph {
  nodes: WorkflowNode[]
  edges: WorkflowEdge[]
}

// Chart/Analytics Types
export interface ChartDataPoint {
  timestamp: string
  value: number
  label?: string
}

export interface ChartConfig {
  type: 'line' | 'bar' | 'pie' | 'area'
  data: ChartDataPoint[]
  title: string
  yAxisLabel?: string
  xAxisLabel?: string
}

// Configuration Types
export interface AppConfig {
  api_url: string
  ws_url: string
  default_theme: 'light' | 'dark' | 'system'
  enable_animations: boolean
  debug_mode: boolean
}

// Request/Response Types
export interface CreateWorkflowRequest {
  workflow_type: WorkflowType
  input_data: any
  client_id?: string
}

export interface ExecuteAgentRequest {
  agent_type: AgentType
  task: string
  context?: any
  client_id?: string
}

// Error Types
export interface APIError {
  message: string
  code?: string
  details?: any
  timestamp: string
}

// Store Types
export interface AgentStore {
  agents: Record<string, Agent>
  activeExecutions: Record<string, AgentExecution>
  isLoading: boolean
  error: string | null
  
  // Actions
  updateAgent: (agentId: string, updates: Partial<Agent>) => void
  addExecution: (execution: AgentExecution) => void
  removeExecution: (executionId: string) => void
  setLoading: (loading: boolean) => void
  setError: (error: string | null) => void
  reset: () => void
}

export interface WorkflowStore {
  workflows: Record<string, Workflow>
  templates: WorkflowTemplate[]
  activeWorkflow: string | null
  isBuilding: boolean
  currentGraph: WorkflowGraph | null
  
  // Actions
  addWorkflow: (workflow: Workflow) => void
  updateWorkflow: (workflowId: string, updates: Partial<Workflow>) => void
  removeWorkflow: (workflowId: string) => void
  setActiveWorkflow: (workflowId: string | null) => void
  setBuilding: (building: boolean) => void
  setCurrentGraph: (graph: WorkflowGraph | null) => void
  reset: () => void
}

export interface UIStore {
  theme: 'light' | 'dark' | 'system'
  sidebarOpen: boolean
  toasts: ToastMessage[]
  isConnected: boolean
  connectionStatus: 'connected' | 'connecting' | 'disconnected' | 'error'
  
  // Actions
  setTheme: (theme: 'light' | 'dark' | 'system') => void
  toggleSidebar: () => void
  setSidebarOpen: (open: boolean) => void
  addToast: (toast: Omit<ToastMessage, 'id'>) => void
  removeToast: (toastId: string) => void
  setConnectionStatus: (status: 'connected' | 'connecting' | 'disconnected' | 'error') => void
  reset: () => void
}

// Utility Types
export type DeepPartial<T> = {
  [P in keyof T]?: T[P] extends object ? DeepPartial<T[P]> : T[P]
}

export type Optional<T, K extends keyof T> = Omit<T, K> & Partial<Pick<T, K>>

export type RequiredFields<T, K extends keyof T> = T & Required<Pick<T, K>>

// Component Props Types
export interface BaseComponentProps {
  className?: string
  children?: React.ReactNode
}

export interface GlassCardProps extends BaseComponentProps {
  variant?: 'default' | 'dark'
  hover?: boolean
  onClick?: () => void
}

export interface ButtonProps extends BaseComponentProps {
  variant?: 'default' | 'primary' | 'secondary' | 'ghost' | 'glass'
  size?: 'sm' | 'md' | 'lg'
  disabled?: boolean
  loading?: boolean
  onClick?: () => void
  type?: 'button' | 'submit' | 'reset'
}