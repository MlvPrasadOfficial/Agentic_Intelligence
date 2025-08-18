'use client'

import { useState, useEffect } from 'react'
import { motion } from 'framer-motion'
import { ArrowRightIcon, SparklesIcon, CpuChipIcon, ChartBarIcon, DocumentTextIcon } from '@heroicons/react/24/outline'
import { HeroSection } from '@/components/hero-section'
import { AgentStatusGrid } from '@/components/agent-status-grid'
import { WorkflowBuilder } from '@/components/workflow-builder'
import { Dashboard } from '@/components/dashboard'
import { ChatInterface } from '@/components/chat-interface'
import { Navigation } from '@/components/navigation'
import { useWebSocket } from '@/hooks/use-websocket'
import { useAgentStore } from '@/store/agent-store'
import { useWorkflowStore } from '@/store/workflow-store'

type ViewType = 'home' | 'dashboard' | 'workflows' | 'agents' | 'analytics'

const containerVariants = {
  hidden: { opacity: 0 },
  visible: {
    opacity: 1,
    transition: {
      staggerChildren: 0.1
    }
  }
}

const itemVariants = {
  hidden: { y: 20, opacity: 0 },
  visible: {
    y: 0,
    opacity: 1,
    transition: {
      duration: 0.5
    }
  }
}

export default function HomePage() {
  const [currentView, setCurrentView] = useState<ViewType>('home')
  const [isLoaded, setIsLoaded] = useState(false)
  
  // WebSocket connection for real-time updates
  const { isConnected, connectionStatus } = useWebSocket()
  
  // Store state
  const agents = useAgentStore(state => state.agents)
  const workflows = useWorkflowStore(state => state.workflows)

  useEffect(() => {
    // Simulate loading delay for smooth animations
    const timer = setTimeout(() => setIsLoaded(true), 500)
    return () => clearTimeout(timer)
  }, [])

  const renderView = () => {
    switch (currentView) {
      case 'home':
        return (
          <motion.div
            key="home"
            variants={containerVariants}
            initial="hidden"
            animate="visible"
            className="space-y-12"
          >
            {/* Hero Section */}
            <motion.div variants={itemVariants}>
              <HeroSection onGetStarted={() => setCurrentView('workflows')} />
            </motion.div>

            {/* Agent Status Overview */}
            <motion.div variants={itemVariants} className="container mx-auto px-6">
              <div className="text-center mb-8">
                <h2 className="text-3xl font-bold text-white mb-4">
                  Agent Status Overview
                </h2>
                <p className="text-white/70 text-lg">
                  Monitor your AI agents in real-time
                </p>
              </div>
              <AgentStatusGrid />
            </motion.div>

            {/* Quick Actions */}
            <motion.div variants={itemVariants} className="container mx-auto px-6">
              <div className="grid md:grid-cols-2 lg:grid-cols-4 gap-6">
                {quickActions.map((action, index) => (
                  <motion.div
                    key={action.title}
                    variants={itemVariants}
                    whileHover={{ scale: 1.05 }}
                    whileTap={{ scale: 0.95 }}
                    className="glass-card text-center cursor-pointer group"
                    onClick={() => setCurrentView(action.view as ViewType)}
                  >
                    <div className="inline-flex items-center justify-center w-12 h-12 rounded-lg bg-blue-500/20 text-blue-400 mb-4 group-hover:bg-blue-500/30 transition-all">
                      <action.icon className="w-6 h-6" />
                    </div>
                    <h3 className="font-semibold text-white mb-2">{action.title}</h3>
                    <p className="text-white/70 text-sm">{action.description}</p>
                  </motion.div>
                ))}
              </div>
            </motion.div>
          </motion.div>
        )

      case 'dashboard':
        return (
          <motion.div
            key="dashboard"
            initial={{ opacity: 0, x: 100 }}
            animate={{ opacity: 1, x: 0 }}
            exit={{ opacity: 0, x: -100 }}
            transition={{ duration: 0.3 }}
          >
            <Dashboard />
          </motion.div>
        )

      case 'workflows':
        return (
          <motion.div
            key="workflows"
            initial={{ opacity: 0, x: 100 }}
            animate={{ opacity: 1, x: 0 }}
            exit={{ opacity: 0, x: -100 }}
            transition={{ duration: 0.3 }}
          >
            <WorkflowBuilder />
          </motion.div>
        )

      case 'agents':
        return (
          <motion.div
            key="agents"
            initial={{ opacity: 0, x: 100 }}
            animate={{ opacity: 1, x: 0 }}
            exit={{ opacity: 0, x: -100 }}
            transition={{ duration: 0.3 }}
            className="container mx-auto px-6 py-8"
          >
            <div className="mb-8">
              <h1 className="text-3xl font-bold text-white mb-4">Agent Management</h1>
              <p className="text-white/70">
                Manage and monitor your AI agents
              </p>
            </div>
            <AgentStatusGrid detailed />
          </motion.div>
        )

      case 'analytics':
        return (
          <motion.div
            key="analytics"
            initial={{ opacity: 0, x: 100 }}
            animate={{ opacity: 1, x: 0 }}
            exit={{ opacity: 0, x: -100 }}
            transition={{ duration: 0.3 }}
            className="container mx-auto px-6 py-8"
          >
            <div className="mb-8">
              <h1 className="text-3xl font-bold text-white mb-4">Analytics</h1>
              <p className="text-white/70">
                Insights and performance metrics
              </p>
            </div>
            {/* Analytics components would go here */}
            <div className="glass-card text-center py-12">
              <ChartBarIcon className="w-16 h-16 mx-auto text-white/50 mb-4" />
              <h3 className="text-xl font-semibold text-white mb-2">
                Analytics Dashboard
              </h3>
              <p className="text-white/70">
                Detailed analytics coming soon...
              </p>
            </div>
          </motion.div>
        )

      default:
        return null
    }
  }

  if (!isLoaded) {
    return (
      <div className="min-h-screen flex items-center justify-center">
        <div className="glass-card text-center">
          <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-blue-400 mx-auto mb-4"></div>
          <p className="text-white/70">Loading IntelliFlow...</p>
        </div>
      </div>
    )
  }

  return (
    <div className="min-h-screen">
      {/* Navigation */}
      <Navigation 
        currentView={currentView} 
        onViewChange={setCurrentView}
        connectionStatus={connectionStatus}
      />

      {/* Main Content */}
      <main className="pt-16">
        {renderView()}
      </main>

      {/* Floating Chat Interface */}
      <ChatInterface />

      {/* Connection Status Indicator */}
      {!isConnected && (
        <motion.div
          initial={{ opacity: 0, y: 100 }}
          animate={{ opacity: 1, y: 0 }}
          className="fixed bottom-4 left-4 glass-card px-4 py-2 text-sm"
        >
          <div className="flex items-center gap-2">
            <div className="w-2 h-2 rounded-full bg-red-400 animate-pulse"></div>
            <span className="text-white/70">Connecting to server...</span>
          </div>
        </motion.div>
      )}
    </div>
  )
}

const quickActions = [
  {
    title: 'Create Workflow',
    description: 'Build custom AI workflows',
    icon: SparklesIcon,
    view: 'workflows'
  },
  {
    title: 'View Dashboard',
    description: 'Monitor system metrics',
    icon: ChartBarIcon,
    view: 'dashboard'
  },
  {
    title: 'Manage Agents',
    description: 'Configure AI agents',
    icon: CpuChipIcon,
    view: 'agents'
  },
  {
    title: 'Analytics',
    description: 'View performance data',
    icon: DocumentTextIcon,
    view: 'analytics'
  }
]