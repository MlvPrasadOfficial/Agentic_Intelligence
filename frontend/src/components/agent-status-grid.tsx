'use client'

import React, { useState, useEffect } from 'react'
import { motion, AnimatePresence } from 'framer-motion'
import { 
  CpuChipIcon, 
  ChartBarIcon, 
  ClockIcon, 
  CheckCircleIcon,
  ExclamationTriangleIcon,
  PlayIcon
} from '@heroicons/react/24/outline'
import { GlassCard, InteractiveGlassCard } from './ui/glass-card'
import { cn, getAgentStatusColor, getAgentTypeIcon, getAgentTypeName, formatDuration, formatNumber } from '@/lib/utils'
import { Agent, AgentType, AgentStatus } from '@/types'
import { useAgentStore } from '@/store/agent-store'
import { mockData } from '@/lib/api'

interface AgentStatusGridProps {
  detailed?: boolean
}

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
  hidden: { y: 20, opacity: 0, scale: 0.9 },
  visible: {
    y: 0,
    opacity: 1,
    scale: 1,
    transition: {
      duration: 0.5,
      ease: [0.25, 0.46, 0.45, 0.94]
    }
  }
}

export function AgentStatusGrid({ detailed = false }: AgentStatusGridProps) {
  const [agents, setAgents] = useState<Agent[]>([])
  const [isLoading, setIsLoading] = useState(true)
  const [selectedAgent, setSelectedAgent] = useState<string | null>(null)

  // Simulate loading agents (in production, this would come from API/store)
  useEffect(() => {
    const loadAgents = async () => {
      setIsLoading(true)
      // Simulate API call
      await new Promise(resolve => setTimeout(resolve, 1000))
      setAgents(mockData.agents)
      setIsLoading(false)
    }

    loadAgents()

    // Simulate real-time updates
    const interval = setInterval(() => {
      setAgents(prev => prev.map(agent => ({
        ...agent,
        status: Math.random() > 0.8 ? getRandomStatus() : agent.status,
        metrics: {
          ...agent.metrics,
          total_executions: agent.metrics.total_executions + (Math.random() > 0.9 ? 1 : 0)
        }
      })))
    }, 5000)

    return () => clearInterval(interval)
  }, [])

  if (isLoading) {
    return (
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 xl:grid-cols-5 gap-6">
        {Array.from({ length: 5 }).map((_, index) => (
          <GlassCard key={index} className="animate-pulse">
            <div className="space-y-4">
              <div className="flex items-center justify-between">
                <div className="w-8 h-8 bg-white/20 rounded-lg"></div>
                <div className="w-16 h-4 bg-white/20 rounded"></div>
              </div>
              <div className="space-y-2">
                <div className="w-3/4 h-4 bg-white/20 rounded"></div>
                <div className="w-1/2 h-3 bg-white/10 rounded"></div>
              </div>
              <div className="space-y-2">
                <div className="w-full h-2 bg-white/10 rounded"></div>
                <div className="flex justify-between">
                  <div className="w-12 h-3 bg-white/10 rounded"></div>
                  <div className="w-12 h-3 bg-white/10 rounded"></div>
                </div>
              </div>
            </div>
          </GlassCard>
        ))}
      </div>
    )
  }

  return (
    <motion.div
      variants={containerVariants}
      initial="hidden"
      animate="visible"
      className={cn(
        "grid gap-6",
        detailed 
          ? "grid-cols-1 md:grid-cols-2 lg:grid-cols-3" 
          : "grid-cols-1 md:grid-cols-2 lg:grid-cols-3 xl:grid-cols-5"
      )}
    >
      <AnimatePresence>
        {agents.map((agent, index) => (
          <motion.div
            key={agent.agent_id}
            variants={itemVariants}
            layout
            whileHover={{ scale: 1.02, y: -2 }}
            className="group"
          >
            <InteractiveGlassCard
              onClick={() => setSelectedAgent(selectedAgent === agent.agent_id ? null : agent.agent_id)}
              className="relative h-full"
              size={detailed ? 'lg' : 'md'}
            >
              {/* Status Indicator */}
              <div className="absolute top-4 right-4">
                <motion.div
                  className={cn(
                    'w-3 h-3 rounded-full border-2',
                    getAgentStatusColor(agent.status),
                    agent.status === 'working' && 'animate-pulse'
                  )}
                  animate={agent.status === 'working' ? { scale: [1, 1.2, 1] } : {}}
                  transition={{ repeat: Infinity, duration: 2 }}
                />
              </div>

              {/* Agent Header */}
              <div className="flex items-start gap-3 mb-4">
                <motion.div
                  className="flex-shrink-0 w-10 h-10 rounded-xl bg-gradient-to-br from-blue-500/20 to-purple-500/20 border border-white/20 flex items-center justify-center text-xl"
                  whileHover={{ rotate: 5, scale: 1.1 }}
                  transition={{ duration: 0.2 }}
                >
                  {getAgentTypeIcon(agent.agent_type)}
                </motion.div>
                
                <div className="flex-1 min-w-0">
                  <h3 className="font-semibold text-white text-sm group-hover:text-blue-300 transition-colors truncate">
                    {agent.agent_name}
                  </h3>
                  <p className="text-xs text-white/60 mt-0.5 capitalize">
                    {agent.status}
                  </p>
                </div>
              </div>

              {/* Agent Description */}
              <p className={cn(
                "text-white/70 text-xs leading-relaxed mb-4",
                detailed ? "line-clamp-3" : "line-clamp-2"
              )}>
                {agent.description}
              </p>

              {/* Metrics */}
              <div className="space-y-3">
                {/* Execution Stats */}
                <div className="flex items-center justify-between text-xs">
                  <span className="text-white/60">Success Rate</span>
                  <span className="text-green-400 font-medium">
                    {(agent.metrics.success_rate * 100).toFixed(1)}%
                  </span>
                </div>

                {/* Progress Bar */}
                <div className="relative">
                  <div className="w-full h-1.5 bg-white/10 rounded-full overflow-hidden">
                    <motion.div
                      className="h-full bg-gradient-to-r from-green-500 to-blue-500 rounded-full"
                      initial={{ width: 0 }}
                      animate={{ width: `${agent.metrics.success_rate * 100}%` }}
                      transition={{ duration: 1, ease: 'easeOut' }}
                    />
                  </div>
                </div>

                {/* Additional Stats */}
                <div className="grid grid-cols-2 gap-2 text-xs">
                  <div className="text-center">
                    <div className="text-white font-medium">
                      {formatNumber(agent.metrics.total_executions)}
                    </div>
                    <div className="text-white/50">Executions</div>
                  </div>
                  <div className="text-center">
                    <div className="text-white font-medium">
                      {formatDuration(agent.metrics.avg_execution_time)}
                    </div>
                    <div className="text-white/50">Avg Time</div>
                  </div>
                </div>

                {/* Tools (in detailed view) */}
                {detailed && (
                  <div className="pt-2 border-t border-white/10">
                    <div className="text-xs text-white/60 mb-2">Available Tools:</div>
                    <div className="flex flex-wrap gap-1">
                      {agent.tools.slice(0, 3).map(tool => (
                        <span
                          key={tool}
                          className="px-2 py-1 bg-white/10 rounded text-xs text-white/70"
                        >
                          {tool.replace(/_/g, ' ')}
                        </span>
                      ))}
                      {agent.tools.length > 3 && (
                        <span className="px-2 py-1 bg-white/5 rounded text-xs text-white/50">
                          +{agent.tools.length - 3}
                        </span>
                      )}
                    </div>
                  </div>
                )}
              </div>

              {/* Expanded Details */}
              <AnimatePresence>
                {selectedAgent === agent.agent_id && (
                  <motion.div
                    initial={{ opacity: 0, height: 0 }}
                    animate={{ opacity: 1, height: 'auto' }}
                    exit={{ opacity: 0, height: 0 }}
                    transition={{ duration: 0.3 }}
                    className="mt-4 pt-4 border-t border-white/10 space-y-3"
                  >
                    <div className="grid grid-cols-3 gap-2 text-xs">
                      <div className="text-center">
                        <div className="text-green-400 font-medium">
                          {agent.metrics.successful_executions}
                        </div>
                        <div className="text-white/50">Success</div>
                      </div>
                      <div className="text-center">
                        <div className="text-red-400 font-medium">
                          {agent.metrics.failed_executions}
                        </div>
                        <div className="text-white/50">Failed</div>
                      </div>
                      <div className="text-center">
                        <div className="text-blue-400 font-medium">
                          {formatNumber(agent.metrics.total_tokens)}
                        </div>
                        <div className="text-white/50">Tokens</div>
                      </div>
                    </div>

                    {/* Action Buttons */}
                    <div className="flex gap-2 mt-4">
                      <motion.button
                        whileHover={{ scale: 1.05 }}
                        whileTap={{ scale: 0.95 }}
                        className="flex-1 px-3 py-2 bg-blue-500/20 hover:bg-blue-500/30 border border-blue-500/30 rounded-lg text-xs text-blue-400 font-medium transition-colors"
                      >
                        Execute Task
                      </motion.button>
                      <motion.button
                        whileHover={{ scale: 1.05 }}
                        whileTap={{ scale: 0.95 }}
                        className="px-3 py-2 bg-white/10 hover:bg-white/20 border border-white/20 rounded-lg text-xs text-white/70 font-medium transition-colors"
                      >
                        Configure
                      </motion.button>
                    </div>
                  </motion.div>
                )}
              </AnimatePresence>

              {/* Activity Indicator */}
              {agent.status === 'working' && (
                <div className="absolute -top-2 -right-2">
                  <motion.div
                    animate={{ rotate: 360 }}
                    transition={{ repeat: Infinity, duration: 2, ease: 'linear' }}
                    className="w-6 h-6 rounded-full bg-blue-500/20 border-2 border-blue-400 flex items-center justify-center"
                  >
                    <div className="w-2 h-2 bg-blue-400 rounded-full" />
                  </motion.div>
                </div>
              )}
            </InteractiveGlassCard>
          </motion.div>
        ))}
      </AnimatePresence>

      {/* Summary Card (in detailed view) */}
      {detailed && (
        <motion.div
          variants={itemVariants}
          className="md:col-span-2 lg:col-span-3"
        >
          <GlassCard size="lg">
            <div className="grid grid-cols-2 md:grid-cols-4 gap-6 text-center">
              <div>
                <div className="text-2xl font-bold text-white mb-2">
                  {agents.length}
                </div>
                <div className="text-sm text-white/70">Total Agents</div>
              </div>
              
              <div>
                <div className="text-2xl font-bold text-green-400 mb-2">
                  {agents.filter(a => a.status === 'idle' || a.status === 'complete').length}
                </div>
                <div className="text-sm text-white/70">Available</div>
              </div>
              
              <div>
                <div className="text-2xl font-bold text-blue-400 mb-2">
                  {agents.filter(a => a.status === 'working').length}
                </div>
                <div className="text-sm text-white/70">Active</div>
              </div>
              
              <div>
                <div className="text-2xl font-bold text-white mb-2">
                  {((agents.reduce((acc, agent) => acc + agent.metrics.success_rate, 0) / agents.length) * 100).toFixed(1)}%
                </div>
                <div className="text-sm text-white/70">Avg Success</div>
              </div>
            </div>
          </GlassCard>
        </motion.div>
      )}
    </motion.div>
  )
}

function getRandomStatus(): AgentStatus {
  const statuses: AgentStatus[] = ['idle', 'working', 'complete', 'error']
  return statuses[Math.floor(Math.random() * statuses.length)]
}