'use client'

import React, { useState } from 'react'
import { motion, AnimatePresence } from 'framer-motion'
import { 
  HomeIcon,
  ChartBarSquareIcon,
  CubeIcon,
  CpuChipIcon,
  ChartPieIcon,
  Bars3Icon,
  XMarkIcon,
  WifiIcon,
  SunIcon,
  MoonIcon,
  ComputerDesktopIcon
} from '@heroicons/react/24/outline'
import { GlassCard } from './ui/glass-card'
import { cn } from '@/lib/utils'
import { ViewType } from '@/types'
import { useTheme } from 'next-themes'

interface NavigationProps {
  currentView: ViewType
  onViewChange: (view: ViewType) => void
  connectionStatus: 'connected' | 'connecting' | 'disconnected' | 'error'
}

const navItems = [
  { id: 'home' as ViewType, label: 'Home', icon: HomeIcon },
  { id: 'dashboard' as ViewType, label: 'Dashboard', icon: ChartBarSquareIcon },
  { id: 'workflows' as ViewType, label: 'Workflows', icon: CubeIcon },
  { id: 'agents' as ViewType, label: 'Agents', icon: CpuChipIcon },
  { id: 'analytics' as ViewType, label: 'Analytics', icon: ChartPieIcon },
]

const connectionStatusConfig = {
  connected: { color: 'text-green-400', bgColor: 'bg-green-400/20', label: 'Connected' },
  connecting: { color: 'text-yellow-400', bgColor: 'bg-yellow-400/20', label: 'Connecting...' },
  disconnected: { color: 'text-red-400', bgColor: 'bg-red-400/20', label: 'Disconnected' },
  error: { color: 'text-red-500', bgColor: 'bg-red-500/20', label: 'Connection Error' },
}

export function Navigation({ currentView, onViewChange, connectionStatus }: NavigationProps) {
  const [isMobileMenuOpen, setIsMobileMenuOpen] = useState(false)
  const { theme, setTheme } = useTheme()

  const toggleMobileMenu = () => {
    setIsMobileMenuOpen(!isMobileMenuOpen)
  }

  const handleNavItemClick = (view: ViewType) => {
    onViewChange(view)
    setIsMobileMenuOpen(false)
  }

  const cycleTheme = () => {
    if (theme === 'light') setTheme('dark')
    else if (theme === 'dark') setTheme('system')
    else setTheme('light')
  }

  const getThemeIcon = () => {
    switch (theme) {
      case 'light': return SunIcon
      case 'dark': return MoonIcon
      default: return ComputerDesktopIcon
    }
  }

  return (
    <>
      {/* Desktop Navigation */}
      <motion.nav
        initial={{ y: -100, opacity: 0 }}
        animate={{ y: 0, opacity: 1 }}
        transition={{ duration: 0.5, ease: 'easeOut' }}
        className="fixed top-0 left-0 right-0 z-50 hidden md:block"
      >
        <div className="container mx-auto px-6 py-4">
          <GlassCard className="flex items-center justify-between">
            {/* Logo */}
            <motion.div
              whileHover={{ scale: 1.05 }}
              className="flex items-center gap-3"
            >
              <div className="w-8 h-8 bg-gradient-to-br from-blue-500 to-purple-600 rounded-lg flex items-center justify-center">
                <CubeIcon className="w-5 h-5 text-white" />
              </div>
              <span className="text-xl font-bold text-white">IntelliFlow</span>
            </motion.div>

            {/* Navigation Items */}
            <div className="flex items-center gap-2">
              {navItems.map((item) => {
                const Icon = item.icon
                const isActive = currentView === item.id
                
                return (
                  <motion.button
                    key={item.id}
                    onClick={() => handleNavItemClick(item.id)}
                    className={cn(
                      'relative px-4 py-2 rounded-lg text-sm font-medium transition-all duration-200',
                      'hover:bg-white/20',
                      isActive 
                        ? 'text-blue-400 bg-blue-500/20 border border-blue-500/30' 
                        : 'text-white/70 hover:text-white'
                    )}
                    whileHover={{ scale: 1.05 }}
                    whileTap={{ scale: 0.95 }}
                  >
                    <div className="flex items-center gap-2">
                      <Icon className="w-4 h-4" />
                      <span>{item.label}</span>
                    </div>
                    
                    {isActive && (
                      <motion.div
                        layoutId="activeTab"
                        className="absolute inset-0 bg-blue-500/10 rounded-lg border border-blue-500/20 -z-10"
                        initial={false}
                        transition={{ type: 'spring', stiffness: 300, damping: 30 }}
                      />
                    )}
                  </motion.button>
                )
              })}
            </div>

            {/* Right Side Controls */}
            <div className="flex items-center gap-3">
              {/* Connection Status */}
              <motion.div
                className="flex items-center gap-2"
                whileHover={{ scale: 1.05 }}
              >
                <div className={cn(
                  'flex items-center gap-2 px-3 py-1.5 rounded-full text-xs font-medium',
                  connectionStatusConfig[connectionStatus].bgColor,
                  connectionStatusConfig[connectionStatus].color
                )}>
                  <WifiIcon className="w-3 h-3" />
                  <span>{connectionStatusConfig[connectionStatus].label}</span>
                  {connectionStatus === 'connected' && (
                    <motion.div
                      animate={{ scale: [1, 1.2, 1] }}
                      transition={{ repeat: Infinity, duration: 2 }}
                      className="w-1.5 h-1.5 bg-current rounded-full"
                    />
                  )}
                </div>
              </motion.div>

              {/* Theme Toggle */}
              <motion.button
                onClick={cycleTheme}
                className="p-2 rounded-lg text-white/70 hover:text-white hover:bg-white/20 transition-colors"
                whileHover={{ scale: 1.1, rotate: 15 }}
                whileTap={{ scale: 0.9 }}
                title={`Current theme: ${theme}`}
              >
                <motion.div
                  key={theme}
                  initial={{ rotate: -180, opacity: 0 }}
                  animate={{ rotate: 0, opacity: 1 }}
                  transition={{ duration: 0.3 }}
                >
                  {React.createElement(getThemeIcon(), { className: 'w-5 h-5' })}
                </motion.div>
              </motion.button>
            </div>
          </GlassCard>
        </div>
      </motion.nav>

      {/* Mobile Navigation */}
      <motion.nav
        initial={{ y: -100, opacity: 0 }}
        animate={{ y: 0, opacity: 1 }}
        transition={{ duration: 0.5, ease: 'easeOut' }}
        className="fixed top-0 left-0 right-0 z-50 md:hidden"
      >
        <div className="px-4 py-3">
          <GlassCard className="flex items-center justify-between" size="sm">
            {/* Logo */}
            <div className="flex items-center gap-2">
              <div className="w-6 h-6 bg-gradient-to-br from-blue-500 to-purple-600 rounded-md flex items-center justify-center">
                <CubeIcon className="w-4 h-4 text-white" />
              </div>
              <span className="text-lg font-bold text-white">IntelliFlow</span>
            </div>

            {/* Mobile Menu Button */}
            <div className="flex items-center gap-2">
              {/* Connection Status (Mobile) */}
              <div className={cn(
                'w-2 h-2 rounded-full',
                connectionStatus === 'connected' ? 'bg-green-400' :
                connectionStatus === 'connecting' ? 'bg-yellow-400 animate-pulse' :
                'bg-red-400'
              )} />

              <motion.button
                onClick={toggleMobileMenu}
                className="p-2 rounded-lg text-white/70 hover:text-white hover:bg-white/20 transition-colors"
                whileHover={{ scale: 1.1 }}
                whileTap={{ scale: 0.9 }}
              >
                {isMobileMenuOpen ? (
                  <XMarkIcon className="w-5 h-5" />
                ) : (
                  <Bars3Icon className="w-5 h-5" />
                )}
              </motion.button>
            </div>
          </GlassCard>
        </div>
      </motion.nav>

      {/* Mobile Menu Overlay */}
      <AnimatePresence>
        {isMobileMenuOpen && (
          <>
            {/* Backdrop */}
            <motion.div
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
              exit={{ opacity: 0 }}
              className="fixed inset-0 bg-black/50 backdrop-blur-sm z-40 md:hidden"
              onClick={() => setIsMobileMenuOpen(false)}
            />

            {/* Menu Panel */}
            <motion.div
              initial={{ x: '100%', opacity: 0 }}
              animate={{ x: 0, opacity: 1 }}
              exit={{ x: '100%', opacity: 0 }}
              transition={{ type: 'spring', damping: 25, stiffness: 200 }}
              className="fixed top-0 right-0 h-full w-80 z-50 md:hidden"
            >
              <GlassCard className="h-full rounded-none rounded-l-2xl p-6">
                <div className="flex flex-col h-full">
                  {/* Header */}
                  <div className="flex items-center justify-between mb-8">
                    <h2 className="text-xl font-semibold text-white">Navigation</h2>
                    <motion.button
                      onClick={() => setIsMobileMenuOpen(false)}
                      className="p-2 rounded-lg text-white/70 hover:text-white hover:bg-white/20 transition-colors"
                      whileHover={{ scale: 1.1 }}
                      whileTap={{ scale: 0.9 }}
                    >
                      <XMarkIcon className="w-5 h-5" />
                    </motion.button>
                  </div>

                  {/* Navigation Items */}
                  <div className="flex-1 space-y-2">
                    {navItems.map((item, index) => {
                      const Icon = item.icon
                      const isActive = currentView === item.id
                      
                      return (
                        <motion.button
                          key={item.id}
                          initial={{ x: 50, opacity: 0 }}
                          animate={{ x: 0, opacity: 1 }}
                          transition={{ delay: index * 0.1 }}
                          onClick={() => handleNavItemClick(item.id)}
                          className={cn(
                            'w-full flex items-center gap-3 px-4 py-3 rounded-xl text-left transition-all duration-200',
                            'hover:bg-white/20',
                            isActive 
                              ? 'text-blue-400 bg-blue-500/20 border border-blue-500/30' 
                              : 'text-white/70 hover:text-white'
                          )}
                          whileHover={{ scale: 1.02, x: 5 }}
                          whileTap={{ scale: 0.98 }}
                        >
                          <Icon className="w-5 h-5" />
                          <span className="font-medium">{item.label}</span>
                          {isActive && (
                            <motion.div
                              className="ml-auto w-2 h-2 bg-blue-400 rounded-full"
                              initial={{ scale: 0 }}
                              animate={{ scale: 1 }}
                              transition={{ type: 'spring' }}
                            />
                          )}
                        </motion.button>
                      )
                    })}
                  </div>

                  {/* Footer */}
                  <div className="pt-6 border-t border-white/10 space-y-4">
                    {/* Connection Status */}
                    <div className={cn(
                      'flex items-center gap-3 px-4 py-3 rounded-xl',
                      connectionStatusConfig[connectionStatus].bgColor
                    )}>
                      <WifiIcon className={cn('w-5 h-5', connectionStatusConfig[connectionStatus].color)} />
                      <span className={cn('text-sm font-medium', connectionStatusConfig[connectionStatus].color)}>
                        {connectionStatusConfig[connectionStatus].label}
                      </span>
                    </div>

                    {/* Theme Toggle */}
                    <motion.button
                      onClick={cycleTheme}
                      className="w-full flex items-center gap-3 px-4 py-3 rounded-xl text-white/70 hover:text-white hover:bg-white/20 transition-colors"
                      whileHover={{ scale: 1.02 }}
                      whileTap={{ scale: 0.98 }}
                    >
                      {React.createElement(getThemeIcon(), { className: 'w-5 h-5' })}
                      <span className="text-sm font-medium">
                        Theme: {theme === 'system' ? 'Auto' : theme === 'light' ? 'Light' : 'Dark'}
                      </span>
                    </motion.button>
                  </div>
                </div>
              </GlassCard>
            </motion.div>
          </>
        )}
      </AnimatePresence>
    </>
  )
}