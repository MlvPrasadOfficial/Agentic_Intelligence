'use client'

import React from 'react'
import { motion, MotionProps } from 'framer-motion'
import { cn } from '@/lib/utils'

export interface GlassCardProps extends Omit<React.HTMLAttributes<HTMLDivElement>, keyof MotionProps>, MotionProps {
  variant?: 'default' | 'dark' | 'subtle'
  hover?: boolean
  glow?: boolean
  size?: 'sm' | 'md' | 'lg'
  children: React.ReactNode
}

const variants = {
  default: 'bg-white/10 border-white/20 shadow-glass',
  dark: 'bg-black/20 border-black/30 shadow-glass-dark', 
  subtle: 'bg-white/5 border-white/10 shadow-sm'
}

const sizes = {
  sm: 'p-4 rounded-lg',
  md: 'p-6 rounded-xl',
  lg: 'p-8 rounded-2xl'
}

const hoverEffects = {
  default: 'hover:bg-white/20 hover:shadow-xl hover:scale-[1.02]',
  dark: 'hover:bg-black/30 hover:shadow-xl hover:scale-[1.02]',
  subtle: 'hover:bg-white/10 hover:shadow-lg hover:scale-[1.01]'
}

export const GlassCard = React.forwardRef<HTMLDivElement, GlassCardProps>(
  ({ 
    className,
    variant = 'default',
    hover = false,
    glow = false,
    size = 'md',
    children,
    ...props
  }, ref) => {
    return (
      <motion.div
        ref={ref}
        className={cn(
          // Base styles
          'backdrop-blur-xl border transition-all duration-300',
          
          // Variant styles
          variants[variant],
          
          // Size styles
          sizes[size],
          
          // Hover effects
          hover && hoverEffects[variant],
          
          // Glow effect
          glow && 'animate-glow',
          
          // Custom className
          className
        )}
        {...props}
      >
        {children}
      </motion.div>
    )
  }
)

GlassCard.displayName = 'GlassCard'

// Specialized glass card variants
export const FloatingGlassCard = React.forwardRef<HTMLDivElement, GlassCardProps>(
  ({ className, ...props }, ref) => {
    return (
      <GlassCard
        ref={ref}
        className={cn('animate-float', className)}
        initial={{ y: 20, opacity: 0 }}
        animate={{ y: 0, opacity: 1 }}
        transition={{ duration: 0.5, ease: 'easeOut' }}
        {...props}
      />
    )
  }
)

FloatingGlassCard.displayName = 'FloatingGlassCard'

export const InteractiveGlassCard = React.forwardRef<HTMLDivElement, GlassCardProps & {
  onClick?: () => void
}>(
  ({ className, onClick, ...props }, ref) => {
    return (
      <GlassCard
        ref={ref}
        className={cn(
          'cursor-pointer select-none',
          'active:scale-95',
          className
        )}
        hover={true}
        onClick={onClick}
        whileHover={{ y: -2 }}
        whileTap={{ scale: 0.98 }}
        {...props}
      />
    )
  }
)

InteractiveGlassCard.displayName = 'InteractiveGlassCard'

export const GlassCardHeader = React.forwardRef<
  HTMLDivElement,
  React.HTMLAttributes<HTMLDivElement>
>(({ className, ...props }, ref) => (
  <div
    ref={ref}
    className={cn('flex flex-col space-y-1.5 pb-6', className)}
    {...props}
  />
))

GlassCardHeader.displayName = 'GlassCardHeader'

export const GlassCardTitle = React.forwardRef<
  HTMLParagraphElement,
  React.HTMLAttributes<HTMLHeadingElement>
>(({ className, ...props }, ref) => (
  <h3
    ref={ref}
    className={cn(
      'text-2xl font-semibold leading-none tracking-tight text-white',
      className
    )}
    {...props}
  />
))

GlassCardTitle.displayName = 'GlassCardTitle'

export const GlassCardDescription = React.forwardRef<
  HTMLParagraphElement,
  React.HTMLAttributes<HTMLParagraphElement>
>(({ className, ...props }, ref) => (
  <p
    ref={ref}
    className={cn('text-sm text-white/70', className)}
    {...props}
  />
))

GlassCardDescription.displayName = 'GlassCardDescription'

export const GlassCardContent = React.forwardRef<
  HTMLDivElement,
  React.HTMLAttributes<HTMLDivElement>
>(({ className, ...props }, ref) => (
  <div ref={ref} className={cn('pt-0', className)} {...props} />
))

GlassCardContent.displayName = 'GlassCardContent'

export const GlassCardFooter = React.forwardRef<
  HTMLDivElement,
  React.HTMLAttributes<HTMLDivElement>
>(({ className, ...props }, ref) => (
  <div
    ref={ref}
    className={cn('flex items-center pt-6', className)}
    {...props}
  />
))

GlassCardFooter.displayName = 'GlassCardFooter'