import type { Metadata } from 'next'
import { Inter } from 'next/font/google'
import './globals.css'
import { ThemeProvider } from '@/components/theme-provider'
import { QueryProvider } from '@/components/query-provider'
import { WebSocketProvider } from '@/components/websocket-provider'
import { Toaster } from '@/components/ui/toaster'

const inter = Inter({ subsets: ['latin'] })

export const metadata: Metadata = {
  title: 'IntelliFlow - Multi-Agent Workflow Automation',
  description: 'Enterprise-grade AI-powered workflow automation platform with specialized agents for research, code generation, data analysis, and more.',
  keywords: 'AI, automation, workflow, agents, artificial intelligence, machine learning',
  authors: [{ name: 'IntelliFlow Team' }],
  viewport: 'width=device-width, initial-scale=1',
  themeColor: [
    { media: '(prefers-color-scheme: light)', color: 'white' },
    { media: '(prefers-color-scheme: dark)', color: 'black' },
  ],
  openGraph: {
    title: 'IntelliFlow - Multi-Agent Workflow Automation',
    description: 'Automate complex business workflows with AI-powered specialized agents',
    type: 'website',
    locale: 'en_US',
  },
}

export default function RootLayout({
  children,
}: {
  children: React.ReactNode
}) {
  return (
    <html lang="en" suppressHydrationWarning>
      <head>
        <link rel="icon" href="/favicon.ico" />
        <link rel="apple-touch-icon" href="/apple-touch-icon.png" />
        <link rel="manifest" href="/manifest.json" />
      </head>
      <body className={`${inter.className} min-h-screen bg-gradient-app dark:bg-gradient-app-dark antialiased`}>
        <ThemeProvider
          attribute="class"
          defaultTheme="system"
          enableSystem
          disableTransitionOnChange
        >
          <QueryProvider>
            <WebSocketProvider>
              <div className="relative min-h-screen">
                {/* Background pattern */}
                <div className="fixed inset-0 bg-grid-pattern opacity-5 dark:opacity-10" />
                
                {/* Main content */}
                <div className="relative z-10">
                  {children}
                </div>
                
                {/* Toast notifications */}
                <Toaster />
              </div>
            </WebSocketProvider>
          </QueryProvider>
        </ThemeProvider>
      </body>
    </html>
  )
}