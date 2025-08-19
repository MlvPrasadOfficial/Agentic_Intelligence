"""
Research Agent - Specialized agent for web research and information gathering
"""
from typing import Dict, Any, Optional, List
from langchain.agents import AgentExecutor, create_react_agent
from langchain.prompts import PromptTemplate

from .base_agent import BaseAgent
from .tools import get_research_agent_tools

class ResearchAgent(BaseAgent):
    """
    Research Agent specialized in:
    - Web scraping and information gathering
    - Document analysis and summarization
    - Fact-checking and source validation
    """
    
    def __init__(
        self,
        llm_provider,
        memory_window: int = 10,
        websocket_manager=None
    ):
        # Get specialized tools for research
        tools = get_research_agent_tools()
        
        super().__init__(
            name="Research Agent",
            description="Specialized agent for web research, information gathering, and fact-checking",
            llm_provider=llm_provider,
            tools=tools,
            memory_window=memory_window,
            websocket_manager=websocket_manager
        )
        
        # Initialize the agent executor
        self.agent_executor = self.initialize_agent()
    
    def get_system_prompt(self) -> str:
        """
        Get the system prompt for the research agent
        """
        return """You are a highly skilled research agent specializing in information gathering and analysis.

Your capabilities include:
1. Web searching and scraping to find relevant information
2. Analyzing and summarizing documents
3. Fact-checking and validating sources
4. Identifying key insights and trends
5. Organizing information in a structured format

Guidelines:
- Always verify information from multiple sources when possible
- Cite your sources
- Be objective and present balanced viewpoints
- Focus on accuracy and relevance
- Summarize complex information clearly

You have access to the following tools:
{tools}

To use a tool, please use the following format:
Thought: I need to [describe what you need to do]
Action: [tool name]
Action Input: [tool input]
Observation: [tool output]

When you have gathered sufficient information, provide a comprehensive summary.

Current conversation:
{chat_history}