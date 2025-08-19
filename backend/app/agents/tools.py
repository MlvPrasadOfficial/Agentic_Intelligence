from typing import Optional, Dict, Any
from langchain.tools import Tool, StructuredTool
from langchain.pydantic_v1 import BaseModel, Field
import httpx
import json
from bs4 import BeautifulSoup
import pandas as pd
from datetime import datetime
import asyncio

# Tool Input Schemas
class WebSearchInput(BaseModel):
    query: str = Field(description="The search query")
    max_results: int = Field(default=5, description="Maximum number of results to return")

class WebScraperInput(BaseModel):
    url: str = Field(description="The URL to scrape")
    extract_text: bool = Field(default=True, description="Extract text content")
    extract_links: bool = Field(default=False, description="Extract links")

class DataAnalysisInput(BaseModel):
    data: str = Field(description="JSON string of data to analyze")
    analysis_type: str = Field(description="Type of analysis: summary, statistics, correlation")

class CodeGenerationInput(BaseModel):
    description: str = Field(description="Description of the code to generate")
    language: str = Field(default="python", description="Programming language")

class DatabaseQueryInput(BaseModel):
    query: str = Field(description="Database query to execute")
    database: str = Field(default="default", description="Database to query")

# Tool Functions
async def web_search_tool(query: str, max_results: int = 5) -> str:
    """
    Search the web for information
    """
    try:
        # This is a placeholder - in production, use a real search API
        async with httpx.AsyncClient() as client:
            # Mock search results for demo
            results = [
                {
                    "title": f"Result {i+1} for {query}",
                    "url": f"https://example.com/result{i+1}",
                    "snippet": f"This is a snippet for search result {i+1} about {query}"
                }
                for i in range(min(max_results, 3))
            ]
            
            return json.dumps(results, indent=2)
    except Exception as e:
        return f"Error performing web search: {str(e)}"

async def web_scraper_tool(url: str, extract_text: bool = True, extract_links: bool = False) -> str:
    """
    Scrape content from a webpage
    """
    try:
        async with httpx.AsyncClient() as client:
            response = await client.get(url, timeout=10)
            response.raise_for_status()
            
            soup = BeautifulSoup(response.text, 'html.parser')
            result = {"url": url}
            
            if extract_text:
                # Remove script and style elements
                for script in soup(["script", "style"]):
                    script.decompose()
                text = soup.get_text(separator='\n', strip=True)
                result["text"] = text[:5000]  # Limit text length
            
            if extract_links:
                links = []
                for link in soup.find_all('a', href=True):
                    links.append({
                        "text": link.text.strip(),
                        "url": link['href']
                    })
                result["links"] = links[:20]  # Limit number of links
            
            return json.dumps(result, indent=2)
    except Exception as e:
        return f"Error scraping webpage: {str(e)}"

def data_analysis_tool(data: str, analysis_type: str) -> str:
    """
    Perform data analysis on provided data
    """
    try:
        # Parse the JSON data
        parsed_data = json.loads(data)
        
        if analysis_type == "summary":
            # Create summary statistics
            df = pd.DataFrame(parsed_data)
            summary = {
                "shape": df.shape,
                "columns": list(df.columns),
                "dtypes": df.dtypes.to_dict(),
                "null_counts": df.isnull().sum().to_dict(),
                "description": df.describe().to_dict() if not df.empty else {}
            }
            return json.dumps(summary, indent=2, default=str)
        
        elif analysis_type == "statistics":
            df = pd.DataFrame(parsed_data)
            numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
            stats = {}
            for col in numeric_cols:
                stats[col] = {
                    "mean": df[col].mean(),
                    "median": df[col].median(),
                    "std": df[col].std(),
                    "min": df[col].min(),
                    "max": df[col].max()
                }
            return json.dumps(stats, indent=2, default=str)
        
        elif analysis_type == "correlation":
            df = pd.DataFrame(parsed_data)
            numeric_df = df.select_dtypes(include=['number'])
            if not numeric_df.empty:
                correlation = numeric_df.corr().to_dict()
                return json.dumps(correlation, indent=2, default=str)
            else:
                return "No numeric columns found for correlation analysis"
        
        else:
            return f"Unknown analysis type: {analysis_type}"
            
    except Exception as e:
        return f"Error performing data analysis: {str(e)}"

def code_generation_tool(description: str, language: str = "python") -> str:
    """
    Generate code based on description
    """
    try:
        # This is a placeholder - in production, use LLM for code generation
        templates = {
            "python": f'''"""
{description}
"""

def generated_function():
    # TODO: Implement {description}
    pass

if __name__ == "__main__":
    generated_function()
''',
            "javascript": f'''/**
 * {description}
 */

function generatedFunction() {{
    // TODO: Implement {description}
}}

module.exports = {{ generatedFunction }};
''',
            "java": f'''/**
 * {description}
 */

public class GeneratedCode {{
    public static void main(String[] args) {{
        // TODO: Implement {description}
    }}
}}
'''
        }
        
        code = templates.get(language.lower(), f"// TODO: Implement {description}")
        return code
        
    except Exception as e:
        return f"Error generating code: {str(e)}"

def file_operations_tool(operation: str, path: str, content: Optional[str] = None) -> str:
    """
    Perform file operations (read, write, list)
    """
    try:
        import os
        
        if operation == "read":
            with open(path, 'r') as f:
                return f.read()
        
        elif operation == "write":
            if content is None:
                return "Content is required for write operation"
            with open(path, 'w') as f:
                f.write(content)
            return f"Successfully wrote to {path}"
        
        elif operation == "list":
            if os.path.isdir(path):
                files = os.listdir(path)
                return json.dumps(files, indent=2)
            else:
                return f"{path} is not a directory"
        
        else:
            return f"Unknown operation: {operation}"
            
    except Exception as e:
        return f"Error performing file operation: {str(e)}"

def calculator_tool(expression: str) -> str:
    """
    Evaluate mathematical expressions
    """
    try:
        # Safe evaluation of mathematical expressions
        import ast
        import operator as op
        
        # Supported operators
        operators = {
            ast.Add: op.add,
            ast.Sub: op.sub,
            ast.Mult: op.mul,
            ast.Div: op.truediv,
            ast.Pow: op.pow,
            ast.USub: op.neg,
        }
        
        def eval_expr(expr):
            if isinstance(expr, ast.Num):
                return expr.n
            elif isinstance(expr, ast.BinOp):
                return operators[type(expr.op)](
                    eval_expr(expr.left),
                    eval_expr(expr.right)
                )
            elif isinstance(expr, ast.UnaryOp):
                return operators[type(expr.op)](eval_expr(expr.operand))
            else:
                raise TypeError(f"Unsupported expression type: {type(expr)}")
        
        tree = ast.parse(expression, mode='eval')
        result = eval_expr(tree.body)
        return str(result)
        
    except Exception as e:
        return f"Error evaluating expression: {str(e)}"

# Create LangChain Tools
def create_agent_tools() -> list:
    """
    Create and return all available tools for agents
    """
    tools = [
        StructuredTool.from_function(
            func=web_search_tool,
            name="web_search",
            description="Search the web for information",
            args_schema=WebSearchInput,
            coroutine=web_search_tool
        ),
        StructuredTool.from_function(
            func=web_scraper_tool,
            name="web_scraper",
            description="Scrape content from a webpage",
            args_schema=WebScraperInput,
            coroutine=web_scraper_tool
        ),
        Tool(
            name="data_analysis",
            func=data_analysis_tool,
            description="Perform data analysis on provided data. Input should be JSON string and analysis type."
        ),
        Tool(
            name="code_generator",
            func=code_generation_tool,
            description="Generate code based on description. Specify the description and programming language."
        ),
        Tool(
            name="calculator",
            func=calculator_tool,
            description="Evaluate mathematical expressions. Input should be a valid mathematical expression."
        ),
        Tool(
            name="file_operations",
            func=file_operations_tool,
            description="Perform file operations (read, write, list). Specify operation, path, and content if needed."
        )
    ]
    
    return tools

# Specialized tools for specific agents
def get_research_agent_tools() -> list:
    """
    Get tools specific to research agent
    """
    return [
        tool for tool in create_agent_tools()
        if tool.name in ["web_search", "web_scraper", "file_operations"]
    ]

def get_data_agent_tools() -> list:
    """
    Get tools specific to data analysis agent
    """
    return [
        tool for tool in create_agent_tools()
        if tool.name in ["data_analysis", "calculator", "file_operations"]
    ]

def get_code_agent_tools() -> list:
    """
    Get tools specific to code generation agent
    """
    return [
        tool for tool in create_agent_tools()
        if tool.name in ["code_generator", "file_operations", "web_search"]
    ]

def get_communication_agent_tools() -> list:
    """
    Get tools specific to communication agent
    """
    return [
        tool for tool in create_agent_tools()
        if tool.name in ["web_search", "file_operations"]
    ]

def get_planning_agent_tools() -> list:
    """
    Get tools specific to planning agent
    """
    return [
        tool for tool in create_agent_tools()
        if tool.name in ["calculator", "file_operations"]
    ]