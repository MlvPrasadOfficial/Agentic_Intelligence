"""
LLM Service for managing Ollama and language model interactions
"""
from typing import Dict, Any, Optional, List, AsyncGenerator
import httpx
import logging
from datetime import datetime
import json

from langchain.llms import Ollama
from langchain.chat_models import ChatOllama
from langchain.callbacks.manager import CallbackManagerForLLMRun
from langchain.schema import BaseMessage

logger = logging.getLogger(__name__)

class LLMService:
    """
    Service for managing LLM interactions with Ollama
    """
    
    def __init__(
        self,
        base_url: str = "http://localhost:11434",
        model: str = "llama3.1",
        temperature: float = 0.7,
        max_tokens: int = 2048
    ):
        self.base_url = base_url
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.is_connected = False
        
        # Initialize Ollama clients
        self.llm = None
        self.chat_llm = None
        
        # Metrics
        self.total_requests = 0
        self.total_tokens = 0
        self.total_cost = 0.0
        
    async def connect(self) -> bool:
        """
        Connect to Ollama server and verify model availability
        """
        try:
            async with httpx.AsyncClient() as client:
                # Check if Ollama is running
                response = await client.get(f"{self.base_url}/api/tags")
                if response.status_code != 200:
                    logger.error(f"Failed to connect to Ollama at {self.base_url}")
                    return False
                
                # Check if model is available
                models = response.json().get("models", [])
                model_names = [m.get("name", "").split(":")[0] for m in models]
                
                if self.model not in model_names:
                    logger.warning(f"Model {self.model} not found. Available models: {model_names}")
                    # Try to pull the model
                    await self.pull_model(self.model)
                
                # Initialize LangChain Ollama instances
                self.llm = Ollama(
                    base_url=self.base_url,
                    model=self.model,
                    temperature=self.temperature,
                    num_predict=self.max_tokens
                )
                
                self.chat_llm = ChatOllama(
                    base_url=self.base_url,
                    model=self.model,
                    temperature=self.temperature
                )
                
                self.is_connected = True
                logger.info(f"Connected to Ollama at {self.base_url} with model {self.model}")
                return True
                
        except Exception as e:
            logger.error(f"Failed to connect to Ollama: {str(e)}")
            self.is_connected = False
            return False
    
    async def pull_model(self, model_name: str) -> bool:
        """
        Pull a model from Ollama library
        """
        try:
            logger.info(f"Pulling model {model_name}...")
            async with httpx.AsyncClient(timeout=None) as client:
                response = await client.post(
                    f"{self.base_url}/api/pull",
                    json={"name": model_name},
                    timeout=3600  # 1 hour timeout for large models
                )
                
                if response.status_code == 200:
                    logger.info(f"Successfully pulled model {model_name}")
                    return True
                else:
                    logger.error(f"Failed to pull model {model_name}: {response.text}")
                    return False
                    
        except Exception as e:
            logger.error(f"Error pulling model {model_name}: {str(e)}")
            return False
    
    async def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        stream: bool = False
    ) -> Union[str, AsyncGenerator[str, None]]:
        """
        Generate text using the LLM
        """
        if not self.is_connected:
            raise ConnectionError("Not connected to Ollama. Call connect() first.")
        
        self.total_requests += 1
        
        # Build the full prompt
        full_prompt = ""
        if system_prompt:
            full_prompt = f"System: {system_prompt}\n\n"
        full_prompt += f"User: {prompt}\n\nAssistant:"
        
        try:
            if stream:
                return self._stream_generate(
                    full_prompt,
                    temperature or self.temperature,
                    max_tokens or self.max_tokens
                )
            else:
                response = await self._generate(
                    full_prompt,
                    temperature or self.temperature,
                    max_tokens or self.max_tokens
                )
                return response
                
        except Exception as e:
            logger.error(f"Generation failed: {str(e)}")
            raise
    
    async def _generate(
        self,
        prompt: str,
        temperature: float,
        max_tokens: int
    ) -> str:
        """
        Generate text without streaming
        """
        async with httpx.AsyncClient(timeout=60) as client:
            response = await client.post(
                f"{self.base_url}/api/generate",
                json={
                    "model": self.model,
                    "prompt": prompt,
                    "temperature": temperature,
                    "num_predict": max_tokens,
                    "stream": False
                }
            )
            
            if response.status_code == 200:
                result = response.json()
                generated_text = result.get("response", "")
                
                # Update metrics
                if "eval_count" in result:
                    self.total_tokens += result["eval_count"]
                
                return generated_text
            else:
                raise Exception(f"Generation failed: {response.text}")
    
    async def _stream_generate(
        self,
        prompt: str,
        temperature: float,
        max_tokens: int
    ) -> AsyncGenerator[str, None]:
        """
        Stream text generation
        """
        async with httpx.AsyncClient(timeout=60) as client:
            async with client.stream(
                "POST",
                f"{self.base_url}/api/generate",
                json={
                    "model": self.model,
                    "prompt": prompt,
                    "temperature": temperature,
                    "num_predict": max_tokens,
                    "stream": True
                }
            ) as response:
                async for line in response.aiter_lines():
                    if line:
                        try:
                            data = json.loads(line)
                            if "response" in data:
                                yield data["response"]
                            if data.get("done", False):
                                if "eval_count" in data:
                                    self.total_tokens += data["eval_count"]
                                break
                        except json.JSONDecodeError:
                            continue
    
    async def chat(
        self,
        messages: List[Dict[str, str]],
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        stream: bool = False
    ) -> Union[str, AsyncGenerator[str, None]]:
        """
        Chat with the LLM using message history
        """
        if not self.is_connected:
            raise ConnectionError("Not connected to Ollama. Call connect() first.")
        
        self.total_requests += 1
        
        # Format messages for Ollama
        formatted_messages = []
        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")
            formatted_messages.append({"role": role, "content": content})
        
        try:
            if stream:
                return self._stream_chat(
                    formatted_messages,
                    temperature or self.temperature,
                    max_tokens or self.max_tokens
                )
            else:
                response = await self._chat(
                    formatted_messages,
                    temperature or self.temperature,
                    max_tokens or self.max_tokens
                )
                return response
                
        except Exception as e:
            logger.error(f"Chat failed: {str(e)}")
            raise
    
    async def _chat(
        self,
        messages: List[Dict[str, str]],
        temperature: float,
        max_tokens: int
    ) -> str:
        """
        Chat without streaming
        """
        async with httpx.AsyncClient(timeout=60) as client:
            response = await client.post(
                f"{self.base_url}/api/chat",
                json={
                    "model": self.model,
                    "messages": messages,
                    "temperature": temperature,
                    "num_predict": max_tokens,
                    "stream": False
                }
            )
            
            if response.status_code == 200:
                result = response.json()
                message = result.get("message", {})
                generated_text = message.get("content", "")
                
                # Update metrics
                if "eval_count" in result:
                    self.total_tokens += result["eval_count"]
                
                return generated_text
            else:
                raise Exception(f"Chat failed: {response.text}")
    
    async def _stream_chat(
        self,
        messages: List[Dict[str, str]],
        temperature: float,
        max_tokens: int
    ) -> AsyncGenerator[str, None]:
        """
        Stream chat responses
        """
        async with httpx.AsyncClient(timeout=60) as client:
            async with client.stream(
                "POST",
                f"{self.base_url}/api/chat",
                json={
                    "model": self.model,
                    "messages": messages,
                    "temperature": temperature,
                    "num_predict": max_tokens,
                    "stream": True
                }
            ) as response:
                async for line in response.aiter_lines():
                    if line:
                        try:
                            data = json.loads(line)
                            if "message" in data:
                                content = data["message"].get("content", "")
                                if content:
                                    yield content
                            if data.get("done", False):
                                if "eval_count" in data:
                                    self.total_tokens += data["eval_count"]
                                break
                        except json.JSONDecodeError:
                            continue
    
    async def embed(self, text: str) -> List[float]:
        """
        Generate embeddings for text
        """
        if not self.is_connected:
            raise ConnectionError("Not connected to Ollama. Call connect() first.")
        
        try:
            async with httpx.AsyncClient(timeout=30) as client:
                response = await client.post(
                    f"{self.base_url}/api/embeddings",
                    json={
                        "model": self.model,
                        "prompt": text
                    }
                )
                
                if response.status_code == 200:
                    result = response.json()
                    return result.get("embedding", [])
                else:
                    raise Exception(f"Embedding failed: {response.text}")
                    
        except Exception as e:
            logger.error(f"Embedding generation failed: {str(e)}")
            raise
    
    def get_llm(self) -> Ollama:
        """
        Get LangChain Ollama LLM instance
        """
        if not self.llm:
            raise ConnectionError("LLM not initialized. Call connect() first.")
        return self.llm
    
    def get_chat_llm(self) -> ChatOllama:
        """
        Get LangChain ChatOllama instance
        """
        if not self.chat_llm:
            raise ConnectionError("Chat LLM not initialized. Call connect() first.")
        return self.chat_llm
    
    async def list_models(self) -> List[Dict[str, Any]]:
        """
        List available models
        """
        try:
            async with httpx.AsyncClient() as client:
                response = await client.get(f"{self.base_url}/api/tags")
                if response.status_code == 200:
                    models = response.json().get("models", [])
                    return [
                        {
                            "name": m.get("name", ""),
                            "size": m.get("size", 0),
                            "modified": m.get("modified_at", "")
                        }
                        for m in models
                    ]
                else:
                    return []
                    
        except Exception as e:
            logger.error(f"Failed to list models: {str(e)}")
            return []
    
    async def switch_model(self, model_name: str) -> bool:
        """
        Switch to a different model
        """
        self.model = model_name
        return await self.connect()
    
    def get_metrics(self) -> Dict[str, Any]:
        """
        Get LLM usage metrics
        """
        return {
            "total_requests": self.total_requests,
            "total_tokens": self.total_tokens,
            "avg_tokens_per_request": (
                self.total_tokens / self.total_requests
                if self.total_requests > 0 else 0
            ),
            "model": self.model,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "is_connected": self.is_connected
        }
    
    async def health_check(self) -> Dict[str, Any]:
        """
        Check Ollama service health
        """
        try:
            async with httpx.AsyncClient(timeout=5) as client:
                response = await client.get(f"{self.base_url}/api/tags")
                
                if response.status_code == 200:
                    models = response.json().get("models", [])
                    model_names = [m.get("name", "").split(":")[0] for m in models]
                    
                    return {
                        "status": "healthy",
                        "ollama_url": self.base_url,
                        "current_model": self.model,
                        "model_available": self.model in model_names,
                        "available_models": model_names,
                        "timestamp": datetime.now().isoformat()
                    }
                else:
                    return {
                        "status": "unhealthy",
                        "error": f"Ollama returned status {response.status_code}",
                        "timestamp": datetime.now().isoformat()
                    }
                    
        except Exception as e:
            return {
                "status": "unhealthy",
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }