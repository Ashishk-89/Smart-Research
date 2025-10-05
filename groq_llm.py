from langchain.llms.base import LLM
from typing import Optional, List, Mapping, Any
from groq import Client
from dotenv import load_dotenv
import os

load_dotenv()

class GroqLLM(LLM):
    """LangChain-compatible wrapper for Groq API"""

    api_key: str = os.getenv("GROQ_API_KEY")
    base_url: str = os.getenv("GROQ_API_BASE", "https://api.groq.com")
    model: str = "qwen/qwen3-32b"

    # lazy-loaded client
    _client: Optional[Client] = None

    @property
    def _llm_type(self) -> str:
        return "groq-llm"

    @property
    def client(self) -> Client:
        """Create the Groq client only once"""
        if self._client is None:
            self._client = Client(api_key=self.api_key, base_url=self.base_url)
        return self._client

    @property
    def _identifying_params(self) -> Mapping[str, Any]:
        return {"model": self.model, "base_url": self.base_url}

    def _call(self, prompt: str, stop: Optional[List[str]] = None) -> str:
        messages = [{"role": "user", "content": prompt}]
        resp = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            max_tokens=4096,
            temperature=0.0
        )
        try:
            return resp.choices[0].message.content
        except Exception:
            return str(resp)
