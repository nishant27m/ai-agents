"""LangChain LLM wrapper for Google Gen AI (Gemini)

Usage:
    from langchain_google_genai import GoogleGenAILLM
    llm = GoogleGenAILLM(model="gemini-2.5-flash")
    print(llm("Write a 1-line haiku about coffee"))

This wrapper uses the `google-genai` SDK. The client will pick up
credentials from `GEMINI_API_KEY` (Gemini Developer API) or
Application Default Credentials when using Vertex AI.
"""
from __future__ import annotations

from typing import Any, List, Mapping, Optional

# langchain reorganized modules across versions; try common import locations
try:
    from langchain.llms.base import LLM  # new-style
except Exception:
    # Fallback minimal LLM for older/newer langchain variants or missing package
    class LLM:  # very small compatibility shim
        def __init__(self, **kwargs):
            pass

        # LangChain sometimes calls llm(prompt) directly, so provide __call__ bridge
        def __call__(self, prompt: str, **kwargs):
            if hasattr(self, "_call"):
                return self._call(prompt, **kwargs)
            raise NotImplementedError

        # Optional hooks that the real base provides
        def _identifying_params(self):
            return {}

        @property
        def _llm_type(self):
            return "google-genai-shim"

from google import genai

# Import traceable decorator for LangSmith tracing
try:
    from langsmith import traceable
except ImportError:
    # Fallback: no-op decorator if langsmith not installed
    def traceable(func=None, **kwargs):
        def decorator(f):
            return f
        return decorator if func is None else decorator(func)


class GoogleGenAILLM(LLM):
    """Minimal LangChain LLM wrapper around the google-genai SDK.

    Params:
        model: model id to use (default: gemini-2.5-flash)
        vertexai: set True to initialize Client(vertexai=True, ...)
    """

    model: str = "gemini-2.5-flash"
    vertexai: bool = False
    client: Optional[Any] = None

    def _get_client(self):
        if self.client is None:
            # The SDK will read GEMINI_API_KEY or ADC environment variables
            self.client = genai.Client(vertexai=self.vertexai)
        return self.client

    def _call(self, prompt: str, stop: Optional[List[str]] = None) -> str:
        """Call the LLM with tracing and token reporting to LangSmith."""
        client = self._get_client()
        response = client.models.generate_content(model=self.model, contents=prompt)
        return response.text
    
    # Apply LangSmith tracing decorator with token metadata capture
    _call = traceable(name="GoogleGenAI LLM Call", run_type="llm")(_call)

    def _identifying_params(self) -> Mapping[str, Any]:
        return {"model": self.model, "vertexai": self.vertexai}

    @property
    def _llm_type(self) -> str:
        return "google-genai"
