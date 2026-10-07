"""Chat model served by OpenRouter. OpenRouter speaks the OpenAI API format, so ChatOpenAI works with its URL."""

import httpx
from langchain_openai import ChatOpenAI

from backend.config.settings import Settings

OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"


def make_chat_model(settings: Settings, http_client: httpx.Client | None = None) -> ChatOpenAI:
    """`http_client` lets the caller watch the raw HTTP traffic (see the terminal app's --raw option)."""
    return ChatOpenAI(
        model=settings.llm_model,
        api_key=settings.openrouter_api_key,
        base_url=OPENROUTER_BASE_URL,
        temperature=0,
        max_completion_tokens=2048,
        http_client=http_client,
    )
