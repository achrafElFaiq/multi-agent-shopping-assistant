"""Chat model served by OpenRouter. OpenRouter speaks the OpenAI API format, so ChatOpenAI works with its URL."""

from langchain_openai import ChatOpenAI

from backend.config.settings import Settings

OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"


def make_chat_model(settings: Settings) -> ChatOpenAI:
    return ChatOpenAI(
        model=settings.llm_model,
        api_key=settings.openrouter_api_key,
        base_url=OPENROUTER_BASE_URL,
        temperature=0,  # same input, same answer: we want filled-in fields, not creativity
    )
