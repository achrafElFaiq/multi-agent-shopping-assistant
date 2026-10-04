"""Settings read from environment variables or a local `.env` file. Secrets never live in the code.

Copy `.env.example` to `.env` and fill in your own values. Real environment variables win over `.env`.
"""

from pydantic import SecretStr, ValidationError
from pydantic_settings import BaseSettings, SettingsConfigDict

DEFAULT_LLM_MODEL = "google/gemini-2.5-flash-lite"


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_ignore_empty=True, extra="ignore")

    openrouter_api_key: SecretStr  # read from OPENROUTER_API_KEY
    llm_model: str = DEFAULT_LLM_MODEL  # read from LLM_MODEL, any OpenRouter model id


def load_settings() -> Settings:
    try:
        return Settings()  # type: ignore[call-arg]  # values come from the environment, not arguments
    except ValidationError as error:
        raise RuntimeError(
            "OPENROUTER_API_KEY is not set. Copy .env.example to .env and add your key from https://openrouter.ai/keys"
        ) from error
