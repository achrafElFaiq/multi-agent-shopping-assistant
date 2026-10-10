"""Settings read from environment variables or a local `.env` file. Secrets never live in the code."""

from pydantic import SecretStr, ValidationError
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_ignore_empty=True, extra="ignore")
    openrouter_api_key: SecretStr
    database_url: SecretStr
    llm_model: str
    searcher_url: str = "http://localhost:8001/"  # the search agent's A2A server


def load_settings() -> Settings:
    try:
        return Settings()  # type: ignore[call-arg]
    except ValidationError as error:
        fields = ", ".join(str(issue["loc"][0]).upper() for issue in error.errors(include_input=False))
        raise RuntimeError(f"Missing or invalid environment variables: {fields}. Check your .env file.") from None
