"""Fake chat model for tests: no network, no API key, scripted answers."""

from collections.abc import Callable, Sequence
from typing import Any

from langchain_core.language_models import LanguageModelInput
from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
from langchain_core.messages import AIMessage
from langchain_core.runnables import Runnable, RunnableLambda
from langchain_core.tools import BaseTool
from pydantic import BaseModel


class FakeChatModel(GenericFakeChatModel):
    """Replies with `messages` in order (they may contain tool calls).

    Structured output returns `structured_answer`, and records the conversation it was given in `last_input`.
    """

    structured_answer: BaseModel | None = None
    last_input: Any = None

    def bind_tools(
        self,
        tools: Sequence[dict[str, Any] | type | Callable[..., Any] | BaseTool],
        *,
        tool_choice: str | None = None,
        **kwargs: Any,
    ) -> Runnable[LanguageModelInput, AIMessage]:
        return self

    def with_structured_output(self, schema: Any, **kwargs: Any) -> Runnable[LanguageModelInput, Any]:
        def answer(conversation: LanguageModelInput) -> BaseModel | None:
            self.last_input = conversation
            return self.structured_answer

        return RunnableLambda(answer)
