"""Fake chat model for tests: no network, no API key, scripted answers."""

from collections.abc import Callable, Sequence
from typing import Any

from langchain_core.language_models import LanguageModelInput
from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
from langchain_core.messages import AIMessage
from langchain_core.runnables import Runnable, RunnableConfig
from langchain_core.tools import BaseTool


class FakeChatModel(GenericFakeChatModel):
    """Replies with `messages` in order and records each conversation in `seen_inputs`."""

    seen_inputs: list[Any] = []
    tool_choices: list[str | None] = []

    def invoke(
        self,
        input: LanguageModelInput,
        config: RunnableConfig | None = None,
        *,
        stop: list[str] | None = None,
        **kwargs: Any,
    ) -> AIMessage:
        self.seen_inputs.append(list(input) if isinstance(input, list) else input)  # a snapshot, not a reference
        return super().invoke(input, config, stop=stop, **kwargs)

    def bind_tools(
        self,
        tools: Sequence[dict[str, Any] | type | Callable[..., Any] | BaseTool],
        *,
        tool_choice: str | None = None,
        **kwargs: Any,
    ) -> Runnable[LanguageModelInput, AIMessage]:
        self.tool_choices.append(tool_choice)
        return self
