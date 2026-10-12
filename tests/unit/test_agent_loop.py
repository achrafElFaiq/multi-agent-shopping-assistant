"""The shared loop: a model that answers with text instead of a tool is asked again, not stopped."""

from langchain_core.messages import AIMessage, ToolMessage
from langchain_core.messages.tool import ToolCall
from langchain_core.tools import tool

from backend.agents.agent_loop import run_tool_agent
from tests.fakes import FakeChatModel


@tool
def finish() -> str:
    """Finish."""
    return "done"


def test_text_answer_is_reminded_then_the_agent_goes_on() -> None:
    model = FakeChatModel(
        messages=iter(
            [AIMessage("Here are some shoes!"), AIMessage("", tool_calls=[ToolCall(name="finish", args={}, id="f")])]
        )
    )
    messages = run_tool_agent(
        model,
        [finish],
        "system",
        "context",
        done=lambda ms: any(isinstance(m, ToolMessage) for m in ms),
        max_steps=10,
    )
    assert any(isinstance(m, ToolMessage) and m.name == "finish" for m in messages)
