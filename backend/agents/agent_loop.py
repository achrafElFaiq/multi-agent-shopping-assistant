"""The loop shared by the search and recommendation agents: the model calls tools until the work is done."""

import logging
from collections.abc import Callable, Sequence
from typing import TYPE_CHECKING

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AnyMessage, BaseMessage, HumanMessage, SystemMessage
from langchain_core.tools import BaseTool
from langgraph.errors import GraphRecursionError
from langgraph.graph import END, START, MessagesState, StateGraph
from langgraph.prebuilt import ToolNode, tools_condition

if TYPE_CHECKING:
    from langchain_core.runnables import RunnableConfig

logger = logging.getLogger(__name__)


def run_tool_agent(
    model: BaseChatModel,
    tools: list[BaseTool],
    system_prompt: str,
    context: str,
    done: Callable[[Sequence[BaseMessage]], bool],
    max_steps: int,
) -> list[BaseMessage]:
    """Run the model with `tools` until `done(messages)` is true after a tool call, and return the messages.

    The model must call a tool at every turn. If it is still not done after `max_steps` steps (one model turn or
    one tool turn each), the loop stops and only the opening messages are returned: tools keep their results
    elsewhere (the offer pool, the saved answers), so the caller can still go on.
    A tool can pause the whole graph with `interrupt()`; the loop resumes where it stopped.
    """
    model_with_tools = model.bind_tools(tools, tool_choice="required")

    def call_model(state: MessagesState) -> dict[str, list[BaseMessage]]:
        return {"messages": [model_with_tools.invoke(state["messages"])]}

    def after_tools(state: MessagesState) -> str:
        return END if done(state["messages"]) else "agent"

    graph = StateGraph(MessagesState)
    graph.add_node("agent", call_model)
    graph.add_node("tools", ToolNode(tools))
    graph.add_edge(START, "agent")
    graph.add_conditional_edges("agent", tools_condition)
    graph.add_conditional_edges("tools", after_tools, ["agent", END])
    agent = graph.compile()

    messages: list[AnyMessage] = [SystemMessage(system_prompt), HumanMessage(context)]
    try:
        config: RunnableConfig = {"recursion_limit": max_steps}
        result = agent.invoke({"messages": messages}, config)
        final: list[BaseMessage] = result["messages"]
        return final
    except GraphRecursionError:
        logger.warning("the agent did not finish in %d steps, so it was stopped", max_steps)
        return list(messages)
