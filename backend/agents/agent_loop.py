"""The loop shared by the search and recommendation agents: the model calls tools until the work is done."""

import logging
from collections.abc import Callable, Sequence
from typing import TYPE_CHECKING

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, AnyMessage, BaseMessage, HumanMessage, SystemMessage
from langchain_core.tools import BaseTool
from langgraph.errors import GraphRecursionError
from langgraph.graph import END, START, MessagesState, StateGraph
from langgraph.graph.state import CompiledStateGraph
from langgraph.prebuilt import ToolNode
from langgraph.types import Checkpointer

if TYPE_CHECKING:
    from langchain_core.runnables import RunnableConfig

logger = logging.getLogger(__name__)


def build_tool_agent(
    model: BaseChatModel,
    tools: list[BaseTool],
    done: Callable[[Sequence[BaseMessage]], bool],
    checkpointer: Checkpointer = None,
) -> CompiledStateGraph[MessagesState, None, MessagesState, MessagesState]:
    """The graph: agent -> tools -> agent ... until `done(messages)` is true after a tool call.

    The model must call a tool at every turn; if it answers with text instead (some models do, even when told not
    to), it is reminded and asked again, until the step limit.
    Give a `checkpointer` when this graph is the top one and pauses (`interrupt()`); a graph run inside another one
    uses the checkpointer of that one.
    """
    model_with_tools = model.bind_tools(tools, tool_choice="required")

    def call_model(state: MessagesState) -> dict[str, list[BaseMessage]]:
        return {"messages": [model_with_tools.invoke(state["messages"])]}

    def after_model(state: MessagesState) -> str:
        last = state["messages"][-1]
        return "tools" if isinstance(last, AIMessage) and last.tool_calls else "remind"

    def remind(state: MessagesState) -> dict[str, list[BaseMessage]]:
        logger.warning("the model answered without calling a tool, so it is asked again")
        return {"messages": [HumanMessage("Don't answer with text: call one of your tools.")]}

    def after_tools(state: MessagesState) -> str:
        return END if done(state["messages"]) else "agent"

    graph = StateGraph(MessagesState)
    graph.add_node("agent", call_model)
    graph.add_node("tools", ToolNode(tools))
    graph.add_node("remind", remind)
    graph.add_edge(START, "agent")
    graph.add_conditional_edges("agent", after_model, ["tools", "remind"])
    graph.add_edge("remind", "agent")
    graph.add_conditional_edges("tools", after_tools, ["agent", END])
    return graph.compile(checkpointer=checkpointer)


def run_tool_agent(
    model: BaseChatModel,
    tools: list[BaseTool],
    system_prompt: str,
    context: str,
    done: Callable[[Sequence[BaseMessage]], bool],
    max_steps: int,
) -> list[BaseMessage]:
    """Run the model with `tools` until `done(messages)` is true after a tool call, and return the messages.

    If it is still not done after `max_steps` steps (one model turn or one tool turn each), the loop stops and
    only the opening messages are returned: tools keep their results elsewhere (the offer pool, the saved
    answers), so the caller can still go on.
    A tool can pause the whole graph with `interrupt()`; the loop resumes where it stopped.
    """
    agent = build_tool_agent(model, tools, done)

    messages: list[AnyMessage] = [SystemMessage(system_prompt), HumanMessage(context)]
    try:
        config: RunnableConfig = {"recursion_limit": max_steps}
        result = agent.invoke({"messages": messages}, config)
        final: list[BaseMessage] = result["messages"]
        return final
    except GraphRecursionError:
        logger.warning("the agent did not finish in %d steps, so it was stopped", max_steps)
        return list(messages)
