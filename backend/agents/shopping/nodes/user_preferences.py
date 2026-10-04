"""Chapter 1: User preferences.

An LLM agent reads the user's profile and past recommendations through tools,
then submits UserPreferences for the category of the request.
"""

from typing import Protocol

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.tools import BaseTool

from backend.agents.shopping.schemas import UserPreferences
from backend.agents.shopping.state import ShoppingState
from backend.agents.shopping.tools.user_preferences import make_preference_tools
from backend.core.ports.preferences import PreferencesRepository

SYSTEM_PROMPT = """You describe a shopper's preferences for one shopping request.

1. Call get_profile and get_recommendations.
2. Decide the product category of the request, using the category names you see in that data (for example "shoes").
3. From what the tools return, work out what this user likes and what to avoid in that category.
   Rejected recommendations and their reasons matter most. Recent answers matter more than old ones.
4. Call submit_preferences with the result. This ends your work.

The spending limit of the request is a hard maximum for this purchase. Budget, in your answer,
is what the user usually spends: if it is above the limit, say so in the summary.

Only use facts the tools returned. If there is little or no data, say so in the summary."""

MAX_TOOL_ROUNDS = 5  # stops a model that keeps calling tools forever


class UserPreferencesNode(Protocol):
    def __call__(self, state: ShoppingState) -> dict[str, UserPreferences]: ...


def make_user_preferences_node(model: BaseChatModel, repository: PreferencesRepository) -> UserPreferencesNode:
    def build_user_preferences(state: ShoppingState) -> dict[str, UserPreferences]:
        request = state["request"]
        tools = {t.name: t for t in make_preference_tools(repository, request.user_id)}
        # "required": the model must call a tool every turn, so it finishes by calling submit_preferences
        # instead of first sending an empty reply (which would cost one more call).
        model_with_tools = model.bind_tools(list(tools.values()), tool_choice="required")
        # The whole request, not just the query: the model needs the spending limit, the currency and the deadline.
        request_text = f"Shopping request:\n{request.model_dump_json(indent=2)}"
        messages: list[BaseMessage] = [SystemMessage(SYSTEM_PROMPT), HumanMessage(request_text)]

        # Agent loop: the model asks for tools, we run them and send back the results,
        # until a return_direct tool (submit_preferences) gives the final answer.
        for _ in range(MAX_TOOL_ROUNDS):
            reply = model_with_tools.invoke(messages)
            if not isinstance(reply, AIMessage):
                raise TypeError(f"Expected an AIMessage from the model, got {type(reply).__name__}")
            messages.append(reply)
            if not reply.tool_calls:
                break
            answer = run_tool_calls(tools, reply, messages)
            if answer is not None:
                # The model never decides whose preferences these are: user_id always comes from the request.
                return {"preferences": answer.model_copy(update={"user_id": request.user_id})}

        # Fallback when the model never submitted: ask once more, in the exact UserPreferences shape.
        fallback = model.with_structured_output(UserPreferences).invoke(
            [*messages, HumanMessage("Now fill in this user's preferences for the request.")]
        )
        if not isinstance(fallback, UserPreferences):
            raise TypeError(f"Expected UserPreferences from the model, got {type(fallback).__name__}")
        return {"preferences": fallback.model_copy(update={"user_id": request.user_id})}

    return build_user_preferences


def run_tool_calls(tools: dict[str, BaseTool], reply: AIMessage, messages: list[BaseMessage]) -> UserPreferences | None:
    """Runs the tools the model asked for and adds their results to the conversation.

    Returns the final answer when a return_direct tool (submit_preferences) gave one.
    """
    for call in reply.tool_calls:
        tool = tools.get(call["name"])
        if tool is None:
            messages.append(ToolMessage(f"Unknown tool {call['name']!r}", tool_call_id=call["id"] or ""))
            continue
        result = tool.invoke(call)
        messages.append(result)
        # The artifact is missing when the submitted preferences were invalid: the model gets the error instead.
        if tool.return_direct and isinstance(result.artifact, UserPreferences):
            return result.artifact
    return None
