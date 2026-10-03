"""Chapter 1: User preferences.

An LLM agent reads the user's profile and past recommendations through tools,
then fills in UserPreferences for the category of the request.
"""

from typing import Protocol

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage, ToolMessage

from backend.agents.shopping.schemas import UserPreferences
from backend.agents.shopping.state import ShoppingState
from backend.agents.shopping.tools.user_preferences import make_preference_tools
from backend.core.ports.preferences import PreferencesRepository

SYSTEM_PROMPT = """You describe a shopper's preferences for one shopping request.

1. Decide the product category of the request, for example "shoes".
2. Call get_profile, and get_recommendations for that category.
3. From what the tools return, work out what this user likes and what to avoid in that category.
   Rejected recommendations and their reasons matter most. Recent answers matter more than old ones.

Only use facts the tools returned. If there is little or no data, say so in the summary."""

MAX_TOOL_ROUNDS = 5  # stops a model that keeps calling tools forever


class UserPreferencesNode(Protocol):
    def __call__(self, state: ShoppingState) -> dict[str, UserPreferences]: ...


def make_user_preferences_node(model: BaseChatModel, repository: PreferencesRepository) -> UserPreferencesNode:
    def build_user_preferences(state: ShoppingState) -> dict[str, UserPreferences]:
        request = state["request"]
        tools = {t.name: t for t in make_preference_tools(repository, request.user_id)}
        model_with_tools = model.bind_tools(list(tools.values()))
        messages: list[BaseMessage] = [SystemMessage(SYSTEM_PROMPT), HumanMessage(request.query)]

        # Agent loop: the model asks for tools, we run them and send back the results, until it stops asking.
        for _ in range(MAX_TOOL_ROUNDS):
            reply = model_with_tools.invoke(messages)
            if not isinstance(reply, AIMessage):
                raise TypeError(f"Expected an AIMessage from the model, got {type(reply).__name__}")
            messages.append(reply)
            if not reply.tool_calls:
                break
            for call in reply.tool_calls:
                tool = tools.get(call["name"])
                if tool is None:
                    messages.append(ToolMessage(f"Unknown tool {call['name']!r}", tool_call_id=call["id"] or ""))
                else:
                    messages.append(tool.invoke(call))

        # Final answer: the same conversation, answered in the exact UserPreferences shape.
        answer = model.with_structured_output(UserPreferences).invoke(
            [*messages, HumanMessage("Now fill in this user's preferences for the request.")]
        )
        if not isinstance(answer, UserPreferences):
            raise TypeError(f"Expected UserPreferences from the model, got {type(answer).__name__}")
        # The model never decides whose preferences these are.
        return {"preferences": answer.model_copy(update={"user_id": request.user_id})}

    return build_user_preferences
