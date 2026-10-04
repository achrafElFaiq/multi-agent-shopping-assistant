"""Chapter 1: User preferences.

An LLM agent reads the user's profile and past recommendations through tools,
then fills in UserPreferences for the category of the request.
"""

from typing import Any, Protocol

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.utils.function_calling import convert_to_openai_tool
from pydantic import ValidationError

from backend.agents.shopping.schemas import UserPreferences
from backend.agents.shopping.state import ShoppingState
from backend.agents.shopping.tools.user_preferences import make_preference_tools
from backend.core.ports.preferences import PreferencesRepository

SYSTEM_PROMPT = """You describe a shopper's preferences for one shopping request.

1. Decide the product category of the request, for example "shoes".
2. Call get_profile, and get_recommendations for that category.
3. From what the tools return, work out what this user likes and what to avoid in that category.
   Rejected recommendations and their reasons matter most. Recent answers matter more than old ones.
4. Call submit_preferences with the result. This ends your work.

Only use facts the tools returned. If there is little or no data, say so in the summary."""

MAX_TOOL_ROUNDS = 5  # stops a model that keeps calling tools forever
SUBMIT_TOOL_NAME = "submit_preferences"


def make_submit_tool() -> dict[str, Any]:
    """A tool whose arguments are the UserPreferences fields: calling it is how the model gives its final answer.

    Without it, the model needs one more call just to say it is done, then another to fill in the answer.
    `user_id` is left out: the code sets it, never the model.
    """
    tool = convert_to_openai_tool(UserPreferences)
    function = tool["function"]
    function["name"] = SUBMIT_TOOL_NAME
    function["description"] = "Give the user's preferences for this request, once you know enough."
    parameters = function["parameters"]
    del parameters["properties"]["user_id"]
    # Every field is required, so the model must decide on each one (e.g. `avoid`) instead of skipping it.
    # Fields that may be unknown (size, budget) still accept null.
    parameters["required"] = list(parameters["properties"])
    return tool


class UserPreferencesNode(Protocol):
    def __call__(self, state: ShoppingState) -> dict[str, UserPreferences]: ...


def make_user_preferences_node(model: BaseChatModel, repository: PreferencesRepository) -> UserPreferencesNode:
    def build_user_preferences(state: ShoppingState) -> dict[str, UserPreferences]:
        request = state["request"]
        tools = {t.name: t for t in make_preference_tools(repository, request.user_id)}
        # "required": the model must call a tool every turn, so it ends by calling submit_preferences
        # instead of sending an empty reply first (which would cost one more call).
        model_with_tools = model.bind_tools([*tools.values(), make_submit_tool()], tool_choice="required")
        messages: list[BaseMessage] = [SystemMessage(SYSTEM_PROMPT), HumanMessage(request.query)]

        # Agent loop: the model asks for tools, we run them and send back the results,
        # until it calls submit_preferences with its answer.
        for _ in range(MAX_TOOL_ROUNDS):
            reply = model_with_tools.invoke(messages)
            if not isinstance(reply, AIMessage):
                raise TypeError(f"Expected an AIMessage from the model, got {type(reply).__name__}")
            messages.append(reply)
            if not reply.tool_calls:
                break
            for call in reply.tool_calls:
                if call["name"] == SUBMIT_TOOL_NAME:
                    try:
                        # The model never decides whose preferences these are.
                        answer = UserPreferences.model_validate({**call["args"], "user_id": request.user_id})
                    except ValidationError as error:
                        messages.append(ToolMessage(f"Invalid preferences: {error}", tool_call_id=call["id"] or ""))
                        continue
                    return {"preferences": answer}
                tool = tools.get(call["name"])
                if tool is None:
                    messages.append(ToolMessage(f"Unknown tool {call['name']!r}", tool_call_id=call["id"] or ""))
                else:
                    messages.append(tool.invoke(call))

        # Fallback when the model never called submit_preferences: ask once more, in the exact UserPreferences shape.
        fallback = model.with_structured_output(UserPreferences).invoke(
            [*messages, HumanMessage("Now fill in this user's preferences for the request.")]
        )
        if not isinstance(fallback, UserPreferences):
            raise TypeError(f"Expected UserPreferences from the model, got {type(fallback).__name__}")
        # The model never decides whose preferences these are.
        return {"preferences": fallback.model_copy(update={"user_id": request.user_id})}

    return build_user_preferences
