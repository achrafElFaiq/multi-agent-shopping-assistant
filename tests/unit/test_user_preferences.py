from decimal import Decimal
from itertools import repeat

from langchain_core.messages import AIMessage, ToolMessage
from langchain_core.messages.tool import ToolCall

from backend.agents.shopping.nodes.user_preferences import MAX_TOOL_ROUNDS, make_user_preferences_node
from backend.agents.shopping.schemas import ShoppingRequest, UserPreferences
from backend.agents.shopping.state import ShoppingState
from backend.agents.shopping.tools.user_preferences import make_preference_tools
from backend.core.adapters.mock_preferences import MockPreferencesRepository
from tests.fakes import FakeChatModel

ANSWER = UserPreferences(user_id="user-1", category="shoes", summary="Likes Asics, avoids Nike.")


def make_state(user_id: str = "user-1") -> ShoppingState:
    return {"request": ShoppingRequest(user_id=user_id, query="running shoes", spending_limit=Decimal("120"))}


def call(name: str, call_id: str, **args: str) -> ToolCall:
    return ToolCall(name=name, args=args, id=call_id)


def tool_results(model: FakeChatModel) -> list[ToolMessage]:
    return [m for m in model.last_input if isinstance(m, ToolMessage)]


def test_tool_results_reach_the_model() -> None:
    model = FakeChatModel(
        messages=iter(
            [
                AIMessage(
                    "", tool_calls=[call("get_profile", "1"), call("get_recommendations", "2", category="shoes")]
                ),
                AIMessage("Done."),
            ]
        ),
        structured_answer=ANSWER,
    )
    node = make_user_preferences_node(model, MockPreferencesRepository())

    result = node(make_state())

    profile, recommendations = tool_results(model)
    assert "Asics" in str(profile.content)
    assert "too narrow" in str(recommendations.content)
    assert result["preferences"].summary == ANSWER.summary


def test_user_id_comes_from_the_request_not_the_model() -> None:
    model = FakeChatModel(
        messages=iter([AIMessage("Done.")]),
        structured_answer=ANSWER.model_copy(update={"user_id": "someone-else"}),
    )
    node = make_user_preferences_node(model, MockPreferencesRepository())

    result = node(make_state(user_id="user-1"))

    assert result["preferences"].user_id == "user-1"


def test_unknown_tool_is_reported_to_the_model() -> None:
    model = FakeChatModel(
        messages=iter([AIMessage("", tool_calls=[call("delete_user", "1")]), AIMessage("Done.")]),
        structured_answer=ANSWER,
    )
    node = make_user_preferences_node(model, MockPreferencesRepository())

    node(make_state())

    [reply] = tool_results(model)
    assert "Unknown tool 'delete_user'" in str(reply.content)


def test_loop_stops_when_the_model_never_stops_calling_tools() -> None:
    model = FakeChatModel(
        messages=repeat(AIMessage("", tool_calls=[call("get_profile", "1")])),
        structured_answer=ANSWER,
    )
    node = make_user_preferences_node(model, MockPreferencesRepository())

    result = node(make_state())

    assert len(tool_results(model)) == MAX_TOOL_ROUNDS
    assert result["preferences"] == ANSWER


def test_tools_only_read_the_request_user() -> None:
    get_profile, get_recommendations = make_preference_tools(MockPreferencesRepository(), user_id="unknown-user")

    assert get_profile.invoke({}) == {"sizes": {}, "budgets": {}, "likes": [], "dislikes": []}
    assert get_recommendations.invoke({"category": "shoes"}) == []
    assert "user_id" not in get_recommendations.args


def test_recommendations_are_filtered_by_category() -> None:
    _, get_recommendations = make_preference_tools(MockPreferencesRepository(), user_id="user-1")

    recommendations = get_recommendations.invoke({"category": "t-shirt"})

    assert [r["product_name"] for r in recommendations] == ["Dri-FIT Tee"]
