"""The user preferences agent as an A2A server.

It publishes its Agent Card at /.well-known/agent-card.json and answers A2A messages at /.
One conversation (A2A context_id) per shopping request, in steps:
    step 0:  {"step": 0, "request": ShoppingRequest}  ->  {"question": {...}} or {"preferences": UserPreferences}
    step n:  {"step": n, "answer": "43"}               ->  the next question, or the preferences
When the agent asks the user something (ask_user), it pauses; the answer of the next step resumes it.
A step already answered is answered again from memory, without running the agent: the master may send it twice.
"""

import asyncio
import logging
from typing import TYPE_CHECKING, Any, NotRequired, TypedDict

from a2a.helpers import get_data_parts, new_data_message
from a2a.server.agent_execution import AgentExecutor, RequestContext
from a2a.server.events import EventQueue
from a2a.server.request_handlers import DefaultRequestHandler
from a2a.server.routes import create_agent_card_routes, create_jsonrpc_routes
from a2a.server.tasks import InMemoryTaskStore
from a2a.types import AgentCapabilities, AgentCard, AgentInterface, AgentSkill, Role
from langchain_core.language_models import BaseChatModel
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer
from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph
from langgraph.types import Command
from starlette.applications import Starlette

from backend.agents.user_preferences_agent.agent import UserPreferencesAgent, make_user_preferences_agent
from backend.core.models.preferences import UserPreferences
from backend.core.models.shopping import ShoppingRequest
from backend.core.ports.preferences import PreferencesRepository
from backend.core.ports.recommendations import RecommendationRepository

if TYPE_CHECKING:
    from langchain_core.runnables import RunnableConfig

logger = logging.getLogger(__name__)


class Conversation(TypedDict):
    request: ShoppingRequest
    preferences: NotRequired[UserPreferences]


def pausable(agent: UserPreferencesAgent) -> CompiledStateGraph[Conversation, None, Conversation, Conversation]:
    """The agent inside a graph with a checkpointer, so its questions (interrupt) pause it until the answer."""
    graph = StateGraph(Conversation)
    graph.add_node("preferences", lambda conversation: {"preferences": agent(conversation["request"])})
    graph.add_edge(START, "preferences")
    graph.add_edge("preferences", END)
    serde = JsonPlusSerializer(allowed_msgpack_modules=[ShoppingRequest, UserPreferences])
    return graph.compile(checkpointer=InMemorySaver(serde=serde))


class PreferencesAgentExecutor(AgentExecutor):
    def __init__(
        self, model: BaseChatModel, preferences: PreferencesRepository, recommendations: RecommendationRepository
    ) -> None:
        self.graph = pausable(make_user_preferences_agent(model, preferences, recommendations))
        self.replies: dict[str, list[dict[str, Any]]] = {}  # per conversation, the reply sent at each step

    async def execute(self, context: RequestContext, event_queue: EventQueue) -> None:
        """Answer one step of a conversation: a question for the user, or the preferences."""
        if context.message is None or not context.context_id:
            raise ValueError("Preferences need a message, in a conversation (context_id).")
        [data] = get_data_parts(context.message.parts)
        replies = self.replies.setdefault(context.context_id, [])
        step = int(data["step"])  # A2A data travels as JSON-like protobuf: numbers come back as floats
        if step < len(replies):
            reply = replies[step]  # sent again: answer the same, don't run the agent twice
        else:  # the agent is not async
            reply = await asyncio.to_thread(self.run_step, context.context_id, step, data)
            replies.append(reply)
        message = new_data_message(reply, role=Role.ROLE_AGENT, context_id=context.context_id)
        await event_queue.enqueue_event(message)

    def run_step(self, conversation_id: str, step: int, data: dict[str, Any]) -> dict[str, Any]:
        config: RunnableConfig = {"configurable": {"thread_id": conversation_id}}
        if step == 0:
            request = ShoppingRequest.model_validate(data["request"])
            logger.info("request: %s", request.query)
            self.graph.invoke({"request": request}, config)
        else:
            logger.info("answer: %s", data["answer"])
            self.graph.invoke(Command(resume=data["answer"]), config)
        state = self.graph.get_state(config)
        if state.interrupts:
            question: dict[str, Any] = state.interrupts[0].value
            logger.info("asking the user: %s", question["question"])
            return {"question": question}
        preferences: UserPreferences = state.values["preferences"]
        logger.info("answering with category %r: %s", preferences.category, preferences.summary)
        return {"preferences": preferences.model_dump(mode="json")}

    async def cancel(self, context: RequestContext, event_queue: EventQueue) -> None:
        raise NotImplementedError("Preferences can't be cancelled.")


def agent_card(url: str) -> AgentCard:
    """Who the preferences agent is and what it can do, for other agents."""
    return AgentCard(
        name="User preferences",
        description="Knows the user: their stored preferences and past answers to offers, kept up to date.",
        version="0.1.0",
        supported_interfaces=[AgentInterface(url=url, protocol_binding="JSONRPC", protocol_version="1.0")],
        capabilities=AgentCapabilities(streaming=False),
        default_input_modes=["application/json"],
        default_output_modes=["application/json"],
        skills=[
            AgentSkill(
                id="get_preferences",
                name="Get preferences",
                description=(
                    "A conversation per request. Step 0 sends a ShoppingRequest (query, spending_limit, currency); "
                    "each reply is a question for the user, answered in the next step, or the UserPreferences: "
                    "category, attributes, likes, dislikes, requirements, usual budget and a summary."
                ),
                tags=["shopping", "preferences", "user"],
                examples=["running shoes, limit 120 EUR"],
            )
        ],
    )


def create_app(
    model: BaseChatModel, preferences: PreferencesRepository, recommendations: RecommendationRepository, url: str
) -> Starlette:
    card = agent_card(url)
    handler = DefaultRequestHandler(
        agent_executor=PreferencesAgentExecutor(model, preferences, recommendations),
        task_store=InMemoryTaskStore(),
        agent_card=card,
    )
    return Starlette(routes=create_agent_card_routes(card) + create_jsonrpc_routes(handler, rpc_url="/"))
