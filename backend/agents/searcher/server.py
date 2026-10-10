"""The search agent as an A2A server.

It publishes its Agent Card at /.well-known/agent-card.json and answers A2A messages at /:
each message carries a SearchRequest as its data part, and the answer carries SearchResults.
"""

import asyncio
import logging

from a2a.helpers import get_data_parts, new_data_message
from a2a.server.agent_execution import AgentExecutor, RequestContext
from a2a.server.events import EventQueue
from a2a.server.request_handlers import DefaultRequestHandler
from a2a.server.routes import create_agent_card_routes, create_jsonrpc_routes
from a2a.server.tasks import InMemoryTaskStore
from a2a.types import AgentCapabilities, AgentCard, AgentInterface, AgentSkill, Role
from langchain_core.language_models import BaseChatModel
from starlette.applications import Starlette

from backend.agents.searcher.agent import find_offers
from backend.core.models.search import SearchRequest
from backend.core.ports.catalog import ProductCatalog

logger = logging.getLogger(__name__)


class SearchAgentExecutor(AgentExecutor):
    def __init__(self, model: BaseChatModel, catalog: ProductCatalog) -> None:
        self.model = model
        self.catalog = catalog

    async def execute(self, context: RequestContext, event_queue: EventQueue) -> None:
        """Answer one message: read the SearchRequest, run the search agent, send back SearchResults."""
        if context.message is None:
            raise ValueError("A search needs a message.")
        [data] = get_data_parts(context.message.parts)
        request = SearchRequest.model_validate(data)
        logger.info("brief: %s", request.brief)
        results = await asyncio.to_thread(find_offers, self.model, self.catalog, request)  # the agent is not async
        logger.info("answering with %d offers after %d searches", len(results.offers), len(results.queries))
        answer = new_data_message(results.model_dump(mode="json"), role=Role.ROLE_AGENT, context_id=context.context_id)
        await event_queue.enqueue_event(answer)

    async def cancel(self, context: RequestContext, event_queue: EventQueue) -> None:
        raise NotImplementedError("Searches can't be cancelled.")


def agent_card(url: str) -> AgentCard:
    """Who the searcher is and what it can do, for other agents."""
    return AgentCard(
        name="Product searcher",
        description="Finds products the shopper can buy across all Shopify stores, from a brief.",
        version="0.1.0",
        supported_interfaces=[AgentInterface(url=url, protocol_binding="JSONRPC", protocol_version="1.0")],
        capabilities=AgentCapabilities(streaming=False),
        default_input_modes=["application/json"],
        default_output_modes=["application/json"],
        skills=[
            AgentSkill(
                id="search_products",
                name="Search products",
                description=(
                    "Takes a SearchRequest (brief, country, currency, max_price, seen) and returns SearchResults: "
                    "new, in-stock offers that ship to the country, within the price limit."
                ),
                tags=["shopping", "search", "shopify"],
                examples=["Running shoes for a marathon, Asics, size 43, men. Refused: too flashy."],
            )
        ],
    )


def create_app(model: BaseChatModel, catalog: ProductCatalog, url: str) -> Starlette:
    card = agent_card(url)
    handler = DefaultRequestHandler(
        agent_executor=SearchAgentExecutor(model, catalog), task_store=InMemoryTaskStore(), agent_card=card
    )
    return Starlette(routes=create_agent_card_routes(card) + create_jsonrpc_routes(handler, rpc_url="/"))
