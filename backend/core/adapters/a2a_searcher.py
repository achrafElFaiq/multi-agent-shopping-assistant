"""ProductSearcher over A2A: the search agent runs as its own A2A server (backend/agents/searcher).

Each search is one A2A message from the recommender ("user") to the searcher ("agent"), whose single data part
is a SearchRequest; the searcher answers with one message whose data part is SearchResults.
"""

import asyncio
import logging

import httpx
from a2a.client import ClientConfig, create_client
from a2a.helpers import get_data_parts, new_data_message
from a2a.types import Role, SendMessageRequest

from backend.core.models.search import SearchRequest, SearchResults

logger = logging.getLogger(__name__)


class A2ASearcher:
    def __init__(self, url: str, http_client: httpx.AsyncClient | None = None) -> None:
        self._url = url  # where the searcher's Agent Card is: {url}/.well-known/agent-card.json
        self._http_client = http_client

    def search(self, request: SearchRequest) -> SearchResults:
        return asyncio.run(self._search(request))

    async def _search(self, request: SearchRequest) -> SearchResults:
        logger.info("A2A → searcher: %s", request.brief)
        http = self._http_client or httpx.AsyncClient(timeout=120)
        client = await create_client(self._url, ClientConfig(streaming=False, httpx_client=http))
        message = new_data_message(request.model_dump(mode="json"), role=Role.ROLE_USER)
        async for response in client.send_message(SendMessageRequest(message=message)):
            if response.HasField("message"):
                [data] = get_data_parts(response.message.parts)
                results = SearchResults.model_validate(data)
                logger.info("A2A ← searcher: %d offers", len(results.offers))
                return results
        raise RuntimeError("The searcher answered without a message.")
