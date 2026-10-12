"""PreferencesAgent over A2A: the preferences agent runs as its own A2A server (backend/agents/user_preferences_agent).

One A2A conversation per request, in steps (see its server): the request first, then one answer per question,
until the preferences come back. Each question goes to `ask_user`, which returns the user's answer.
"""

import asyncio
import logging
from collections.abc import Callable
from typing import Any

import httpx
from a2a.client import ClientConfig, create_client
from a2a.helpers import get_data_parts, new_data_message
from a2a.types import Role, SendMessageRequest

from backend.core.models.preferences import UserPreferences
from backend.core.models.shopping import ShoppingRequest

logger = logging.getLogger(__name__)


class A2APreferencesAgent:
    def __init__(self, url: str, http_client: httpx.AsyncClient | None = None) -> None:
        self._url = url  # where the agent's Agent Card is: {url}/.well-known/agent-card.json
        self._http_client = http_client

    def get_preferences(
        self, request: ShoppingRequest, conversation_id: str, ask_user: Callable[[dict[str, Any]], Any]
    ) -> UserPreferences:
        logger.info("A2A → preferences: %s", request.query)
        reply = self._send(conversation_id, {"step": 0, "request": request.model_dump(mode="json")})
        step = 0
        while "question" in reply:
            logger.info("A2A ← preferences asks: %s", reply["question"]["question"])
            answer = ask_user(reply["question"])
            step += 1
            reply = self._send(conversation_id, {"step": step, "answer": str(answer)})
        preferences = UserPreferences.model_validate(reply["preferences"])
        logger.info("A2A ← preferences: %s", preferences.summary)
        return preferences

    def _send(self, conversation_id: str, data: dict[str, Any]) -> dict[str, Any]:
        return asyncio.run(self._send_async(conversation_id, data))

    async def _send_async(self, conversation_id: str, data: dict[str, Any]) -> dict[str, Any]:
        http = self._http_client or httpx.AsyncClient(timeout=120)
        client = await create_client(self._url, ClientConfig(streaming=False, httpx_client=http))
        message = new_data_message(data, role=Role.ROLE_USER, context_id=conversation_id)
        async for response in client.send_message(SendMessageRequest(message=message)):
            if response.HasField("message"):
                [reply] = get_data_parts(response.message.parts)
                return dict(reply)
        raise RuntimeError("The preferences agent answered without a message.")
