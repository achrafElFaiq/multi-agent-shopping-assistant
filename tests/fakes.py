"""Fakes for tests: no network, no API key, scripted answers."""

from collections.abc import Callable, Sequence
from decimal import Decimal
from typing import Any

from langchain_core.language_models import LanguageModelInput
from langchain_core.language_models.fake_chat_models import GenericFakeChatModel
from langchain_core.messages import AIMessage
from langchain_core.runnables import Runnable, RunnableConfig
from langchain_core.tools import BaseTool

from backend.core.models.products import CatalogPage, Offer, Seller
from backend.core.models.search import SearchRequest, SearchResults


class FakeChatModel(GenericFakeChatModel):
    """Replies with `messages` in order and records each conversation in `seen_inputs`."""

    seen_inputs: list[Any] = []
    tool_choices: list[str | None] = []

    def invoke(
        self,
        input: LanguageModelInput,
        config: RunnableConfig | None = None,
        *,
        stop: list[str] | None = None,
        **kwargs: Any,
    ) -> AIMessage:
        self.seen_inputs.append(list(input) if isinstance(input, list) else input)  # a snapshot, not a reference
        return super().invoke(input, config, stop=stop, **kwargs)

    def bind_tools(
        self,
        tools: Sequence[dict[str, Any] | type | Callable[..., Any] | BaseTool],
        *,
        tool_choice: str | None = None,
        **kwargs: Any,
    ) -> Runnable[LanguageModelInput, AIMessage]:
        self.tool_choices.append(tool_choice)
        return self


class FakeCatalog:
    """A product catalog that answers with the given pages, one per search or next page, and records the searches."""

    def __init__(self, *pages: CatalogPage) -> None:
        self.pages = list(pages)
        self.searches: list[tuple[str, dict[str, list[str]], str | None]] = []

    def search(
        self,
        query: str,
        country: str,
        currency: str,
        max_price: Decimal | None,
        attributes: dict[str, list[str]],
        page: str | None = None,
    ) -> CatalogPage:
        self.searches.append((query, attributes, page))
        return self.pages.pop(0) if self.pages else CatalogPage(offers=[], messages=[], next_page=None)


def make_offer(store: str = "Pro:Direct", price: str = "99.90", **changes: object) -> Offer:
    """A buyable offer: in stock, new, in EUR."""
    offer = Offer(
        product_id=f"gid://shopify/p/{store}-{price}",
        title="Nike Pegasus 42",
        seller=Seller(shopify_id=f"gid://shopify/Shop/{store}", name=store, url=f"https://{store.lower()}.example"),
        price=Decimal(price),
        currency="EUR",
        condition=["new"],
        available=True,
        options={"Size": "43"},
        url=f"https://{store.lower()}.example/products/pegasus-42",
        checkout_url=f"https://{store.lower()}.example/cart/1:1",
    )
    return offer.model_copy(update=changes)


def catalog_with(*offers: Offer) -> FakeCatalog:
    """A catalog whose first search finds these offers."""
    return FakeCatalog(CatalogPage(offers=list(offers), messages=[], next_page=None))


class FakeSearcher:
    """A search agent that answers each request with the next given offers, and records the requests."""

    def __init__(self, *answers: list[Offer]) -> None:
        self.answers = list(answers)
        self.requests: list[SearchRequest] = []

    def search(self, request: SearchRequest) -> SearchResults:
        self.requests.append(request)
        offers = self.answers.pop(0) if self.answers else []
        return SearchResults(queries=[request.brief], offers=offers)
