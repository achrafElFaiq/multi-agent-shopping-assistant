"""Tools for the recommendation agent: ask the search agent for offers (over A2A), present them one at a time,
wait for the shopper's answer and save it.

The offers shown (price, store, link) come from the offers themselves, never from the model: the model only
chooses which offer and writes the pitch. Offers and answers are kept in the tool results (artifacts), so they
survive the pauses while the graph waits for the shopper.
"""

from collections.abc import Sequence
from datetime import UTC, datetime
from typing import Annotated, Any

from langchain_core.messages import BaseMessage, ToolMessage
from langchain_core.tools import BaseTool, tool
from langgraph.prebuilt import InjectedState
from langgraph.types import interrupt

from backend.agents.shopping.schemas import ShoppingRequest
from backend.core.models.products import Offer
from backend.core.models.recommendations import RecommendationRecord
from backend.core.models.search import SearchRequest
from backend.core.ports.recommendations import RecommendationRepository
from backend.core.ports.searcher import ProductSearcher
from backend.core.services.product_finder import OfferPool

MAX_SEARCHES = 3  # requests to the search agent
MAX_PRESENTED = 5  # offers shown to the shopper
YES = {"yes", "y", "oui", "o"}
NO = {"no", "n", "non"}


def make_recommendation_tools(
    searcher: ProductSearcher,
    repository: RecommendationRepository,
    request: ShoppingRequest,
    country: str,
    category: str,
) -> list[BaseTool]:
    @tool(response_format="content_and_artifact")
    def ask_searcher(brief: str, state: Annotated[dict[str, Any], InjectedState]) -> tuple[str, dict[str, Any] | None]:
        """Ask the search agent for offers. Call this tool alone.

        brief: what to find, for the search agent: the product and what matters to the shopper (brand, size,
            colour, likes, requirements). After a refusal, say what to find instead and why, e.g. "Find better:
            the shopper refused the Nike Pegasus 42 at Lovell Sports because: too flashy".
        """
        messages = state["messages"]
        if len(searches(messages)) >= MAX_SEARCHES:
            return f"{MAX_SEARCHES} searches were made already: present an offer found or call no_more_offers.", None
        known = offers_found(messages)
        search = SearchRequest(
            brief=brief,
            country=country,
            currency=request.currency,
            max_price=request.spending_limit,
            seen=[offer.product_id for offer in known],
        )
        try:
            results = searcher.search(search)
        except Exception as error:
            return f"The search agent failed ({error}). Try again, or call no_more_offers.", None

        # Never trust another agent with the shopper's limits: check its offers again.
        pool = OfferPool(country, request.currency, request.spending_limit, [offer.product_id for offer in known])
        pool.add(results.offers)
        found = {"brief": brief, "offers": [offer.model_dump(mode="json") for offer in pool.offers]}
        if not pool.offers:
            return "The search agent found no new offer. Try another brief, or call no_more_offers.", found
        lines = [describe(number, offer) for number, offer in enumerate(pool.offers, start=len(known) + 1)]
        return f"{len(pool.offers)} new offers:\n" + "\n".join(lines), found

    @tool(response_format="content_and_artifact")
    def present_offer(
        offer_number: int, pitch: str, state: Annotated[dict[str, Any], InjectedState]
    ) -> tuple[str, dict[str, Any] | None]:
        """Show one offer to the shopper with your pitch and wait for their answer. Call this tool alone.

        offer_number: the offer's number in the lists the search agent sent.
        pitch: one or two sentences, in the shopper's language, on why this offer fits them. The system shows the
            title, store, price, options and link: don't repeat them.
        """
        messages = state["messages"]
        offers, shown = offers_found(messages), answers(messages)
        if not 1 <= offer_number <= len(offers):
            return f"There is no offer {offer_number}: choose a number from 1 to {len(offers)}.", None
        if any(answer["offer_number"] == offer_number for answer in shown):
            return f"Offer {offer_number} was already presented: choose another one.", None
        if len(shown) >= MAX_PRESENTED:
            return f"{MAX_PRESENTED} offers were presented already: call no_more_offers.", None

        offer = offers[offer_number - 1]
        accepted = ask_yes_or_no(offer, pitch, len(shown) + 1)
        reason = "" if accepted else str(interrupt({"type": "reason", "question": "Why not this one?"})).strip()

        # The action: save the answer (in memory for now).
        repository.save(
            RecommendationRecord(
                query=request.query,
                category=category,
                offer=offer,
                pitch=pitch,
                accepted=accepted,
                reason=reason,
                decided_at=datetime.now(UTC),
            )
        )
        answer = {"offer_number": offer_number, "pitch": pitch, "accepted": accepted, "reason": reason}
        if accepted:
            return "The shopper accepted this offer.", answer
        said = f" Their reason: {reason}" if reason else " They gave no reason."
        return (
            f"The shopper said no.{said} Present a better offer already found, or ask the search agent to find "
            "better, telling it why this one was refused. Or call no_more_offers.",
            answer,
        )

    @tool
    def no_more_offers() -> str:
        """Call this alone when no offer fits the shopper and searching again won't help."""
        return "No more offers."

    return [ask_searcher, present_offer, no_more_offers]


def ask_yes_or_no(offer: Offer, pitch: str, number: int) -> bool:
    """Pause the graph to show the offer, until the shopper answers yes or no."""
    question: dict[str, Any] = {
        "type": "recommendation",
        "number": number,
        "pitch": pitch,
        "offer": {
            "title": offer.title,
            "store": offer.seller.name,
            "price": str(offer.price),
            "currency": offer.currency,
            "options": offer.options,
            "rating": offer.rating,
            "url": offer.url,
        },
    }
    # Everything above runs again when the graph resumes; only the answers are remembered.
    while True:
        answer = str(interrupt(question)).strip().lower()
        if answer in YES:
            return True
        if answer in NO:
            return False
        question = {**question, "error": "Please answer yes or no."}


def describe(number: int, offer: Offer) -> str:
    """One line per offer for the model: what it needs to judge the fit."""
    options = ", ".join(f"{name} {value}" for name, value in offer.options.items()) or "-"
    rating = f"{offer.rating}/5 ({offer.rating_count} reviews)" if offer.rating else "no rating"
    return (
        f"{number}. {offer.title} | {offer.seller.name} | {offer.price} {offer.currency} | {options} | {rating} | "
        f"{offer.description}"
    )


def results_of(messages: Sequence[BaseMessage], tool_name: str) -> list[dict[str, Any]]:
    return [
        message.artifact
        for message in messages
        if isinstance(message, ToolMessage) and message.name == tool_name and isinstance(message.artifact, dict)
    ]


def searches(messages: Sequence[BaseMessage]) -> list[dict[str, Any]]:
    """The answers of the search agent so far."""
    return results_of(messages, "ask_searcher")


def offers_found(messages: Sequence[BaseMessage]) -> list[Offer]:
    """Every offer found so far, numbered from 1 in the order they came."""
    return [Offer.model_validate(offer) for found in searches(messages) for offer in found["offers"]]


def answers(messages: Sequence[BaseMessage]) -> list[dict[str, Any]]:
    """The shopper's answers so far, in order."""
    return results_of(messages, "present_offer")
