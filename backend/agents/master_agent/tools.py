"""Tools for the master agent: ask the preferences agent, ask the product finder, present offers to the user, pay.

The rules live here, in code: the limits sent to the product finder, its offers checked again, what is shown to
the user (price, store, link come from the offer, never from the model), and no payment without a yes.
Every result is kept in the tool messages (artifacts), so it survives the pauses while the user answers.
"""

from collections.abc import Sequence
from datetime import UTC, datetime
from typing import Annotated, Any
from uuid import uuid4

from langchain_core.messages import BaseMessage, ToolMessage
from langchain_core.tools import BaseTool, InjectedToolCallId, tool
from langgraph.prebuilt import InjectedState
from langgraph.types import interrupt

from backend.core.countries import country_code
from backend.core.models.payment import PaymentResult
from backend.core.models.preferences import UserPreferences
from backend.core.models.products import Offer
from backend.core.models.recommendations import RecommendationRecord
from backend.core.models.search import SearchRequest
from backend.core.models.shopping import ShoppingRequest
from backend.core.ports.preferences_agent import PreferencesAgent
from backend.core.ports.recommendations import RecommendationRepository
from backend.core.ports.searcher import ProductSearcher
from backend.core.services.product_finder import OfferPool

MAX_SEARCHES = 3  # requests to the product finder
MAX_PRESENTED = 5  # offers shown to the user
YES = {"yes", "y", "oui", "o"}
NO = {"no", "n", "non"}

State = Annotated[dict[str, Any], InjectedState]  # the agent's messages, given by the system, not the model


def make_master_tools(
    preferences_agent: PreferencesAgent,
    searcher: ProductSearcher,
    recommendations: RecommendationRepository,
    request: ShoppingRequest,
) -> list[BaseTool]:
    @tool(response_format="content_and_artifact")
    def ask_preferences(
        state: State, tool_call_id: Annotated[str, InjectedToolCallId]
    ) -> tuple[str, dict[str, Any] | None]:
        """Ask the preferences agent for the user's preferences for this request. Call this tool alone, first."""
        if preferences_of(state["messages"]):
            return "You already have the preferences.", None
        # Its questions pause the master (interrupt) until the user answers. When the master resumes, this tool
        # runs again with the same id, so the preferences agent knows it is the same conversation.
        preferences = preferences_agent.get_preferences(request, tool_call_id, interrupt)
        return f"Preferences: {preferences.model_dump_json()}", preferences.model_dump(mode="json")

    @tool(response_format="content_and_artifact")
    def find_products(brief: str, state: State) -> tuple[str, dict[str, Any] | None]:
        """Ask the product finder for offers. Call this tool alone.

        brief: what to find: the product and what matters to the user (brand, size, colour, likes, dislikes,
            requirements). After a refusal, say what to find instead and why, e.g. "Find better: the user refused
            the Nike Pegasus 42 at Lovell Sports because: too flashy".
        """
        messages = state["messages"]
        preferences = preferences_of(messages)
        if preferences is None:
            return "Call ask_preferences first.", None
        if len(results_of(messages, "find_products")) >= MAX_SEARCHES:
            return f"{MAX_SEARCHES} searches were made already: present an offer found or call give_up.", None

        country = country_code(str(preferences.attributes.get("country", "")))
        known = offers_found(messages)
        seen = [offer.product_id for offer in known]
        search = SearchRequest(
            brief=brief, country=country, currency=request.currency, max_price=request.spending_limit, seen=seen
        )
        try:
            results = searcher.search(search)
        except Exception as error:
            return f"The product finder failed ({error}). Try again, or call give_up.", None

        # Never trust another agent with the user's limits: check its offers again.
        pool = OfferPool(country, request.currency, request.spending_limit, seen)
        pool.add(results.offers)
        found = {"brief": brief, "offers": [offer.model_dump(mode="json") for offer in pool.offers]}
        if not pool.offers:
            return "The product finder found no new offer. Try another brief, or call give_up.", found
        lines = [describe(number, offer) for number, offer in enumerate(pool.offers, start=len(known) + 1)]
        return f"{len(pool.offers)} new offers:\n" + "\n".join(lines), found

    @tool(response_format="content_and_artifact")
    def present_offer(offer_number: int, pitch: str, state: State) -> tuple[str, dict[str, Any] | None]:
        """Show one offer to the user with your pitch and wait for their answer. Call this tool alone.

        offer_number: the offer's number in the lists the product finder sent.
        pitch: one or two sentences, in the user's language, on why this offer fits them. The system shows the
            title, store, price, options and link: don't repeat them.
        """
        messages = state["messages"]
        offers, shown = offers_found(messages), answers(messages)
        preferences = preferences_of(messages)
        if preferences is None:
            return "Call ask_preferences first.", None
        if not 1 <= offer_number <= len(offers):
            return f"There is no offer {offer_number}: choose a number from 1 to {len(offers)}.", None
        if any(answer["offer_number"] == offer_number for answer in shown):
            return f"Offer {offer_number} was already presented: choose another one.", None
        if len(shown) >= MAX_PRESENTED:
            return f"{MAX_PRESENTED} offers were presented already: call give_up.", None

        offer = offers[offer_number - 1]
        accepted = ask_yes_or_no(offer, pitch, len(shown) + 1)
        reason = "" if accepted else ask_reason()
        recommendations.save(
            RecommendationRecord(
                query=request.query,
                category=preferences.category,
                offer=offer,
                pitch=pitch,
                accepted=accepted,
                reason=reason,
                decided_at=datetime.now(UTC),
            )
        )
        answer = {"offer_number": offer_number, "accepted": accepted, "reason": reason}
        if accepted:
            return f"The user accepted offer {offer_number}: call pay with it.", answer
        return (
            f'The user said no. Their reason: "{reason}". Present a better offer already found, or ask the product '
            "finder to find better, telling it why this one was refused. Or call give_up.",
            answer,
        )

    @tool(response_format="content_and_artifact")
    def pay(offer_number: int, state: State) -> tuple[str, dict[str, Any] | None]:
        """Pay for the offer the user accepted. Call this tool alone."""
        messages = state["messages"]
        # Rule 1: never pay without the user's yes for this offer.
        if not any(answer["offer_number"] == offer_number and answer["accepted"] for answer in answers(messages)):
            return f"The user did not accept offer {offer_number}: payment refused.", None
        offer = offers_found(messages)[offer_number - 1]
        # Rule 2: never go over the spending limit.
        if offer.currency != request.currency or offer.price > request.spending_limit:
            payment = PaymentResult(status="blocked", amount=offer.price)
        else:  # Stub: fakes the checkout until real payment.
            payment = PaymentResult(status="paid", amount=offer.price, transaction_id=f"test_{uuid4().hex[:12]}")
        return f"Payment {payment.status}.", payment.model_dump(mode="json")

    @tool
    def give_up(reason: str) -> str:
        """End without buying: no offer fits the user and searching again won't help. Call this tool alone."""
        return "Ended without buying."

    return [ask_preferences, find_products, present_offer, pay, give_up]


def ask_yes_or_no(offer: Offer, pitch: str, number: int) -> bool:
    """Pause to show the offer, until the user answers yes or no."""
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
    # Everything above runs again when the agent resumes; only the answers are remembered.
    while True:
        answer = str(interrupt(question)).strip().lower()
        if answer in YES:
            return True
        if answer in NO:
            return False
        question = {**question, "error": "Please answer yes or no."}


def ask_reason() -> str:
    """Pause to ask why the user said no, until they give a reason: it is what history learns from."""
    question = {"type": "reason", "question": "Why not this one?"}
    while True:
        reason = str(interrupt(question)).strip()
        if reason:
            return reason
        question = {**question, "error": "Please say why: it helps find a better offer."}


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


def preferences_of(messages: Sequence[BaseMessage]) -> UserPreferences | None:
    found = results_of(messages, "ask_preferences")
    return UserPreferences.model_validate(found[0]) if found else None


def offers_found(messages: Sequence[BaseMessage]) -> list[Offer]:
    """Every offer found so far, numbered from 1 in the order they came."""
    return [Offer.model_validate(offer) for found in results_of(messages, "find_products") for offer in found["offers"]]


def answers(messages: Sequence[BaseMessage]) -> list[dict[str, Any]]:
    """The user's answers so far, in order."""
    return results_of(messages, "present_offer")


def payment_of(messages: Sequence[BaseMessage]) -> PaymentResult | None:
    paid = results_of(messages, "pay")
    return PaymentResult.model_validate(paid[-1]) if paid else None
