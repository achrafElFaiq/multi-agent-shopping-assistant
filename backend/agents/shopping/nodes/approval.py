"""Chapter 3: Decision & Human Approval.

Pauses the graph with `interrupt()` and waits for the user to approve or cancel.
"""

from langgraph.types import interrupt
from pydantic import ValidationError

from backend.agents.shopping.schemas import ApprovalDecision
from backend.agents.shopping.state import ShoppingState


def ask_approval(state: ShoppingState) -> dict[str, ApprovalDecision]:
    question = state["recommendation"].model_dump(mode="json")
    # Everything above the loop runs again when the graph resumes.
    while True:
        answer = interrupt(question)
        try:
            return {"decision": ApprovalDecision.model_validate(answer)}
        except ValidationError:
            # Never crash on a bad answer (that would leave the order stuck): pause again and ask again.
            question = {**question, "error": "Please answer approve or cancel."}
