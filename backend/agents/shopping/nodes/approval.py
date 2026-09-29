"""Chapter 3: Decision & Human Approval.

Pauses the graph with `interrupt()` and waits for the user to approve or cancel.
"""

from langgraph.types import interrupt

from backend.agents.shopping.schemas import ApprovalDecision
from backend.agents.shopping.state import ShoppingState


def ask_approval(state: ShoppingState) -> dict[str, ApprovalDecision]:
    recommendation = state["recommendation"]
    # Everything above this line runs again when the graph resumes.
    answer = interrupt(recommendation.model_dump(mode="json"))
    decision = ApprovalDecision.model_validate(answer)
    return {"decision": decision}
