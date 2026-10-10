"""What was recommended to the shopper and what they answered: the history recommendations will learn from."""

from datetime import datetime

from pydantic import BaseModel

from backend.core.models.products import Offer


class RecommendationRecord(BaseModel):
    query: str  # the shopping request
    category: str
    offer: Offer
    pitch: str  # why the agent thought it fits
    accepted: bool
    reason: str = ""  # why the shopper said no, in their words; empty if they gave none
    decided_at: datetime
