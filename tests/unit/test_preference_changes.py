"""How saves and removals apply to stored preferences: nothing disappears unless it is named."""

from decimal import Decimal

from backend.core.ports.preferences import Budget, PreferenceRemoval, Preferences
from backend.core.services.preferences import removed, saved

STORED = Preferences(
    attributes={"size": "43 EU", "width": "wide"},
    likes=["Asics"],
    dislikes=["narrow fit"],
    requirements=["cushioning"],
    usual_budget=Budget(amount=Decimal("130"), currency="EUR"),
)


def test_a_save_adds_to_lists_and_sets_attributes_without_removing_anything() -> None:
    result = saved(STORED, Preferences(attributes={"size": "44 EU"}, dislikes=["suede", "Narrow Fit"]))

    assert result.attributes == {"size": "44 EU", "width": "wide"}
    assert result.dislikes == ["narrow fit", "suede"]  # no duplicate, the stored spelling is kept
    assert result.likes == ["Asics"] and result.requirements == ["cushioning"]
    assert result.usual_budget == STORED.usual_budget


def test_a_removal_takes_out_only_what_is_named() -> None:
    result = removed(STORED, PreferenceRemoval(attributes=["Width"], likes=["asics"], usual_budget=True))

    assert result.attributes == {"size": "43 EU"}
    assert result.likes == []
    assert result.dislikes == ["narrow fit"] and result.requirements == ["cushioning"]
    assert result.usual_budget is None
