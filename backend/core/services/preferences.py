"""How changes apply to stored preferences, the same for every storage. Nothing is removed unless named."""

from backend.core.ports.preferences import PreferenceRemoval, Preferences


def saved(stored: Preferences, changes: Preferences) -> Preferences:
    return Preferences(
        attributes={**stored.attributes, **changes.attributes},
        likes=added(stored.likes, changes.likes),
        dislikes=added(stored.dislikes, changes.dislikes),
        requirements=added(stored.requirements, changes.requirements),
        usual_budget=changes.usual_budget or stored.usual_budget,
    )


def removed(stored: Preferences, removal: PreferenceRemoval) -> Preferences:
    keys = {key.casefold() for key in removal.attributes}
    return Preferences(
        attributes={key: value for key, value in stored.attributes.items() if key.casefold() not in keys},
        likes=without(stored.likes, removal.likes),
        dislikes=without(stored.dislikes, removal.dislikes),
        requirements=without(stored.requirements, removal.requirements),
        usual_budget=None if removal.usual_budget else stored.usual_budget,
    )


def added(items: list[str], new: list[str]) -> list[str]:
    """Keep the stored items and their spelling, then the new ones not already there."""
    result, known = list(items), {item.casefold() for item in items}
    for item in new:
        if item.casefold() not in known:
            result.append(item)
            known.add(item.casefold())
    return result


def without(items: list[str], names: list[str]) -> list[str]:
    gone = {name.casefold() for name in names}
    return [item for item in items if item.casefold() not in gone]
