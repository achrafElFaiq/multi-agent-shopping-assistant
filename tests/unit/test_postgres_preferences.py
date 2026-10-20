"""The PostgreSQL preferences store applies saves and removals to what is stored."""

from backend.core.adapters.postgres_preferences import PostgresPreferencesRepository
from backend.core.ports.preferences import PreferenceRemoval, Preferences


def test_saves_and_removes_without_touching_the_rest(repository: PostgresPreferencesRepository) -> None:
    repository.save_preferences("shoes", Preferences(attributes={"size": "43 EU"}, dislikes=["narrow fit"]))

    repository.save_preferences("shoes", Preferences(attributes={"gender": "men's"}, dislikes=["suede"]))
    after = repository.remove_preferences("shoes", PreferenceRemoval(dislikes=["suede"]))

    stored = repository.get_preferences("shoes")
    assert stored is not None and stored == after
    assert stored.preferences.attributes == {"size": "43 EU", "gender": "men's"}
    assert stored.preferences.dislikes == ["narrow fit"]
    assert repository.remove_preferences("coffee", PreferenceRemoval(likes=["light roast"])) is None
