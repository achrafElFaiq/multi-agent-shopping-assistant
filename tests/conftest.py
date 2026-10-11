"""PostgreSQL tests use a separate schema per test, removed afterwards."""

from collections.abc import Iterator
from os import environ
from uuid import uuid4

import psycopg
import pytest
from dotenv import dotenv_values
from psycopg import sql
from psycopg.conninfo import make_conninfo

from backend.core.adapters.postgres_preferences import PostgresPreferencesRepository


@pytest.fixture
def database_url() -> Iterator[str]:
    test_database_url = environ.get("TEST_DATABASE_URL") or dotenv_values(".env").get("TEST_DATABASE_URL")
    if not test_database_url:
        pytest.fail("Set TEST_DATABASE_URL to run database tests.", pytrace=False)

    schema = "test_" + uuid4().hex
    with psycopg.connect(test_database_url, autocommit=True, connect_timeout=3) as connection:
        connection.execute(sql.SQL("CREATE SCHEMA {}").format(sql.Identifier(schema)))
        try:
            # public stays visible for extension types such as pgvector's vector; tables go to the test schema.
            yield make_conninfo(test_database_url, options=f"-c search_path={schema},public", connect_timeout=3)
        finally:
            connection.execute(sql.SQL("DROP SCHEMA {} CASCADE").format(sql.Identifier(schema)))


@pytest.fixture
def repository(database_url: str) -> PostgresPreferencesRepository:
    return PostgresPreferencesRepository(database_url)
