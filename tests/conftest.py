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
def repository() -> Iterator[PostgresPreferencesRepository]:
    database_url = environ.get("TEST_DATABASE_URL") or dotenv_values(".env").get("TEST_DATABASE_URL")
    if not database_url:
        pytest.fail("Set TEST_DATABASE_URL to run database tests.", pytrace=False)

    schema = "test_preferences_" + uuid4().hex
    with psycopg.connect(database_url, autocommit=True, connect_timeout=3) as connection:
        connection.execute(sql.SQL("CREATE SCHEMA {}").format(sql.Identifier(schema)))
        try:
            isolated_url = make_conninfo(database_url, options=f"-c search_path={schema}", connect_timeout=3)
            yield PostgresPreferencesRepository(isolated_url)
        finally:
            connection.execute(sql.SQL("DROP SCHEMA {} CASCADE").format(sql.Identifier(schema)))
