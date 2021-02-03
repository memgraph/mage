import os
from typing import Any, Dict, Iterator
from database.connection import Connection

__all__ = ("Memgraph",)


MG_HOST = os.getenv("MG_HOST", "127.0.0.1")
MG_PORT = int(os.getenv("MG_PORT", "7687"))
MG_USERNAME = os.getenv("MG_USERNAME", "")
MG_PASSWORD = os.getenv("MG_PASSWORD", "")
MG_ENCRYPTED = os.getenv("MG_ENCRYPT", "").lower() == "true"


class Memgraph:
    def __init__(
        self,
        host: str = None,
        port: int = None,
        username: str = "",
        password: str = "",
        encrypted: bool = None,
    ):
        self._host = host or MG_HOST
        self._port = port or MG_PORT
        self._username = username or MG_USERNAME
        self._password = password or MG_PASSWORD
        self._encrypted = encrypted if encrypted is not None else MG_ENCRYPTED
        self._cached_connection = None

    def execute_and_fetch(self, query: str, connection: Connection = None) -> Iterator[Dict[str, Any]]:
        """Executes Cypher query and returns iterator of results."""
        connection = connection or self._get_cached_connection()
        return connection.execute_and_fetch(query)

    def execute_query(self, query: str, connection: Connection = None) -> None:
        """Executes Cypher query without returning any results."""
        connection = connection or self._get_cached_connection()
        return connection.execute_query(query)

    def _get_cached_connection(self) -> Connection:
        """Returns cached connection if it exists, creates it otherwise"""
        if self._cached_connection is None or not self._cached_connection.is_active():
            self._cached_connection = self.new_connection()

        return self._cached_connection

    def new_connection(self) -> Connection:
        """Creates new Memgraph connection"""
        args = dict(
            host=self._host,
            port=self._port,
            username=self._username,
            password=self._password,
            encrypted=self._encrypted,
        )
        return Connection.create(**args)
