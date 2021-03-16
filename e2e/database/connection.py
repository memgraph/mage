from abc import ABC, abstractmethod
from typing import Any, Dict, Iterator
from database.models import Node, Relationship

_use_mgclient = True
try:
    import mgclient
except ImportError:
    from neo4j import GraphDatabase, basic_auth
    from neo4j.types import Relationship as Neo4jRelationship
    from neo4j.types import Node as Neo4jNode

    _use_mgclient = False


__all__ = ("Connection",)


class Connection(ABC):
    def __init__(
        self, host: str, port: int, username: str, password: str, encrypted: bool
    ):
        self.host = host
        self.port = port
        self.username = username
        self.password = password
        self.encrypted = encrypted

    @abstractmethod
    def execute_query(self, query: str) -> None:
        """Executes Cypher query without returning any results."""
        pass

    @abstractmethod
    def execute_and_fetch(self, query: str) -> Iterator[Dict[str, Any]]:
        """Executes Cypher query and returns iterator of results."""
        pass

    @abstractmethod
    def is_active(self) -> bool:
        """Returns True if connection is active and can be used"""
        pass

    @staticmethod
    def create(**kwargs) -> "Connection":
        return (
            MemgraphConnection(**kwargs) if _use_mgclient else Neo4jConnection(**kwargs)
        )


class MemgraphConnection(Connection):
    def __init__(
        self,
        host: str,
        port: int,
        username: str,
        password: str,
        encrypted: bool,
        lazy: bool = True,
    ):
        super().__init__(host, port, username, password, encrypted)
        self.lazy = lazy
        self._connection = self._create_connection()

    def execute_query(self, query: str) -> None:
        """Executes Cypher query without returning any results."""
        cursor = self._connection.cursor()
        cursor.execute(query)
        cursor.fetchall()

    def execute_and_fetch(self, query: str) -> Iterator[Dict[str, Any]]:
        """Executes Cypher query and returns iterator of results."""
        cursor = self._connection.cursor()
        cursor.execute(query)
        while True:
            row = cursor.fetchone()
            if row is None:
                break
            yield {
                dsc.name: _convert_memgraph_value(row[index])
                for index, dsc in enumerate(cursor.description)
            }

    def is_active(self) -> bool:
        """Returns True if connection is active and can be used"""
        return (
            self._connection is not None
            and self._connection.status == mgclient.CONN_STATUS_READY
        )

    def _create_connection(self):
        sslmode = (
            mgclient.MG_SSLMODE_REQUIRE
            if self.encrypted
            else mgclient.MG_SSLMODE_DISABLE
        )
        return mgclient.connect(
            host=self.host,
            port=self.port,
            username=self.username,
            password=self.password,
            sslmode=sslmode,
            lazy=self.lazy,
        )


class Neo4jConnection(Connection):
    def __init__(
        self, host: str, port: int, username: str, password: str, encrypted: bool
    ):
        super().__init__(host, port, username, password, encrypted)
        self._connection = self._create_connection()

    def execute_query(self, query: str) -> None:
        """Executes Cypher query without returning any results."""
        with self._connection.session() as session:
            session.run(query)

    def execute_and_fetch(self, query: str) -> Iterator[Dict[str, Any]]:
        """Executes Cypher query and returns iterator of results."""
        with self._connection.session() as session:
            results = session.run(query)
            columns = results.keys()
            for result in results:
                yield {
                    column: _convert_neo4j_value(result[column]) for column in columns
                }

    def is_active(self) -> bool:
        """Returns True if connection is active and can be used"""
        return self._connection is not None

    def _create_connection(self):
        return GraphDatabase.driver(
            f"bolt://{self.host}:{self.port}",
            auth=basic_auth(self.username, self.password),
            encrypted=self.encrypted,
        )


def _convert_memgraph_value(value: Any) -> Any:
    """Converts Memgraph objects to custom Node/Relationship objects"""
    if isinstance(value, mgclient.Relationship):
        return Relationship(
            rel_id=value.id,
            rel_type=value.type,
            start_node=value.start_id,
            end_node=value.end_id,
            properties=value.properties,
        )

    if isinstance(value, mgclient.Node):
        return Node(node_id=value.id, labels=value.labels, properties=value.properties)

    return value


def _convert_neo4j_value(value: Any) -> Any:
    """Converts Neo4j objects to custom Node/Relationship objects"""
    if isinstance(value, Neo4jRelationship):
        return Relationship(
            rel_id=value.id,
            rel_type=value.type,
            start_node=value.start_node,
            end_node=value.end_node,
            properties=dict(value.items()),
        )

    if isinstance(value, Neo4jNode):
        return Node(
            node_id=value.id, labels=value.labels, properties=dict(value.items())
        )

    return value
