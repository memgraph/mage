"""
This module queries Memgraph and Neo4j and creates Graph from JSON exported from Memgraph and 
JSON from APOC from Neo4j

As of 17.7.2023. when importing data via Cypherl, new ids is given to each node in Memgraph and Neo4j.

When exporting data Memgraph export_util uses internal Memgraph ids to export data. 

To overcome the issue of different internal IDs in Neo4j and Memgraph, we use the `id` node property as identifier.

Workaround would be to add API to create nodes by ids on Memgraph when importing via import_util.
"""

import gqlalchemy
import json
import logging
import neo4j
import re

from typing import Any, Dict, List

logging.basicConfig(format="%(asctime)-15s [%(levelname)s]: %(message)s")
logger = logging.getLogger("query_neo_mem")
logger.setLevel(logging.DEBUG)


class Vertex:
    def __init__(self, id: int, labels: List[str], properties: Dict[str, Any]):
        self._id = id
        self._labels = labels
        self._properties = properties
        self._labels.sort()

    @property
    def id(self) -> int:
        return self._id

    def __str__(self) -> str:
        return f"Vertex: {self._id}, {self._labels}, {self._properties}"

    def __eq__(self, other):
        assert isinstance(
            other, Vertex
        ), f"Comparing vertex with object of type {type(other)}"
        logger.debug(f"comparing Vertex with {self._id} to {other._id}")
        if self._id != other._id:
            logger.debug(f"_id different: {self._id} vs {other._id}")
            return False
        if self._labels != other._labels:
            logger.debug(
                f"_labels different between {self._id} and {other._id}: {self._labels} vs {other._labels}"
            )
            return False

        if len(self._properties) != len(other._properties):
            return False
        for k, v in self._properties.items():
            if k not in other._properties:
                logger.debug(
                    f"Property with key {k} not in {other._properties.keys()}"
                )
                return False
            if v != other._properties[k]:
                logger.debug(f"Value {v} not equal to {other._properties[k]}")
                return False

        return True


class Edge:
    def __init__(
        self,
        from_vertex: int,
        to_vertex: int,
        label: str,
        properties: Dict[str, Any],
    ):
        self._from_vertex = from_vertex
        self._to_vertex = to_vertex
        self._label = label
        self._properties = properties

    @property
    def from_vertex(self) -> int:
        return self._from_vertex

    @property
    def to_vertex(self) -> int:
        return self._to_vertex

    def __eq__(self, other):
        assert isinstance(
            other, Edge
        ), f"Comparing Edge with object of type: {type(other)}"
        logger.debug(
            f"comparing Edge ({self._from_vertex}, {self._to_vertex}) to\
              ({other._from_vertex, other._to_vertex})"
        )
        # Return True if self and other have the same length
        if self._from_vertex != other._from_vertex:
            logger.debug(
                f"Source vertex is different {self._from_vertex} <> {other._from_vertex}"
            )
            return False
        if self._to_vertex != other._to_vertex:
            logger.debug(
                f"Destination vertex is different {self._to_vertex} <> {other._to_vertex}"
            )
            return False
        if self._label != other._label:
            logger.debug(f"Label is different {self._label} <> {other._label}")
            return False

        if len(self._properties) != len(other._properties):
            return False
        for k, v in self._properties.items():
            if k not in other._properties:
                logger.debug(
                    f"Property with key {k} not in {other._properties.keys()}"
                )
                return False
            if v != other._properties[k]:
                logger.debug(f"Value {v} not equal to {other._properties[k]}")
                return False
        return True


class Graph:
    def __init__(self):
        self._vertices = []
        self._edges = []

    def add_vertex(self, vertex: Vertex):
        self._vertices.append(vertex)

    def add_edge(self, edge: Edge):
        self._edges.append(edge)

    @property
    def vertices(self):
        return self._vertices

    @property
    def edges(self):
        return self._edges


def get_neo4j_data_json(driver) -> str:
    with driver.session() as session:
        query = neo4j.Query(
            "CALL apoc.export.json.all(null,{useTypes:true, stream:true}) YIELD data RETURN data;"
        )
        result = session.run(query).values()

        res_str = re.sub(r"\\n", ",\n", str(result[0]))
        res_str = re.sub(r"'", "", res_str)

        return json.loads(res_str)


def get_memgraph_data_json_format(memgraph: gqlalchemy.Memgraph):
    result = list(
        memgraph.execute_and_fetch(
            f"""
            CALL export_util.json_stream() YIELD stream RETURN stream;
            """
        )
    )[0]["stream"]
    return json.loads(result)


def extract_vertex_from_json(item) -> Vertex:
    assert (
        item["properties"]["id"] is not None
    ), "Vertex in JSON doesn't have ID property"
    return Vertex(item["properties"]["id"], item["labels"], item["properties"])


def create_edge_from_data(
    from_vertex_id: int, to_vertex_id: int, item
) -> Edge:
    return Edge(
        from_vertex_id, to_vertex_id, item["label"], item["properties"]
    )


def create_graph_memgraph_json(json_memgraph_data) -> Graph:
    logger.debug(f"Memgraph JSON data {json_memgraph_data}")
    graph = Graph()
    vertices_id_mapings = {}
    for item in json_memgraph_data:
        if item["type"] == "node":
            graph.add_vertex(extract_vertex_from_json(item))
            vertices_id_mapings[item["id"]] = item["properties"]["id"]
        else:
            graph.add_edge(
                create_edge_from_data(
                    vertices_id_mapings[item["start"]],
                    vertices_id_mapings[item["end"]],
                    item,
                )
            )

    graph.vertices.sort(key=lambda vertex: vertex.id)
    graph.edges.sort(key=lambda edge: (edge.from_vertex, edge.to_vertex))
    return graph


def create_graph_neo4j_json(json_neo4j_data) -> Graph:
    logger.debug(f"Neo4j JSON data {json_neo4j_data}")
    graph = Graph()
    vertices_id_mapings = {}
    for item in json_neo4j_data:
        if item["type"] == "node":
            graph.add_vertex(extract_vertex_from_json(item))
            vertices_id_mapings[item["id"]] = item["properties"]["id"]
        else:
            if "properties" not in item:
                item["properties"] = {}
            graph.add_edge(
                create_edge_from_data(
                    vertices_id_mapings[item["start"]["id"]],
                    vertices_id_mapings[item["end"]["id"]],
                    item,
                )
            )
    graph.vertices.sort(key=lambda vertex: vertex.id)
    graph.edges.sort(key=lambda edge: (edge.from_vertex, edge.to_vertex))
    return graph


def create_neo4j_driver(port: int) -> neo4j.BoltDriver:
    return neo4j.GraphDatabase.driver(
        f"bolt://localhost:{port}", encrypted=False
    )


def create_memgraph_db(port: int) -> gqlalchemy.Memgraph:
    return gqlalchemy.Memgraph("localhost", port)


def mg_execute_cyphers(input_cyphers: List[str], db: gqlalchemy.Memgraph):
    """
    Execute multiple cypher queries against Memgraph
    """
    for query in input_cyphers:
        db.execute(query)


def neo4j_execute_cyphers(
    input_cyphers: List[str], neo4j_driver: neo4j.BoltDriver
):
    """
    Execute multiple cypher queries against Neo4j
    """
    with neo4j_driver.session() as session:
        for text_query in input_cyphers:
            query = neo4j.Query(text_query)
            session.run(query).values()


def run_memgraph_query(query: str, db: gqlalchemy.Memgraph):
    """
    Execute query against Memgraph
    """
    db.execute(query)


def run_neo4j_query(query: str, neo4j_driver: neo4j.BoltDriver):
    """
    Execute query against Neo4j
    """
    with neo4j_driver.session() as session:
        query = neo4j.Query(query)
        session.run(query).values()


def clean_memgraph_db(memgraph_db: gqlalchemy.Memgraph):
    memgraph_db.drop_database()


def clean_neo4j_db(neo4j_db: neo4j.BoltDriver):
    with neo4j_db.session() as session:
        query = neo4j.Query("MATCH (n) DETACH DELETE n;")
        session.run(query).values()


def mg_get_graph(memgraph_db: gqlalchemy.Memgraph) -> Graph:
    logger.debug("Getting data from Memgraph")
    json_data = get_memgraph_data_json_format(memgraph_db)
    logger.debug("Building the graph from Memgraph JSON data")
    return create_graph_memgraph_json(json_data)


def neo4j_get_graph(neo4j_driver: neo4j.BoltDriver) -> Graph:
    logger.debug("Getting data from Neo4j")
    json_data = get_neo4j_data_json(neo4j_driver)
    logger.debug("Building the graph from Neo4j JSON data")

    return create_graph_neo4j_json(json_data)


# additions for path testing
def sort_dict(dict):
    keys = list(dict.keys())
    keys.sort()
    sorted_dict = {i: dict[i] for i in keys}
    if "id" in sorted_dict: 
        sorted_dict.pop("id")
    return sorted_dict


def execute_query_neo4j(driver, query):
    with driver.session() as session:
        query = neo4j.Query(query)
        results = session.run(query).value()
    return results


def path_to_string_neo4j(path):

    path_string = "PATH: "

    n = len(path.nodes)

    for i in range(0, n):
        
        node = path.nodes[i]
        node_labels = list(node.labels)
        node_labels.sort()
        node_props = str(sort_dict(node._properties))
        path_string += "(id:" + (str(node.get("id")) + " labels: " + str(node_labels) + " " + str(node_props)) + ")-"

        if(i == n - 1):
            path_string = path_string[:-1]
            continue

        relationship = path.relationships[i]
        rel_props = str(sort_dict(relationship._properties))
        path_string += "[" + ("id:" + str(relationship.get("id")) + " type: " + relationship.type + " " + rel_props) + "]-" 
    
    return path_string


def parse_neo4j(results):
    paths = []
    paths = [path_to_string_neo4j(res) for res in results]
    paths.sort()
    return paths


def path_to_string_mem(path):

    path_string = "PATH: "

    n = len(path._nodes)

    for i in range(0, n):
        
        node = path._nodes[i]
        node_labels = list(node._labels)
        node_labels.sort()
        node_props = str(sort_dict(node._properties))
        path_string += "(id:" + (str(node._properties.get("id")) + " labels: " + str(node_labels) + " " + str(node_props)) + ")-"

        if(i == n - 1):
            path_string = path_string[:-1]
            continue

        relationship = path._relationships[i]
        rel_props = str(sort_dict(relationship._properties))
        path_string += "[" + ("id:" + str(relationship._properties.get("id")) + " type: " + relationship._type + " " + rel_props) + "]-" 
    
    return path_string


def parse_mem(results):

    paths = [path_to_string_mem(result["result"]) for result in results]
    paths.sort()
    return paths
