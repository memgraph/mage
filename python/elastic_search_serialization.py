import json
import elasticsearch
import mgp
from typing import Any, List, Dict, Tuple
from elasticsearch import Elasticsearch
from elasticsearch.helpers import streaming_bulk
from datetime import datetime

# Elasticsearch constants
ACTION = "action"
INDEX = "index"
ID = "_id"
SOURCE = "source"
SETTINGS = "settings"
NUMBER_OF_SHARDS = "number_of_shards"
NUMBER_OF_REPLICAS = "number_of_replicas"
MAPPINGS = "mappings"
DYNAMIC_TEMPLATES = "dynamic_templates"
MAPPING = "mapping"
ANALYZER = "analyzer"
STRING = "string"
EVENT_TYPE = "event_type"
CREATED_VERTEX = "created_vertex"
CREATED_EDGE = "created_edge"
VERTEX = "vertex"
EDGE = "edge"

# Linkurious constants
LK_TYPE = "lk_type"
LKE_STRING = "_lke_string"
LKE_NUMBER = "_lke_number"
LKE_BOOLEAN = "_lke_boolean"
LKE_DATE = "_lke_date"
LK_CATEGORIES_HAS_RAW = "lk_categories_has_raw"
LK_TYPE_HAS_RAW = "lk_type_has_raw"
LK_CATEGORIES = "lk_categories"


# Mappings of our data types
lke_mapping: Dict[type, str] = {}
lke_mapping[str] = LKE_STRING
lke_mapping[int] = LKE_NUMBER
lke_mapping[float] = LKE_NUMBER
lke_mapping[bool] = LKE_BOOLEAN
lke_mapping[datetime] = LKE_DATE

# Create node index
node_index_body: Dict[str, Any] = {
    "settings": {
        "index": {
            "number_of_shards": 1,  # numberOfShards (default 1),
            "number_of_replicas": 1,  # numberOfReplicas (default 1),
            "mapping": {
                # Add ignore_malformed at index level
                "ignore_malformed": True
            },
            "analysis": {
                "char_filter": {
                    "dot_to_whitespace": {
                        "type": "pattern_replace",
                        "pattern": "(\\D)\\.(\\D)",  # split words on dots
                        "replacement": "$1 $2",
                    },
                    "underscore_to_whitespace": {
                        "type": "pattern_replace",
                        "pattern": "_",  # split all on underscore
                        "replacement": " ",
                    },
                },
                "filter": {"asciifolding_original": {"type": "asciifolding", "preserve_original": True}},
                "analyzer": {
                    # we define our custom analyzer
                    "lk_analyzer": {
                        "tokenizer": "standard",
                        "char_filter": ["dot_to_whitespace", "underscore_to_whitespace"],
                        "filter": ["asciifolding_original", "lowercase", "stop"],
                    }
                },
            },
        }
    },
    "mappings": {
        "dynamic_templates": [
            {
                # store a copy of labels/types in a non-analyzed way for filtering
                "lk_categories_has_raw": {  # for edges, the key is "lk_type_has_raw"
                    "match": "lk_categories",  # for edges, use "lk_type"
                    "mapping": {
                        "type": "text",
                        "analyzer": "lk_analyzer",  # (default "lk_analyzer")
                        "fields": {"raw": {"type": "keyword"}},
                    },
                }
            },
            {
                "string": {
                    "match": "*_lke_string",
                    "mapping": {
                        "type": "text",
                        "analyzer": "lk_analyzer",  # default: "lk_analyzer"
                    },
                }
            },
            {"date": {"match": "*_lke_date", "mapping": {"type": "date"}}},
            {"number": {"match": "*_lke_number", "mapping": {"type": "double"}}},
            {"boolean": {"match": "*_lke_boolean", "mapping": {"type": "keyword"}}},
        ]
    },
}

# Create node index
edge_index_body: Dict[str, Any] = {
    "settings": {
        "index": {
            "number_of_shards": 1,  # numberOfShards (default 1),
            "number_of_replicas": 1,  # numberOfReplicas (default 1),
            "mapping": {
                # Add ignore_malformed at index level
                "ignore_malformed": True
            },
            "analysis": {
                "char_filter": {
                    "dot_to_whitespace": {
                        "type": "pattern_replace",
                        "pattern": "(\\D)\\.(\\D)",  # split words on dots
                        "replacement": "$1 $2",
                    },
                    "underscore_to_whitespace": {
                        "type": "pattern_replace",
                        "pattern": "_",  # split all on underscore
                        "replacement": " ",
                    },
                },
                "filter": {"asciifolding_original": {"type": "asciifolding", "preserve_original": True}},
                "analyzer": {
                    # we define our custom analyzer
                    "lk_analyzer": {
                        "tokenizer": "standard",
                        "char_filter": ["dot_to_whitespace", "underscore_to_whitespace"],
                        "filter": ["asciifolding_original", "lowercase", "stop"],
                    }
                },
            },
        }
    },
    "mappings": {
        "dynamic_templates": [
            {
                # store a copy of labels/types in a non-analyzed way for filtering
                "lk_type_has_raw": {  # for edges, the key is "lk_type_has_raw"
                    "match": "lk_type",  # for edges, use "lk_type"
                    "mapping": {
                        "type": "text",
                        "analyzer": "lk_analyzer",  # (default "lk_analyzer")
                        "fields": {"raw": {"type": "keyword"}},
                    },
                }
            },
            {
                "string": {
                    "match": "*_lke_string",
                    "mapping": {
                        "type": "text",
                        "analyzer": "lk_analyzer",  # default: "lk_analyzer"
                    },
                }
            },
            {"date": {"match": "*_lke_date", "mapping": {"type": "date"}}},
            {"number": {"match": "*_lke_number", "mapping": {"type": "double"}}},
            {"boolean": {"match": "*_lke_boolean", "mapping": {"type": "keyword"}}},
        ]
    },
}


def serialize_vertex(vertex: mgp.Vertex) -> Dict[str, Any]:
    """
    Serializes vertex to specified ElasticSearch schema.
    Args:
        vertex (mgp.Vertex): Reference to the vertex in Memgraph DB
    Returns:
        Dict[str, Any]: ElasticSearch object representation.
    """
    node: Dict[str, Any] = {}  # whole part
    node[ACTION] = {INDEX: {ID: str(vertex.id)}}
    source = serialize_properties(vertex.properties.items())
    source[LK_CATEGORIES] = [label.name for label in vertex.labels]
    node[SOURCE] = source
    return node


def serialize_edge(edge: mgp.Edge) -> Dict[str, Any]:
    """
    Serializes edge to specified ElasticSearch schema.
    Args:
        edge (mgp.Edge): Reference to the edge in Memgraph DB.
    Returns:
        Dict[str, Any]: ElasticSearch object representation.
    """
    rel: Dict[str, Any] = {}
    rel[ACTION] = {INDEX: {ID: str(edge.id)}}
    source = serialize_properties(edge.properties.items())
    source[LK_TYPE] = edge.type.name
    rel[SOURCE] = source
    return rel


def serialize_properties(properties: Dict[str, Any]) -> Dict[str, Any]:
    """
    The method used to serialize properties of vertices and relationships.
    Args:
        properties (Dict[str, Any]]): Properties of nodes and relationships.
    Returns:
        Dict[str, Any]: Object that conforms ElasticSearch's schema.
    """
    source: Dict[str, Any] = {}
    for prop_key, prop_value in properties:
        if type(prop_value) == datetime:
            # Convert datetime to str, replace microsecond and add Z suffix(Zulu or zero offset) manually because Python doesn't support it out of the box
            prop_value = prop_value.replace(microsecond=0).isoformat() + "Z"
            source[prop_key + LKE_DATE] = prop_value
            print(f"Prop value: {prop_value} {type(prop_value)}")
        elif type(prop_value) in lke_mapping:
            source[prop_key + lke_mapping[type(prop_value)]] = prop_value
    return source


def connect_to_elasticsearch(elastic_url: str, ca_certs: str, elastic_user: str, elastic_password) -> Elasticsearch:
    """
    Establishes connection with the Elasticsearch. This configuration needs to be specific to the Elasticsearch deployment.
    Args:
        elastic_url (str): URL for connecting to the Elasticsearch instance.
        ca_certs (str): Path to the certificate file.
        elastic_user (str): The user trying to connect to the Elasticsearch.
        elastic_password (str): User's password for connecting to the Elasticsearch.
    Returns:
        Elasticsearch: Client reference.
    """
    client = Elasticsearch(
        elastic_url,
        ca_certs=ca_certs,
        basic_auth=(elastic_user, elastic_password),
    )
    print(f"Client info: {client.info()}")
    return client


def create_node_index(
    client: Elasticsearch,
    index_name: str = "node_index",
    number_of_shards: int = 1,
    number_of_replicas: int = 1,
    analyzer: str = "lk_analyzer",
) -> None:
    """
    Creates node index if it doesn't exist before.
    Args:
        client (Elasticsearch): A reference to the Elasticsearch client.
        index_name (str): Name you want to give to the node index. Default one is 'node_index'.
        number_of_shards (int): A number of shards you want to use in your index.
        number_of_replicas (int): A number of replicas you want to use in your index.
    """
    global node_index_body
    node_index_body[SETTINGS][INDEX][NUMBER_OF_SHARDS] = number_of_shards
    node_index_body[SETTINGS][INDEX][NUMBER_OF_REPLICAS] = number_of_replicas
    node_index_body[MAPPINGS][DYNAMIC_TEMPLATES][0][LK_CATEGORIES_HAS_RAW][MAPPING][ANALYZER] = analyzer
    node_index_body[MAPPINGS][DYNAMIC_TEMPLATES][1][STRING][MAPPING][ANALYZER] = analyzer
    client.indices.create(index=index_name, body=node_index_body, ignore=400)


def create_edge_index(
    client: Elasticsearch,
    index_name: str = "edge_index",
    number_of_shards: int = 1,
    number_of_replicas: int = 1,
    analyzer: str = "lk_analyzer",
) -> None:
    """
    Creates edge index if it doesn't exist before.
    Args:
        client (Elasticsearch): A reference to the Elasticsearch client.
        index_name (str): Name you want to give to the edge index. Default one is 'edge_index'.
        number_of_shards (int): A number of shards you want to use in your index.
        number_of_replicas (int): A number of replicas you want to use in your index.
    """
    global edge_index_body
    edge_index_body[SETTINGS][INDEX][NUMBER_OF_SHARDS] = number_of_shards
    edge_index_body[SETTINGS][INDEX][NUMBER_OF_REPLICAS] = number_of_replicas
    edge_index_body[MAPPINGS][DYNAMIC_TEMPLATES][0][LK_TYPE_HAS_RAW][MAPPING][ANALYZER] = analyzer
    edge_index_body[MAPPINGS][DYNAMIC_TEMPLATES][1][STRING][MAPPING][ANALYZER] = analyzer
    client.indices.create(index=index_name, body=edge_index_body, ignore=400)


def generate_documents(
    context_vertices: List[mgp.Vertex], context_edges: List[mgp.Edge]
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """
    Generates nodes and edges documents for indexing and returns them as lists.
    Args:
        context_vertices (List[mgp.Vertex]]): Vertices in Memgraph that were created/updated.
        context_edges (List[mgp.Edge]]): Edges in Memgraph that were created/updated.
    Returns:
        Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]
    """
    nodes, edges = [], []
    for vertex in context_vertices:
        nodes.append(serialize_vertex(vertex))
    for edge in context_edges:
        edges.append(serialize_edge(edge))
    return nodes, edges


@mgp.read_proc
def index(
    context: mgp.ProcCtx,
    createdObjects: mgp.List[mgp.Map],
    elastic_url: str,
    ca_certs: str,
    elastic_user: str,
    elastic_password: str,
    node_index_name: str = "node_index",
    edge_index_name: str = "edge_index",
    number_of_shards: int = 1,
    number_of_replicas: int = 1,
    analyzer: str = "lk_analyzer",
) -> mgp.Record(nodes=mgp.List[mgp.Map], edges=mgp.List[mgp.Map]):
    """
    The method serializes all vertices and relationships that are in Memgraph DB to an ElasticSearch schema.
    Args:
        context (mgp.ProcCtx): Reference to the executing context.
        createdObjects (List[Dict[str, Any]]): List of all objects that were created and then sent as arguments to this method with the help of "create trigger".
        elastic_url (str): URL for connecting to the Elasticsearch instance.
        ca_certs (str): Path to the certificate file.
        elastic_user (str): The user trying to connect to the Elasticsearch.
        elastic_password (str): User's password for connecting to the Elasticsearch.
        node_index_name (str): The name of the node index.
        edge_index_name (str): The name of the edge index.
        number_of_shards (int): A number of shards you want to use in your index.
        number_of_replicas (int): A number of replicas you want to use in your index.

    Returns:
        mgp.Record(): Returns JSON of all nodes and edges.
    """
    # First establish connection with ElasticSearch service.
    client = connect_to_elasticsearch(elastic_url, ca_certs, elastic_user, elastic_password)

    # TODO: decouple creating node and edge index
    # Crete indexes if they don't exist
    # create_node_index(
    #     client=client,
    #     index_name=node_index_name,
    #     number_of_shards=number_of_shards,
    #     number_of_replicas=number_of_replicas,
    # )
    # create_edge_index(
    #     client=client,
    #     index_name=edge_index_name,
    #     number_of_shards=number_of_shards,
    #     number_of_replicas=number_of_replicas,
    # )

    # Created objects can be vertices and edges
    created_vertices, created_edges = [], []
    for createdObject in createdObjects:
        if createdObject[EVENT_TYPE] == CREATED_VERTEX:
            created_vertices.append(createdObject[VERTEX])
        elif createdObject[EVENT_TYPE] == CREATED_EDGE:
            created_edges.append(createdObject[EDGE])

    # Now create iterable of documents that need to be indexed
    nodes, edges = generate_documents(created_vertices, created_edges)
    print(f"Nodes: {nodes}")
    print(f"Edges: {edges}")

    # Send documents on indexing
    print("Indexing vertices...")
    for ok, action in streaming_bulk(client=client, index=node_index_name, actions=nodes):
        print(f"OK: {ok} Action: {action}")
    print()
    for ok, action in streaming_bulk(client=client, index=edge_index_name, actions=edges):
        print(f"OK: {ok} Action: {action}")
    return mgp.Record(nodes=nodes, edges=edges)


@mgp.read_proc
def scan(
    context: mgp.ProcCtx,
    elastic_url: str,
    ca_certs: str,
    elastic_user: str,
    elastic_password: str,
    index_name: str,
    query: str,
) -> mgp.Record(items=mgp.List[mgp.Map]):
    """
    Runs a query on a index specified by the index_name.
    Args:
        context (mgp.ProcCtx): Reference to the executing context.
        elastic_url (str): URL for connecting to the Elasticsearch instance.
        ca_certs (str): Path to the certificate file.
        elastic_user (str): The user trying to connect to the Elasticsearch.
        elastic_password (str): User's password for connecting to the Elasticsearch.
        index_name (str): A name of the index.
        query (str): Query written as JSON.
    Returns:
         mgp.Record(items=mgp.List[mgp.Map]): List of all items matched by the specific query.
    """
    client = connect_to_elasticsearch(elastic_url, ca_certs, elastic_user, elastic_password)
    query_obj = json.loads(query)
    response = elasticsearch.helpers.scan(
        client,
        query=query_obj,
        index=index_name,
    )
    items = []
    for item in response:
        items.append(item)
    return mgp.Record(items=items)
