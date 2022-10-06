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
                "filter": {
                    "asciifolding_original": {
                        "type": "asciifolding",
                        "preserve_original": True,
                    }
                },
                "analyzer": {
                    # we define our custom analyzer
                    "lk_analyzer": {
                        "tokenizer": "standard",
                        "char_filter": [
                            "dot_to_whitespace",
                            "underscore_to_whitespace",
                        ],
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
                "filter": {
                    "asciifolding_original": {
                        "type": "asciifolding",
                        "preserve_original": True,
                    }
                },
                "analyzer": {
                    # we define our custom analyzer
                    "lk_analyzer": {
                        "tokenizer": "standard",
                        "char_filter": [
                            "dot_to_whitespace",
                            "underscore_to_whitespace",
                        ],
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


def get_created_updated_objects(context_objects: List[Dict[str, Any]]) -> Tuple[List[mgp.Vertex], List[mgp.Edge]]:
    """
    Extracts nodes and edges from context_objects.
    Args:
        context_objects (List[Dict[str, Any]]): Objects that are sent as parameters because of some trigger that was called. Trigger can be for update or for create.
    Returns:
        Tuple[List[mgp.Vertex], List[mgp.Edge]]: Extracted vertices and egdes.
    """
    context_vertices, context_edges = [], []
    for context_object in context_objects:
        if context_object[EVENT_TYPE] == CREATED_VERTEX:
            context_vertices.append(context_object[VERTEX])
        elif context_object[EVENT_TYPE] == CREATED_EDGE:
            context_edges.append(context_object[EDGE])
    return context_vertices, context_edges


@mgp.read_proc
def index(
    context: mgp.ProcCtx,
    createdObjects: mgp.List[mgp.Map],
    elastic_url: str,
    ca_certs: str,
    elastic_user: str,
    elastic_password: str,
    node_index: str,
    edge_index: str,
    number_of_shards: int = 1,
    number_of_replicas: int = 1,
    analyzer: str = "lk_analyzer",
) -> mgp.Record(nodes=mgp.List[mgp.Map], edges=mgp.List[mgp.Map]):
    """
    The method serializes all vertices and relationships that are in Memgraph DB to an ElasticSearch schema.
    Args:
        context (mgp.ProcCtx): Reference to the executing context.
        createdObjects (List[Dict[str, Any]]): List of all objects that were created andthen sent as arguments to this method with the help of "create trigger".
        elastic_url (str): URL for connecting to the Elasticsearch instance.
        ca_certs (str): Path to the certificate file.
        elastic_user (str): The user trying to connect to the Elasticsearch.
        elastic_password (str): User's password for connecting to the Elasticsearch.
        node_index (str): The name of the node index.
        edge_index (str): The name of the edge index.
        number_of_shards (int): A number of shards you want to use in your index.
        number_of_replicas (int): A number of replicas you want to use in your index.

    Returns:
        mgp.Record(): Returns JSON of all nodes and edges.
    """
    # First establish connection with ElasticSearch service.
    client = connect_to_elasticsearch(elastic_url, ca_certs, elastic_user, elastic_password)

    # Crete indexes if they don't exist
    create_node_index(
        client=client,
        index_name=node_index,
        number_of_shards=number_of_shards,
        number_of_replicas=number_of_replicas,
    )
    create_edge_index(
        client=client,
        index_name=edge_index,
        number_of_shards=number_of_shards,
        number_of_replicas=number_of_replicas,
    )

    # Created objects can be vertices and edges
    created_vertices, created_edges = get_created_updated_objects(createdObjects)

    # Now create iterable of documents that need to be indexed
    nodes, edges = generate_documents(created_vertices, created_edges)
    print(f"Nodes: {nodes}")
    print(f"Edges: {edges}")

    # Send documents on indexing
    print("Indexing vertices...")
    for ok, action in streaming_bulk(client=client, index=node_index, actions=nodes):
        print(f"OK: {ok} Action: {action}")
    print()
    for ok, action in streaming_bulk(client=client, index=edge_index, actions=edges):
        print(f"OK: {ok} Action: {action}")
    return mgp.Record(nodes=nodes, edges=edges)


@mgp.read_proc
def reindex(
    context: mgp.ProcCtx,
    updatedObjects: mgp.List[mgp.Map],
    elastic_url: str,
    ca_certs: str,
    elastic_user: str,
    elastic_password: str,
    source_index: mgp.Any,
    target_index: str,
    query: str,
    chunk_size: int = 500,
    scroll: str = "5m",
    op_type: mgp.Nullable[str] = None,
) -> mgp.Record(response=str):
    """
    Reindex all documents from one index that satisfy a given query to another, potentia    lly (if target_client is specified) on a different cluster. If you don’t specify the    query you will reindex all the documents.
    Args:
        context (mgp.ProcCtx): Reference to the executing context.
        updatatedObjects (List[Dict[str, Any]]): List of all objects that were updated and then sent as arguments to this method with the help of the "update trigger".
        elastic_url (str): URL for connecting to the Elasticsearch instance.
        ca_certs (str): Path to the certificate file.
        elastic_user (str): The user trying to connect to the Elasticsearch.
        elastic_password str: User's password for connecting to the Elasticsearch.
        source_index (Union[str, List[str]]): Identifies source index(or more of them) from where documents need to be indexed.
        target_index (str): Identifies target index to where documents need to be indexed.
        query (str): Query written as JSON.
        chunk_size (int): number of docs in one chunk sent to es (default: 500).
        scroll (str): Specify how long a consistent view of the index should be maintained for scrolled search.
        op_type (Optional[str]): Explicit operation type. Defaults to ‘_index’. Data streams must be set to ‘create’. If not specified, will auto-detect if target_index is a data stream.
    Returns:
        response (str): Response of the query.
    """
    client = connect_to_elasticsearch(elastic_url, ca_certs, elastic_user, elastic_password)
    query_obj = json.loads(query)
    response = elasticsearch.helpers.reindex(
        client=client,
        source_index=source_index,
        target_index=target_index,
        query=query_obj,
        chunk_size=chunk_size,
        scroll=scroll,
        op_type=op_type,
    )
    print("Reindexing finished")
    return mgp.Record(response=str(response))


@mgp.read_proc
def scan(
    context: mgp.ProcCtx,
    elastic_url: str,
    ca_certs: str,
    elastic_user: str,
    elastic_password: str,
    index_name: str,
    query: str,
    scroll: str = "5m",
    raise_on_error: bool = True,
    preserve_order: bool = False,
    size: int = 1000,
    request_timeout: mgp.Nullable[float] = None,
    clear_scroll: bool = True,
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
        scroll=scroll,
        raise_on_error=raise_on_error,
        preserve_order=preserve_order,
        size=size,
        request_timeout=request_timeout,
        clear_scroll=clear_scroll,
    )
    items = []
    for item in response:
        items.append(item)
    return mgp.Record(items=items)
