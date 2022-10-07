import json
import elasticsearch
import mgp
from typing import Any, List, Dict, Tuple
from elasticsearch import Elasticsearch
from elasticsearch.helpers import streaming_bulk
from datetime import datetime
import pathlib

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

# Create global logger object
logger = mgp.Logger()


def serialize_vertex(vertex: mgp.Vertex) -> Dict[str, Any]:
    """
    Serializes vertex to specified ElasticSearch schema.
    Args:
        vertex (mgp.Vertex): Reference to the vertex in Memgraph DB
    Returns:
        Dict[str, Any]: ElasticSearch object representation.
    """
    source = serialize_properties(vertex.properties.items())
    source[LK_CATEGORIES] = [label.name for label in vertex.labels]
    return {ACTION: {INDEX: {ID: str(vertex.id)}}, SOURCE: source}


def serialize_edge(edge: mgp.Edge) -> Dict[str, Any]:
    """
    Serializes edge to specified ElasticSearch schema.
    Args:
        edge (mgp.Edge): Reference to the edge in Memgraph DB.
    Returns:
        Dict[str, Any]: ElasticSearch object representation.
    """
    source = serialize_properties(edge.properties.items())
    source[LK_TYPE] = edge.type.name
    return {ACTION: {INDEX: {ID: str(edge.id)}}, SOURCE: source}


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
        hosts=elastic_url,
        ca_certs=ca_certs,
        basic_auth=(elastic_user, elastic_password),
    )
    logger.info(f"Client info: {client.info()}")
    return client


def generate_documents_from_context_objects(
    context_objects: List[Dict[str, Any]]
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """
    Generates nodes and edges documents for indexing and returns them as lists.
    Args:
        context_objects (List[Dict[str, Any]]): Objects that are sent as parameters because of some trigger that was called. Trigger can be for update or for create.
    Returns:
        Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]: serialized nodes and edges
    """
    nodes, edges = [], []
    for context_object in context_objects:
        if context_object[EVENT_TYPE] == CREATED_VERTEX:
            nodes.append(serialize_vertex(context_object[VERTEX]))
        elif context_object[EVENT_TYPE] == CREATED_EDGE:
            edges.append(serialize_edge(context_object[EDGE]))
    return nodes, edges


@mgp.read_proc
def create_index(
    context: mgp.ProcCtx,
    index_name: str,
    schema_path: str,
    elastic_url: str,
    ca_certs: str,
    elastic_user: str,
    elastic_password: str,
) -> mgp.Record(response=mgp.Map):
    """ 
    Creates index with the given index name.
    Args:
        index_name(str): Name of the index that needs to be created.
        elastic_url (str): URL for connecting to the Elasticsearch instance.
        ca_certs (str): Path to the certificate file.
        elastic_user (str): The user trying to connect to the Elasticsearch.
        elastic_password (str): User's password for connecting to the Elasticsearch.
    Returns:
       mgp.Map: response message from Elasticsearch service.  
    """
    # First establish connection with ElasticSearch service.
    client = connect_to_elasticsearch(elastic_url, ca_certs, elastic_user, elastic_password)
    # Read schema from the path given
    with open(schema_path, "r") as schema_file:
        schema_json = json.loads(schema_file.read())
    response = dict(client.indices.create(index=index_name, body=schema_json, ignore=400))
    return mgp.Record(response=response)


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
    chunk_size: int = 500,
    max_chunk_bytes: int = 104857600,
    raise_on_error: bool = True,
    raise_on_exception: bool = True,
    max_retries: int = 0,
    initial_backoff: float = 2.0,
    max_backoff: float = 600.0,
    yield_ok: bool = True,
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
        chunk_size (int): number of docs in one chunk sent to es (default: 500).
        max_chunk_bytes (int): the maximum size of the request in bytes (default: 100MB).
        raise_on_error (bool): raise BulkIndexError containing errors (as .errors) from the execution of the last chunk when some occur. By default we raise.
        raise_on_exception (bool): if False then don’t propagate exceptions from call to bulk and just report the items that failed as failed.
        max_retries (int): maximum number of times a document will be retried when 429 is received, set to 0 (default) for no retries on 429.
        initial_backoff (float): number of seconds we should wait before the first retry. Any subsequent retries will be powers of initial_backoff * 2**retry_number
        max_backoff (float): maximum number of seconds a retry will wait
        yield_ok (float): if set to False will skip successful documents in the output
    Returns:
        mgp.Record(): Returns JSON of all nodes and edges.
    """
    # First establish connection with ElasticSearch service.
    client = connect_to_elasticsearch(elastic_url, ca_certs, elastic_user, elastic_password)

    # Now create iterable of documents that need to be indexed
    nodes, edges = generate_documents_from_context_objects(createdObjects)
    logger.info(f"Nodes: {nodes}")
    logger.info(f"Edges: {edges}")

    # Send documents on indexing
    logger.info("Indexing vertices...")
    for ok, action in streaming_bulk(
        client=client,
        index=node_index,
        actions=nodes,
        chunk_size=chunk_size,
        max_chunk_bytes=max_chunk_bytes,
        initial_backoff=initial_backoff,
        max_backoff=max_backoff,
        yield_ok=yield_ok,
        raise_on_error=raise_on_error,
        raise_on_exception=raise_on_exception,
        max_retries=max_retries,
    ):
        logger.info(f"OK: {ok} Action: {action}")
    logger.info("Indexing edges...")
    for ok, action in streaming_bulk(
        client=client,
        index=edge_index,
        actions=edges,
        chunk_size=chunk_size,
        max_chunk_bytes=max_chunk_bytes,
        initial_backoff=initial_backoff,
        max_backoff=max_backoff,
        yield_ok=yield_ok,
        raise_on_error=raise_on_error,
        raise_on_exception=raise_on_exception,
        max_retries=max_retries,
    ):
        logger.info(f"OK: {ok} Action: {action}")
    return mgp.Record(nodes=nodes, edges=edges)


@mgp.read_proc
def reindex(
    context: mgp.ProcCtx,
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
    logger.info("Reindexing finished")
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
    items = [item for item in response]
    return mgp.Record(items=items)
