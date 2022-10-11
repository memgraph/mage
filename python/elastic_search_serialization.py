import json
from typing import Any, List, Dict, Tuple
from datetime import datetime

import elasticsearch
from elasticsearch.helpers import streaming_bulk

import mgp


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
INDEX_TYPE = "index_type"

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
logger: mgp.Logger = mgp.Logger()

# Global client object
client: elasticsearch.Elasticsearch


# Helper method
def serialize_vertex(vertex: mgp.Vertex) -> Dict[str, Any]:
    """Serializes vertex to specified ElasticSearch schema.
    Args:
        vertex (mgp.Vertex): Reference to the vertex in Memgraph DB
    Returns:
        Dict[str, Any]: ElasticSearch object representation.
    """
    source = serialize_properties(vertex.properties.items())
    source[LK_CATEGORIES] = [label.name for label in vertex.labels]
    return {ACTION: {INDEX: {ID: str(vertex.id)}}, SOURCE: source}


def serialize_edge(edge: mgp.Edge) -> Dict[str, Any]:
    """Serializes edge to specified ElasticSearch schema.
    Args:
        edge (mgp.Edge): Reference to the edge in Memgraph DB.
    Returns:
        Dict[str, Any]: ElasticSearch object representation.
    """
    source = serialize_properties(edge.properties.items())
    source[LK_TYPE] = edge.type.name
    return {ACTION: {INDEX: {ID: str(edge.id)}}, SOURCE: source}


def serialize_properties(properties: Dict[str, Any]) -> Dict[str, Any]:
    """The method used to serialize properties of vertices and relationships.
    Args:
        properties (Dict[str, Any]]): Properties of nodes and relationships.
    Returns:
        Dict[str, Any]: Object that conforms ElasticSearch's schema.
    """
    source: Dict[str, Any] = {}
    for prop_key, prop_value in properties:
        if isinstance(prop_value, datetime):
            # Convert datetime to str, replace microsecond and add Z suffix(Zulu or zero offset) manually because Python doesn't support it out of the box
            prop_value = f"{prop_value.replace(microsecond=0).isoformat()}Z"
            source[f"{prop_key}{LKE_DATE}"] = prop_value
        elif type(prop_value) in lke_mapping:
            source[f"{prop_key}{lke_mapping[type(prop_value)]}"] = prop_value
    return source


def generate_documents_from_triggered_objects(
    context_objects: List[Dict[str, Any]]
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Generates vertices and edges documents for indexing and returns them as lists.
    Args:
        context_objects (List[Dict[str, Any]]): Objects that are sent as parameters because of some trigger that was called. Trigger can be for update or for create.
    Returns:
        Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]: Serialized vertices and edges.
    """
    vertices, edges = [], []
    for context_object in context_objects:
        if context_object[EVENT_TYPE] == CREATED_VERTEX:
            vertices.append(serialize_vertex(context_object[VERTEX]))
        elif context_object[EVENT_TYPE] == CREATED_EDGE:
            edges.append(serialize_edge(context_object[EDGE]))
    return vertices, edges


def generate_documents_from_db(context: mgp.ProcCtx) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Generates vertices and edges from the database.
    Args:
        context (mgp.ProcCtx): A reference to the context execution.
    Returns:
        Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]: Serialized vertices and edges.
    """
    vertices, edges = [], []
    for vertex in context.graph.vertices:
        vertices.append(serialize_vertex(vertex))
        for edge in vertex.out_edges:
            edges.append(serialize_edge(edge))

    return vertices, edges


def elastic_search_streaming_bulk(
    objects: List[Any],
    index: str,
    chunk_size: int = 500,
    max_chunk_bytes: int = 104857600,
    raise_on_error: bool = True,
    raise_on_exception: bool = True,
    max_retries: int = 0,
    initial_backoff: float = 2.0,
    max_backoff: float = 600.0,
    yield_ok: bool = True,
) -> None:
    """
    Sends streaming_bulk requests for the given objects to the provided index with the parameters specified.
    Args:
        objects (List[Any]): serialized nodes and edges that will be sent to the ElasticSearch.
        index (str): The name of the index where you want to save the data.
        number_of_shards (int): A number of shards you want to use in your index.
        number_of_replicas (int): A number of replicas you want to use in your index.
        chunk_size (int): The number of docs in one chunk sent to es (default: 500).
        max_chunk_bytes (int): The maximum size of the request in bytes (default: 100MB).
        raise_on_error (bool): Raise BulkIndexError containing errors (as .errors) from the execution of the last chunk when some occur. By default we raise.
        raise_on_exception (bool): If False then don’t propagate exceptions from call to bulk and just report the items that failed as failed.
        max_retries (int): Maximum number of times a document will be retried when 429 is received, set to 0 (default) for no retries on 429.
        initial_backoff (float): The number of seconds we should wait before the first retry. Any subsequent retries will be powers of initial_backoff * 2**retry_number.
        max_backoff (float): The maximum number of seconds a retry will wait.
        yield_ok (float): If set to False will skip successful documents in the output.
    """
    for _, _ in streaming_bulk(
        client=client,
        index=index,
        actions=objects,
        chunk_size=chunk_size,
        max_chunk_bytes=max_chunk_bytes,
        initial_backoff=initial_backoff,
        max_backoff=max_backoff,
        yield_ok=yield_ok,
        raise_on_error=raise_on_error,
        raise_on_exception=raise_on_exception,
        max_retries=max_retries,
    ):
        pass


@mgp.read_proc
def connect(
    elastic_url: str, ca_certs: str, elastic_user: str, elastic_password
) -> mgp.Record(connection_status=mgp.Map):
    """Establishes connection with the Elasticsearch. This configuration needs to be specific to the Elasticsearch deployment. Uses basic authentication
    Args:
        elastic_url (str): URL for connecting to the Elasticsearch instance.
        ca_certs (str): Path to the certificate file.
        elastic_user (str): The user trying to connect to the Elasticsearch.
        elastic_password (str): User's password for connecting to the Elasticsearch.
    Returns:
        mgp.Record(connection_status=mgp.Map): Connection info.
    """
    global client
    client = elasticsearch.Elasticsearch(
        hosts=elastic_url,
        ca_certs=ca_certs,
        basic_auth=(elastic_user, elastic_password),
    )
    logger.info(f"Client info: {client.info()}")
    return mgp.Record(connection_status=dict(client.info()))


@mgp.read_proc
def create_index(
    context: mgp.ProcCtx,
    index_name: str,
    schema_path: str,
    schema_parameters: mgp.Map,
) -> mgp.Record(response=mgp.Map):
    """Creates index with the given index name.
    Args:
        index_name (str): Name of the index that needs to be created.
        schema_path (str): Path to the schema from where it will be loaded.
        schema_parameters: Dict[str, Any]
            number_of_shards (int): Number of shards index will use.
            number_of_replicas (int): Number of replicas index will use.
        analyzer (str): Custom analyzer, can be set to any legal Elasticsearch analyzer.
    Returns:
       mgp.Map: Response message from Elasticsearch service.
    """
    global client
    # Read schema from the path given
    with open(schema_path, "r") as schema_file:
        schema_json = json.loads(schema_file.read())
    # Update default schema if specified
    if NUMBER_OF_SHARDS in schema_parameters:
        schema_json[SETTINGS][INDEX][NUMBER_OF_SHARDS] = schema_parameters[NUMBER_OF_SHARDS]
        logger.info(f"Number of shards updated to: {schema_parameters[NUMBER_OF_SHARDS]}")
    if NUMBER_OF_REPLICAS in schema_parameters:
        schema_json[SETTINGS][INDEX][NUMBER_OF_REPLICAS] = schema_parameters[NUMBER_OF_REPLICAS]
        logger.info(f"Number of replicas updated to: {schema_parameters[NUMBER_OF_REPLICAS]}")
    if ANALYZER in schema_parameters and INDEX_TYPE in schema_parameters:
        schema_json[MAPPINGS][DYNAMIC_TEMPLATES][1][STRING][MAPPING][ANALYZER] = schema_parameters[ANALYZER]
        if schema_parameters[INDEX_TYPE] == VERTEX:
            schema_json[MAPPINGS][DYNAMIC_TEMPLATES][0][LK_CATEGORIES_HAS_RAW][MAPPING][ANALYZER] = schema_parameters[
                ANALYZER
            ]
        else:
            schema_json[MAPPINGS][DYNAMIC_TEMPLATES][0][LK_TYPE_HAS_RAW][MAPPING][ANALYZER] = schema_parameters[
                ANALYZER
            ]
        logger.info(f"Analyzer set to: {schema_parameters[ANALYZER]}")

    return mgp.Record(response=dict(client.indices.create(index=index_name, body=schema_json, ignore=400)))


@mgp.read_proc
def index_db(
    context: mgp.ProcCtx,
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
    """The method serializes all vertices and relationships that are in Memgraph DB to an ElasticSearch schema.
    Args:
        context (mgp.ProcCtx): Reference to the executing context.
        createdObjects (List[Dict[str, Any]]): List of all objects that were created andthen sent as arguments to this method with the help of "create trigger".
        node_index (str): The name of the node index.
        edge_index (str): The name of the edge index.
        number_of_shards (int): A number of shards you want to use in your index.
        number_of_replicas (int): A number of replicas you want to use in your index.
        chunk_size (int): The number of docs in one chunk sent to es (default: 500).
        max_chunk_bytes (int): The maximum size of the request in bytes (default: 100MB).
        raise_on_error (bool): Raise BulkIndexError containing errors (as .errors) from the execution of the last chunk when some occur. By default we raise.
        raise_on_exception (bool): If False then don’t propagate exceptions from call to bulk and just report the items that failed as failed.
        max_retries (int): Maximum number of times a document will be retried when 429 is received, set to 0 (default) for no retries on 429.
        initial_backoff (float): The number of seconds we should wait before the first retry. Any subsequent retries will be powers of initial_backoff * 2**retry_number.
        max_backoff (float): The maximum number of seconds a retry will wait.
        yield_ok (float): If set to False will skip successful documents in the output.
    Returns:
        mgp.Record(): Returns JSON of all nodes and edges.
    """
    global client
    # Now create iterable of documents that need to be indexed
    nodes, edges = generate_documents_from_db(context)

    # Send nodes on indexing
    logger.info("Indexing vertices...")
    elastic_search_streaming_bulk(
        nodes,
        node_index,
        chunk_size,
        max_chunk_bytes,
        raise_on_error,
        raise_on_exception,
        max_retries,
        initial_backoff,
        max_backoff,
        yield_ok,
    )
    # Send edges on indexing
    logger.info("Indexing edges...")
    elastic_search_streaming_bulk(
        edges,
        edge_index,
        chunk_size,
        max_chunk_bytes,
        raise_on_error,
        raise_on_exception,
        max_retries,
        initial_backoff,
        max_backoff,
        yield_ok,
    )
    return mgp.Record(nodes=nodes, edges=edges)


@mgp.read_proc
def index(
    context: mgp.ProcCtx,
    createdObjects: mgp.List[mgp.Map],
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
    """The method serializes all vertices and relationships that came into the Memgraph DB to an ElasticSearch schema and sends streaming_bulk request to ElasticSearch's API.
    Args:
        context (mgp.ProcCtx): Reference to the executing context.
        createdObjects (List[Dict[str, Any]]): List of all objects that were created andthen sent as arguments to this method with the help of "create trigger".
        node_index (str): The name of the node index.
        edge_index (str): The name of the edge index.
        number_of_shards (int): A number of shards you want to use in your index.
        number_of_replicas (int): A number of replicas you want to use in your index.
        chunk_size (int): The number of docs in one chunk sent to es (default: 500).
        max_chunk_bytes (int): The maximum size of the request in bytes (default: 100MB).
        raise_on_error (bool): Raise BulkIndexError containing errors (as .errors) from the execution of the last chunk when some occur. By default we raise.
        raise_on_exception (bool): If False then don’t propagate exceptions from call to bulk and just report the items that failed as failed.
        max_retries (int): Maximum number of times a document will be retried when 429 is received, set to 0 (default) for no retries on 429.
        initial_backoff (float): The number of seconds we should wait before the first retry. Any subsequent retries will be powers of initial_backoff * 2**retry_number.
        max_backoff (float): The maximum number of seconds a retry will wait.
        yield_ok (float): If set to False will skip successful documents in the output.
    Returns:
        mgp.Record(): Returns JSON of all nodes and edges.
    """
    global client
    # Now create iterable of documents that need to be indexed
    nodes, edges = generate_documents_from_triggered_objects(createdObjects)

    # Send nodes on indexing
    logger.info("Indexing vertices...")
    elastic_search_streaming_bulk(
        nodes,
        node_index,
        chunk_size,
        max_chunk_bytes,
        raise_on_error,
        raise_on_exception,
        max_retries,
        initial_backoff,
        max_backoff,
        yield_ok,
    )
    # Send edges on indexing
    logger.info("Indexing edges...")
    elastic_search_streaming_bulk(
        edges,
        edge_index,
        chunk_size,
        max_chunk_bytes,
        raise_on_error,
        raise_on_exception,
        max_retries,
        initial_backoff,
        max_backoff,
        yield_ok,
    )
    return mgp.Record(nodes=nodes, edges=edges)


@mgp.read_proc
def reindex(
    context: mgp.ProcCtx,
    source_index: mgp.Any,
    target_index: str,
    query: str,
    chunk_size: int = 500,
    scroll: str = "5m",
    op_type: mgp.Nullable[str] = None,
) -> mgp.Record(response=mgp.Map):
    """Reindex all documents from one index that satisfy a given query to another, potentially (if target_client is specified) on a different cluster. If you don’t specify the query you will reindex all the documents.
    Args:
        context (mgp.ProcCtx): Reference to the executing context.
        updatatedObjects (List[Dict[str, Any]]): List of all objects that were updated and then sent as arguments to this method with the help of the "update trigger".
        source_index (Union[str, List[str]]): Identifies source index(or more of them) from where documents need to be indexed.
        target_index (str): Identifies target index to where documents need to be indexed.
        query (str): Query written as JSON.
        chunk_size (int): number of docs in one chunk sent to es (default: 500).
        scroll (str): Specify how long a consistent view of the index should be maintained for scrolled search.
        op_type (Optional[str]): Explicit operation type. Defaults to ‘_index’. Data streams must be set to ‘create’. If not specified, will auto-detect if target_index is a data stream.
    Returns:
        response (str): Response of the query.
    """
    global client
    response = dict(
        elasticsearch.helpers.reindex(
            client=client,
            source_index=source_index,
            target_index=target_index,
            query=json.loads(query),
            chunk_size=chunk_size,
            scroll=scroll,
            op_type=op_type,
        )
    )
    logger.info("Reindexing finished")
    return mgp.Record(response=response)


@mgp.read_proc
def scan(
    context: mgp.ProcCtx,
    index_name: str,
    query: str,
    scroll: str = "5m",
    raise_on_error: bool = True,
    preserve_order: bool = False,
    size: int = 1000,
    request_timeout: mgp.Nullable[float] = None,
    clear_scroll: bool = True,
) -> mgp.Record(items=mgp.List[mgp.Map]):
    """Runs a query on a index specified by the index_name.
    Args:
        context (mgp.ProcCtx): Reference to the executing context.
        index_name (str): A name of the index.
        query (str): Query written as JSON.
        scroll (int): Specify how long a consistent view of the index should be maintained for scrolled search.
        raise_on_error (bool): Raises an exception (ScanError) if an error is encountered (some shards fail to execute). By default we raise.
        preserve_order (bool): Don’t set the search_type to scan - this will cause the scroll to paginate with preserving the order. Note that this can be an extremely expensive operation and can easily lead to unpredictable results, use with caution.
        size (int): Size (per shard) of the batch send at each iteration.
        request_timeout (mgp.Nullable[float]): Explicit timeout for each call to scan.
        clear_scroll (bool): Explicitly calls delete on the scroll id via the clear scroll API at the end of the method on completion or error, defaults to true.
    Returns:
         mgp.Record(items=mgp.List[mgp.Map]): List of all items matched by the specific query.
    """
    global client
    response = elasticsearch.helpers.scan(
        client,
        query=json.loads(query),
        index=index_name,
        scroll=scroll,
        raise_on_error=raise_on_error,
        preserve_order=preserve_order,
        size=size,
        request_timeout=request_timeout,
        clear_scroll=clear_scroll,
    )
    return mgp.Record(items=[item for item in response])
