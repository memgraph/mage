import neo4j
import re
import json
import gqlalchemy
from typing import Any, Dict, List

class Vertex:
    def __init__(self, id:int, labels:List[str], properties: Dict[str, Any]):
        self._id = id
        self._labels = labels
        self._properties = properties
        self._labels.sort()

    @property
    def id(self)->int:
        return self._id
    def __str__(self) -> str:
        return f"Vertex: {self._id}, {self._labels}, {self._properties}"
    def __eq__(self, other):
        assert isinstance(other, Vertex)
        print(f'comparing Vertex with {self._id} to {other._id}')
        # Return True if self and other have the same length
        if self._id != other._id:
            print("_id different")
            return False
        if self._labels != other._labels:
            print("_labels different")
            return False
        for k,v in self._properties.items():
            if k not in other._properties:
                print(f"{k} not in {other._properties.keys()}")
                return False
            if v!= other._properties[k]:
                print(f"{v} not equal to {other._properties[k]}")
                return False
        return True

class Edge:
    def __init__(self, from_vertex:int, to_vertex:int, label:str, properties: Dict[str, Any]):
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
        assert isinstance(other, Edge)
        print(f'comparing Edge ({self._from_vertex}, {self._to_vertex}) to\
              ({other._from_vertex, other._to_vertex})')
        # Return True if self and other have the same length
        if self._from_vertex != other._from_vertex:
            return False
        if self._to_vertex != other._to_vertex:
            return False
        if self._label != other._label:
            return False
        for k,v in self._properties.items():
            if k not in other._properties:
                return False
            if v!= other._properties[k]:
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
        query = neo4j.Query("CALL apoc.export.json.all(null,{useTypes:true, stream:true}) YIELD data RETURN data;")
        result = session.run(query).values()
        
    
        res_str = re.sub(r"\\n", ",\n", str(result[0]))
        res_str = re.sub(r"'", "", res_str)
            

        return  json.loads(res_str)

        
def get_memgraph_data_json_format(memgraph: gqlalchemy.Memgraph):
    result = list(
        memgraph.execute_and_fetch(
            f"""
            CALL export_util.json_stream() YIELD stream RETURN stream;
            """
        )
    )[0]["stream"]
    return json.loads(result)

   
def create_graph_memgraph_json(json_memgraph_data)->Graph:
    print(json_memgraph_data)
    graph = Graph()
    vertices_mapings = {}
    for item in json_memgraph_data:
        if item["type"] == "node":
            print(item)
            assert item["properties"]["id"] is not None, "Vertex doesn't have ID property"
            graph.add_vertex(Vertex(item["properties"]["id"], item["labels"], item["properties"]))
            vertices_mapings[item["id"]] = item["properties"]["id"]
        else:
            from_vertex_id = vertices_mapings[item["start"]]
            to_vertex_id = vertices_mapings[item["end"]]
            graph.add_edge(Edge(from_vertex_id, to_vertex_id, item["label"], item["properties"]))

    graph.vertices.sort(key=lambda vertex: vertex.id)
    graph.edges.sort(key = lambda edge: (edge.from_vertex, edge.to_vertex))
    return graph

def create_graph_neo4j_json(json_neo4j_data) -> Graph:
    print(json_neo4j_data)
    graph = Graph()
    vertices_mapings = {}

    for item in json_neo4j_data:
        if item["type"] == "node":
            print(item)
            assert item["properties"]["id"] is not None, "Vertex doesn't have ID property"
            vertices_mapings[item["id"]] = item["properties"]["id"]
            graph.add_vertex(Vertex(item["properties"]["id"],item["labels"], item["properties"]))
        else:
            from_vertex_id = vertices_mapings[item["start"]["id"]]
            to_vertex_id = vertices_mapings[item["end"]["id"]]
            graph.add_edge(Edge(from_vertex_id, to_vertex_id, item["label"], item["properties"]))
    graph.vertices.sort(key=lambda vertex: vertex.id)
    graph.edges.sort(key = lambda edge: (edge.from_vertex, edge.to_vertex))
    return graph



def create_neo4j_driver(port:int) -> neo4j.BoltDriver:
    return neo4j.GraphDatabase.driver(f"bolt://localhost:{port}", encrypted=False)  

def create_memgraph_db(port:int) -> gqlalchemy.Memgraph:
    return gqlalchemy.Memgraph("localhost", port) 


def mg_execute_cyphers(input_cyphers: List[str], db: gqlalchemy.Memgraph):
    """
    Execute commands against Memgraph
    """
    for query in input_cyphers:
        db.execute(query)

def neo4j_execute_cyphers(input_cyphers: List[str], neo4j_driver: neo4j.BoltDriver):
    """
    Execute commands against Neo4j
    """
    with neo4j_driver.session() as session:
        for text_query in input_cyphers:
            query = neo4j.Query(text_query)
            session.run(query).values()

def run_memgraph_query(query: str, db: gqlalchemy.Memgraph):
    """
    Execute command against Memgraph
    """
    db.execute(query)

def run_neo4j_query(query: str, neo4j_driver: neo4j.BoltDriver):
    """
    Execute command against Neo4j
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
    print("getting data from memgraph")
    json_data = get_memgraph_data_json_format(memgraph_db)
    print("memgraph json data", json_data)
    return create_graph_memgraph_json(json_data)

def neo4j_get_graph(neo4j_driver: neo4j.BoltDriver) -> Graph:
    json_data = get_neo4j_data_json(neo4j_driver)
    print("neo4j json data", json_data)
    return create_graph_neo4j_json(json_data)