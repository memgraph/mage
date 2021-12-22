import mgp
import pandas
import sys

try:
    import cugraph
except ImportError as import_error:
    sys.stderr.write(
        "RAPIDS cuGraph is not installed, please build Memgraph from source with Python Conda interpreter where cuGraph is installed."
    )
    raise import_error


class GraphDefinition:
    SOURCE = "source"
    TARGET = "target"

    VERTEX = "vertex"
    PAGERANK = "pagerank"


def _get_mg_data(mg_graph: mgp.Graph) -> pandas.DataFrame:
    graph_data = {GraphDefinition.SOURCE: [], GraphDefinition.TARGET: []}
    for from_vertex in mg_graph.vertices:
        for edge in from_vertex.out_edges:
            to_vertex = edge.to_vertex
            graph_data[GraphDefinition.SOURCE].append(from_vertex.id)
            graph_data[GraphDefinition.TARGET].append(to_vertex.id)

    df = pandas.DataFrame(graph_data)
    return df


@mgp.read_proc
def pagerank(
    context: mgp.ProcCtx,
) -> mgp.Record(vertex_id=mgp.Vertex, pagerank=mgp.Number):
    df = _get_mg_data(context.graph)
    G = cugraph.Graph()

    # Convert Pandas DataFrame to cugraph Graph
    G.from_pandas_edgelist(
        df,
        source=GraphDefinition.SOURCE,
        destination=GraphDefinition.TARGET,
        renumber=True,
    )
    # Call PageRank on NVIDIA graphic card and store results in Memgraph
    df_page = cugraph.pagerank(G)
    return [
        mgp.Record(
            vertex_id=context.graph.get_vertex_by_id(int(row[GraphDefinition.VERTEX])),
            pagerank=float(row[GraphDefinition.PAGERANK]),
        )
        for _, row in df_page.to_pandas().iterrows()
    ]
