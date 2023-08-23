import csv
import io
import json as js
import mgp

from dataclasses import dataclass
from datetime import datetime, date, time, timedelta
from gqlalchemy import Memgraph
from typing import Any, Dict, List, Union

from mage.export_import_util.parameters import Parameter


@dataclass
class Node:
    id: int
    labels: list
    properties: dict

    def get_dict(self) -> dict:
        return {
            Parameter.ID.value: self.id,
            Parameter.LABELS.value: self.labels,
            Parameter.PROPERTIES.value: self.properties,
            Parameter.TYPE.value: Parameter.NODE.value,
        }


@dataclass
class Relationship:
    end: int
    id: int
    label: str
    properties: dict
    start: int

    def get_dict(self) -> dict:
        return {
            Parameter.END.value: self.end,
            Parameter.ID.value: self.id,
            Parameter.LABEL.value: self.label,
            Parameter.PROPERTIES.value: self.properties,
            Parameter.START.value: self.start,
            Parameter.TYPE.value: Parameter.RELATIONSHIP.value,
        }


def convert_to_isoformat(
    property: Union[
        None,
        str,
        bool,
        int,
        float,
        List[Any],
        Dict[str, Any],
        timedelta,
        time,
        datetime,
        date,
    ]
):
    if isinstance(property, timedelta):
        return Parameter.DURATION.value + str(property) + ")"

    elif isinstance(property, time):
        return Parameter.LOCALTIME.value + property.isoformat() + ")"

    elif isinstance(property, datetime):
        return Parameter.LOCALDATETIME.value + property.isoformat() + ")"

    elif isinstance(property, date):
        return Parameter.DATE.value + property.isoformat() + ")"

    else:
        return property


import gqlalchemy
from gqlalchemy.connection import _convert_memgraph_value

WRONG_STRUCTURE_MSG = "The `conditionals` parameter of `do.case` must be structured as follows: [BOOLEAN, STRING, BOOLEAN, STRING, …​]."

def _get_edge_with_endpoint(
    graph: mgp.Graph,
    edge_id: mgp.EdgeId,
    from_id: mgp.VertexId,
) -> mgp.Nullable[mgp.Edge]:
    return next(
        (
            edge
            for edge in graph.get_vertex_by_id(from_id).out_edges
            if edge.id == edge_id
        ),
        None,
    )


def _gqlalchemy_type_to_mgp(graph: mgp.Graph, variable: Any) -> mgp.Any:
    if isinstance(
        variable,
        (
            bool,
            int,
            float,
            str,
            list,
            dict,
            datetime.date,
            datetime.time,
            datetime.datetime,
            datetime.timedelta,
        ),
    ):
        return variable

    elif isinstance(variable, gqlalchemy.models.Node):
        return graph.get_vertex_by_id(int(variable._id))

    elif isinstance(variable, gqlalchemy.models.Relationship):
        return _get_edge_with_endpoint(
            graph, edge_id=int(variable._id), from_id=int(variable._start_node_id)
        )

    elif isinstance(variable, gqlalchemy.models.Path):
        start_id = int(variable._nodes[0]._id)
        path = mgp.Path(graph.get_vertex_by_id(start_id))

        for relationship, node in zip(variable._relationships, variable._nodes[1:]):
            next_edge_id = int(relationship._id)
            next_node_id = int(node._id)

            edge_to_add = _get_edge_with_endpoint(
                graph, edge_id=next_edge_id, from_id=start_id
            )
            if edge_to_add is None:
                # This should happen if and only if the query that returned the path treated graph edges as undirected.
                # For example, such a query might return an (a)-[b]->(c) path from a graph that only contains (c)->[b].
                # In that case, flipping the path direction to retrieve the linking edge is valid.
                edge_to_add = _get_edge_with_endpoint(
                    graph, edge_id=next_edge_id, from_id=next_node_id
                )

            path.expand(edge_to_add)

            start_id = next_node_id

        return path


# Conditional queries are run via gqlalchemy. The module supports parameterized
# queries, which are only supported from gqlalchemy 1.4.0. Since that version’s
# dgl requirement causes arm64 checks for MAGE to fail, this function serves to
# support running parameterized queries with older gqlalchemy versions.
#
# Relevant code:
# https://github.com/memgraph/gqlalchemy/blob/main/gqlalchemy/vendors/database_client.py#L54-L59
# https://github.com/memgraph/gqlalchemy/blob/main/gqlalchemy/connection.py#L87-L100
def _execute_and_fetch_parameterized(
    memgraph_client, query: str, parameters: Dict[str, Any] = {}
) -> Iterator[Dict[str, Any]]:
    cursor = memgraph_client._get_cached_connection()._connection.cursor()
    cursor.execute(query, parameters)
    while True:
        row = cursor.fetchone()
        if row is None:
            break
        yield {
            dsc.name: _convert_memgraph_value(row[index])
            for index, dsc in enumerate(cursor.description)
        }


def _convert_results(
    ctx: mgp.ProcCtx, gqlalchemy_results: Iterator[Dict[str, Any]]
) -> List[mgp.Record]:
    return [
        mgp.Record(
            value={
                field_name: _gqlalchemy_type_to_mgp(ctx.graph, field_value)
                for field_name, field_value in result.items()
            }
        )
        for result in gqlalchemy_results
    ]


@mgp.read_proc
def cypher_all(
    ctx: mgp.ProcCtx,
    file: str,
    config: mgp.Map = {},
) -> mgp.Record(file_path=str, data=str):
    """Exports the graph in cypher with all the constraints, indexes and triggers.

    Args:
        conditionals: List of condition & read-only query pairs structured as [condition, query, condition, query, …​].
                      Conditions are boolean values and queries are strings.
        else_query: The read-only query to be executed if no condition evaluates to true.
        params: If the above queries are parameterized, provide a {key: value} map of parameters applied to the given queries.

    Returns:
        value: {field_name: field_value} map containing the result records of the evaluated query.
    """
    
    #memgraph = gqlalchemy.Memgraph()
    #results = memgraph.execute_and_fetch("SHOW TRIGGERS;")
    #print(results)
    #print(_convert_results(results))

    return mgp.Record(file_path=file_path, data=data)


def get_graph(ctx: mgp.ProcCtx) -> List[Union[Node, Relationship]]:
    nodes = list()
    relationships = list()

    for vertex in ctx.graph.vertices:
        labels = [label.name for label in vertex.labels]
        properties = {
            key: convert_to_isoformat(vertex.properties.get(key))
            for key in vertex.properties.keys()
        }

        nodes.append(Node(vertex.id, labels, properties).get_dict())

        for edge in vertex.out_edges:
            properties = {
                key: convert_to_isoformat(edge.properties.get(key))
                for key in edge.properties.keys()
            }

            relationships.append(
                Relationship(
                    edge.to_vertex.id,
                    edge.id,
                    edge.type.name,
                    properties,
                    edge.from_vertex.id,
                ).get_dict()
            )

    return nodes + relationships


@mgp.read_proc
def json(ctx: mgp.ProcCtx, path: str) -> mgp.Record():
    """
    Procedure to export the whole database to a JSON file.

    Parameters
    ----------
    path : str
        Path to the JSON file containing the exported graph database.
    """

    graph = get_graph(ctx)
    try:
        with open(path, "w") as outfile:
            js.dump(graph, outfile, indent=Parameter.STANDARD_INDENT.value, default=str)
    except PermissionError:
        raise PermissionError(
            "You don't have permissions to write into that file. Make sure to give the necessary permissions to user memgraph."
        )
    except Exception:
        raise OSError("Could not open or write to the file.")

    return mgp.Record()


@mgp.read_proc
def json_stream(ctx: mgp.ProcCtx) -> mgp.Record(stream=str):
    """
    Procedure to export the whole database to a stream.
    """
    return mgp.Record(stream=js.dumps(get_graph(ctx)))


def save_file(file_path: str, data_list: list):
    try:
        with open(
            file_path,
            "w",
            newline="",
            encoding="utf8",
        ) as f:
            writer = csv.writer(f)
            writer.writerows(data_list)
    except PermissionError:
        raise PermissionError(
            "You don't have permissions to write into that file. Make sure to give the necessary permissions to user memgraph."
        )
    except csv.Error as e:
        raise csv.Error(
            "Could not write to the file {}, stopped at line {}: {}".format(
                file_path, writer.line_num, e
            )
        )
    except Exception:
        raise OSError("Could not open or write to the file.")


def csv_to_stream(data_list: list) -> str:
    output = io.StringIO()
    try:
        writer = csv.writer(output, quoting=csv.QUOTE_NONNUMERIC)
        writer.writerows(data_list)
    except csv.Error as e:
        raise csv.Error(
            "Could not write a stream, stopped at line {}: {}".format(
                writer.line_num, e
            )
        )
    return output.getvalue()


@mgp.read_proc
def csv_query(
    context: mgp.ProcCtx,
    query: str,
    file_path: str = "",
    stream: bool = False,
) -> mgp.Record(file_path=str, data=str):
    """
    Procedure to export query results to a CSV file.
    Args:
        context (mgp.ProcCtx): Reference to the context execution.
        query (str): A query from which the results will be saved to a CSV file.
        file_path (str, optional): A path to the CSV file where the query results will be exported. Defaults to an empty string.
        stream (bool, optional): A value which determines whether a stream of query results in a CSV format will be returned.
    Returns:
        mgp.Record(
            file_path (str): A path to the CSV file where the query results are exported. If file_path is not provided, the output will be an empty string.
            data (str): A stream of query results in a CSV format.
        )
    Raises:
        Exception: If neither file nor config are provided, or if only config is provided with stream set to False. Also if query yields no results or if the database is empty.
        PermissionError: If you provided file path that you have no permissions to write at.
        csv.Error: If an error occurred while writing into stream or CSV file.
        OSError: If the file can't be opened or written to.
    """

    # file or config have to be provided
    if not file_path and not stream:
        raise Exception("Please provide file name and/or config.")

    # only config provided with stream set to false
    if not file_path and not stream:
        raise Exception(
            "If you provided only stream value, it has to be set to True to get any results."
        )

    memgraph = Memgraph()
    results = list(memgraph.execute_and_fetch(query))

    # if query yields no result
    if not len(results):
        raise Exception(
            "Your query yields no results. Check if the database is empty or rewrite the provided query."
        )

    result_keys = list(results[0])
    data_list = [result_keys] + [list(result.values()) for result in results]
    data = ""

    if file_path:
        save_file(file_path, data_list)

    if stream:
        data = csv_to_stream(data_list)

    return mgp.Record(file_path=file_path, data=data)
