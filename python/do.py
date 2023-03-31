import datetime
from typing import Any, Dict, List, Iterator

import mgp
import gqlalchemy
from gqlalchemy.connection import _convert_memgraph_value

import utils.subquery

WRONG_STRUCTURE_MSG = "The `conditionals` parameter of `do.case` must be structured as follows: [BOOLEAN, STRING, BOOLEAN, STRING, …​]."
DISALLOWED_QUERY_MSG = 'The query "{query}" isn’t supported by `{procedure}` because it would execute a global operation.'


@mgp.read_proc
def case(
    ctx: mgp.ProcCtx,
    conditionals: mgp.List[mgp.Any],
    else_query: mgp.Nullable[str] = "",
    params: mgp.Nullable[mgp.Map] = None,
) -> mgp.Record(value=mgp.Nullable[mgp.Map]):
    """Given a list of condition & (read-only) query pairs, executes the query associated with the first condition
       evaluating to true (or the else query if none are true) with the given parameters.

    Args:
        conditionals: List of condition & read-only query pairs structured as [condition, query, condition, query, …​].
                      Conditions are boolean values and queries are strings.
        else_query: The read-only query to be executed if no condition evaluates to true.
        params: If the above queries are parameterized, provide a {key: value} map of parameters applied to the given queries.

    Returns:
        value: {field_name: field_value} map containing the result records of the evaluated query.
    """

    if params is None:
        params = {}

    conditions = conditionals[::2]
    if_queries = conditionals[1::2]

    if len(conditions) != len(if_queries):
        raise ValueError(WRONG_STRUCTURE_MSG)

    if any(not isinstance(condition, bool) for condition in conditions) or any(
        not isinstance(if_query, str) for if_query in if_queries
    ):
        raise ValueError(WRONG_STRUCTURE_MSG)

    for query in if_queries + (else_query,):
        if utils.subquery.is_global_operation(query):
            raise ValueError(
                DISALLOWED_QUERY_MSG.format(query=query, procedure="do.case")
            )

    memgraph = gqlalchemy.Memgraph()

    for condition, if_query in zip(conditions, if_queries):
        if condition:
            results = _execute_and_fetch_parameterized(
                memgraph, if_query, parameters=params
            )

            return _convert_results(ctx, results)

    results = _execute_and_fetch_parameterized(memgraph, else_query, parameters=params)

    return _convert_results(ctx, results)


@mgp.read_proc
def when(
    ctx: mgp.ProcCtx,
    condition: bool,
    if_query: str,
    else_query: mgp.Nullable[str] = "",
    params: mgp.Nullable[mgp.Map] = None,
) -> mgp.Record(value=mgp.Nullable[mgp.Map]):
    """Depending on the value of the condition, executes `if_query` or `else_query` (read-only).

    Args:
        condition: The boolean value that determines what query to execute.
        if_query: The read-only query to be executed if the condition is satisfied.
        else_query: The read-only query to be executed if the condition is not satisfied.
        params: {key: value} map of parameters applied to the given queries.

    Returns:
        value: {field_name: field_value} map containing the result records of the evaluated query.
    """

    for query in (if_query, else_query):
        if utils.subquery.is_global_operation(query):
            raise ValueError(
                DISALLOWED_QUERY_MSG.format(query=query, procedure="do.when")
            )

    if params is None:
        params = {}

    memgraph = gqlalchemy.Memgraph()

    results = _execute_and_fetch_parameterized(
        memgraph, if_query if condition else else_query, parameters=params
    )

    return _convert_results(ctx, results)


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
