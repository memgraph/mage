import datetime
from typing import Any

import mgp
import gqlalchemy

WRONG_STRUCTURE_MSG = "The `do_case` argument named `conditionals` must be structured as follows: [BOOLEAN, STRING, BOOLEAN, STRING, …​]."

def get_edge_by_id(
    graph: mgp.Graph, edge_id: mgp.EdgeId, from_id: mgp.VertexId
) -> mgp.Nullable[mgp.Edge]:
    return next(
        (
            edge
            for edge in graph.get_vertex_by_id(from_id).out_edges
            if edge.id == edge_id
        ),
        None,
    )


def gqlalchemy_type_to_mgp(graph: mgp.Graph, variable: Any) -> mgp.Any:
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
        return get_edge_by_id(
            graph, edge_id=int(variable._id), from_id=int(variable._start_node_id)
        )

    elif isinstance(variable, gqlalchemy.models.Path):
        start_id = int(variable._nodes[0]._id)
        path = mgp.Path(graph.get_vertex_by_id(start_id))

        for relationship, node in zip(variable._relationships, variable._nodes[1:]):
            next_edge_id = int(relationship._id)
            next_node_id = int(node._id)

            edge_to_add = get_edge_by_id(graph, edge_id=next_edge_id, from_id=start_id)
            if edge_to_add is None:
                # This should happen if and only if the query that returned the path treated graph edges as undirected.
                # For example, such a query might return an (a)-[b]->(c) path from a graph that only contains (c)->[b].
                # In that case, flipping the path direction to retrieve the linking edge is valid.
                edge_to_add = get_edge_by_id(
                    graph, edge_id=next_edge_id, from_id=next_node_id
                )

            path.expand(edge_to_add)

            start_id = next_node_id

        return path


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
        params: {key: value} map of parameters applied to the given queries.

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

    memgraph = gqlalchemy.Memgraph()

    for condition, if_query in zip(conditions, if_queries):
        if condition:
            results = memgraph.execute_and_fetch(if_query, parameters=params)

            return [
                mgp.Record(
                    value={
                        field_name: gqlalchemy_type_to_mgp(ctx.graph, field_value)
                        for field_name, field_value in result.items()
                    }
                )
                for result in results
            ]

    results = memgraph.execute_and_fetch(else_query, parameters=params)

    return [
        mgp.Record(
            value={
                field_name: gqlalchemy_type_to_mgp(ctx.graph, field_value)
                for field_name, field_value in result.items()
            }
        )
        for result in results
    ]


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

    if params is None:
        params = {}

    memgraph = gqlalchemy.Memgraph()

    results = memgraph.execute_and_fetch(
        if_query if condition else else_query, parameters=params
    )

    return [
        mgp.Record(
            value={
                field_name: gqlalchemy_type_to_mgp(ctx.graph, field_value)
                for field_name, field_value in result.items()
            }
        )
        for result in results
    ]
