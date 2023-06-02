import mgp

from typing import Any, Dict, List, Tuple

from gqlalchemy import Memgraph, Node, Relationship


class PeriodicIterateConstants:
    batch_size = "batch_size"
    batch_variable = "__batch"
    batch_row = "__batch_row"


@mgp.read_proc
def iterate(
    context: mgp.ProcCtx, input_query: str, running_query: str, config: mgp.Map
) -> mgp.Record(success=bool):
    if PeriodicIterateConstants.batch_size not in config:
        raise Exception(
            f"{PeriodicIterateConstants.batch_size} is not specified in periodic.iterate config property!"
        )

    batch_size = config[PeriodicIterateConstants.batch_size]
    if not isinstance(batch_size, int):
        raise Exception(
            f"Config parameter {PeriodicIterateConstants.batch_size} is not an integer!"
        )

    memgraph = Memgraph()

    input_results = memgraph.execute_and_fetch(input_query)
    input_results = list(input_results)

    if not input_results:
        return mgp.Record(success=True)

    offset = 0
    while True:
        start = offset
        end = (
            offset + batch_size
            if offset + batch_size <= len(input_results)
            else len(input_results)
        )

        input_results_batch = input_results[start:end]

        final_query, query_params = _prepare_query(input_results_batch, running_query)

        memgraph.execute(final_query, query_params)

        if end == len(input_results):
            break

        offset = end

    return mgp.Record(success=True)


def _prepare_query(
    input_results_batch: List, running_query: str
) -> Tuple[str, Dict[str, List[Dict[str, Any]]]]:
    unwind_query = _prepare_unwind_statement()
    nodes, relationships, primitives = _prepare_input_params(input_results_batch[0])
    with_scan_id = _prepare_with_scan_id_statement(nodes, relationships, primitives)

    final_query = f"{unwind_query} {with_scan_id} {running_query}"

    batch = _prepare_batch(input_results_batch, nodes, relationships, primitives)

    return final_query, batch


def _prepare_unwind_statement() -> str:
    return f"UNWIND ${PeriodicIterateConstants.batch_variable} AS {PeriodicIterateConstants.batch_row}"


def _prepare_with_scan_id_statement(nodes, relationships, primitives) -> str:
    with_statement = ""
    with_nodes_and_rels = [
        f"{PeriodicIterateConstants.batch_row}.{x} AS __{x}_id"
        for x in nodes + relationships
    ]
    with_primitives = [
        f"{PeriodicIterateConstants.batch_row}.{primitive_name} AS {primitive_name}"
        for primitive_name in primitives
    ]
    if len(with_nodes_and_rels) or len(with_primitives):
        with_statement = f"WITH {', '.join(with_nodes_and_rels + with_primitives)}"

    scan_nodes = " ".join(
        [
            f"MATCH ({node_name}) WHERE ID({node_name}) = __{node_name}_id"
            for node_name in nodes
        ]
    )
    scan_relationships = " ".join(
        [
            f"MATCH ()-[{relationship_name}]->() WHERE ID({relationship_name}) = __{relationship_name}_id"
            for relationship_name in relationships
        ]
    )

    return " ".join([with_statement, scan_nodes, scan_relationships])


def _prepare_input_params(input_result: Any) -> Tuple[List[str], List[str], List[str]]:
    nodes = []
    relationships = []
    primitives = []

    for name, value in input_result.items():
        if isinstance(value, Node):
            nodes.append(name)
        elif isinstance(value, Relationship):
            relationships.append(name)
        else:
            primitives.append(name)

    return nodes, relationships, primitives


def _prepare_batch(
    input_results, nodes, relationships, primitives
) -> Dict[str, List[Dict[str, Any]]]:
    new_input_results = []

    for batch_row in input_results:
        new_batch_entry = {}
        for name in batch_row:
            if name in nodes or name in relationships:
                new_batch_entry[name] = batch_row[name]._id
            else:
                new_batch_entry[name] = batch_row[name]

        new_input_results.append(new_batch_entry)

    return {PeriodicIterateConstants.batch_variable: new_input_results}
