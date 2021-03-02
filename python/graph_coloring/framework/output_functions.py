from typing import Dict, Any
from telco.components.individual import Individual
from telco.graph import Graph
from telco.framework.parameters_utils import param_value


def _write_dict_to_file(
        graph: Graph,
        params: Dict[str, Any],
        filename: str,
        indentation: str = "") -> None:

    writer = open(filename, "a+")
    for key, value in params.items():
        if isinstance(value, dict):
            writer.write(indentation + key + ": \n")
            writer.close()
            _write_dict_to_file(graph, value, filename, indentation + "   ")
            writer = open(filename, "a+")
        else:
            value = param_value(graph, params, key)
            if isinstance(value, list):
                value = [str(val) for val in value]
            writer.write(indentation + key + ": " + str(value) + "\n")
    writer.close()


def write_to_file(
        params: Dict[str, Any],
        sol: Individual,
        graph: Graph,
        filename: str) -> None:

    writer = open(filename, "a+")
    writer.write("PARAMETERS\n")
    writer.close()
    _write_dict_to_file(graph, params, filename)

    writer = open(filename, "a+")
    writer.write("\n")

    writer.write("ERROR\n")
    error = param_value(graph, params, "error")
    indv_error_params = param_value(graph, params, "indv_error_parameters")
    writer.write(str(error.individual_err(graph, sol, indv_error_params)) + "\n\n")

    writer.write("ASSIGNMENT\n")
    for node in graph.nodes:
        writer.write(str(graph.label(node)) + ": " + str(sol[node]) + "\n")
    writer.close()
