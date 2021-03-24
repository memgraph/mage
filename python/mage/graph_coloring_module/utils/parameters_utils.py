import types
from typing import Dict, Any
from mage.graph_coloring_module.graph import Graph


def param_value(
    graph: Graph, parameters: Dict[str, Any], param: str, init_value: Any = None
) -> Any:

    if parameters is None:
        if init_value is None:
            return None
        return init_value

    param = parameters.get(param)

    if param is None:
        if init_value is None:
            return None
        return init_value

    if isinstance(param, types.FunctionType):
        param = param(graph)

    return param
