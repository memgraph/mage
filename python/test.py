from collections import defaultdict
from typing import Optional, Dict, Any
import mage.graph_coloring_module
from mage.graph_coloring_module import Parameter
from mage.graph_coloring_module import Graph


def _str2Class(name: str):
    return getattr(mage.graph_coloring_module, name)


def _map_parameters(params: Dict[str, Any]) -> Dict[str, Any]:
    for key in params:
        if isinstance(params[key], str):
            params[key] = _str2Class(params[key])()
        if isinstance(params[key], list):
            new_list = []
            for val in params[key]:
                if isinstance(val, str):
                    new_list.append(_str2Class(val)())
                else:
                    new_list.append(val)
            params[key] = new_list
    return params


def _get_parameters(parameters: Dict[str, Any]) -> Dict[str, Any]:
    params = _map_parameters(
        {
            Parameter.ALGORITHM: parameters.get(Parameter.ALGORITHM.value, "QA"),
            Parameter.NO_OF_COLORS: parameters.get(Parameter.NO_OF_COLORS.value, 1),
            Parameter.NO_OF_PROCESSES: parameters.get(
                Parameter.NO_OF_PROCESSES.value, 2
            ),
            Parameter.NO_OF_CHUNKS: parameters.get(Parameter.NO_OF_CHUNKS.value, 2),
            Parameter.POPULATION_SIZE: parameters.get(
                Parameter.POPULATION_SIZE.value, 10
            ),
            Parameter.INIT_ALGORITHMS: parameters.get(
                Parameter.INIT_ALGORITHMS.value, ["SDO", "LDO"]
            ),
            Parameter.POPULATION_FACTORY: parameters.get(
                Parameter.POPULATION_FACTORY.value, "ChainChunkFactory"
            ),
            Parameter.ERROR: parameters.get(Parameter.ERROR.value, "ConflictError"),
            Parameter.MAX_ITERATIONS: parameters.get(
                Parameter.MAX_ITERATIONS.value, 10000
            ),
            Parameter.ITERATION_CALLBACKS: parameters.get(
                Parameter.ITERATION_CALLBACKS.value, []
            ),
            Parameter.COMMUNICATION_DALAY: parameters.get(
                Parameter.COMMUNICATION_DALAY.value, 10
            ),
            Parameter.LOGGING_DELAY: parameters.get(Parameter.LOGGING_DELAY.value, 1),
            Parameter.QA_TEMPERATURE: parameters.get(Parameter.QA_TEMPERATURE.value, 1),
            Parameter.QA_MAX_STEPS: parameters.get(Parameter.QA_MAX_STEPS.value, 10),
            Parameter.CONFLICT_ERR_ALPHA: parameters.get(
                Parameter.CONFLICT_ERR_ALPHA.value, 1
            ),
            Parameter.CONFLICT_ERR_BETA: parameters.get(
                Parameter.CONFLICT_ERR_BETA.value, 5
            ),
            Parameter.MUTATION: parameters.get(
                Parameter.MUTATION.value, "SimpleMutation"
            ),
            Parameter.MULTIPLE_MUTATION_NODES_NO_OF_NODES: parameters.get(
                Parameter.MULTIPLE_MUTATION_NODES_NO_OF_NODES.value, 2
            ),
            Parameter.RANDOM_MUTATION_PROBABILITY: parameters.get(
                Parameter.RANDOM_MUTATION_PROBABILITY.value, 0.1
            ),
            Parameter.SIMPLE_TUNNELING_MUTATION: parameters.get(
                Parameter.SIMPLE_TUNNELING_MUTATION.value, "MultipleMutation"
            ),
            Parameter.SIMPLE_TUNNELING_PROBABILITY: parameters.get(
                Parameter.SIMPLE_TUNNELING_PROBABILITY.value, 0.5
            ),
            Parameter.SIMPLE_TUNNELING_ERROR_CORRECTION: parameters.get(
                Parameter.SIMPLE_TUNNELING_ERROR_CORRECTION.value, 2
            ),
            Parameter.SIMPLE_TUNNELING_MAX_ATTEMPTS: parameters.get(
                Parameter.SIMPLE_TUNNELING_MAX_ATTEMPTS.value, 25
            ),
            Parameter.CONVERGENCE_CALLBACK_TOLERANCE: parameters.get(
                Parameter.CONVERGENCE_CALLBACK_TOLERANCE.value, 100
            ),
            Parameter.CONVERGENCE_CALLBACK_ACTIONS: parameters.get(
                Parameter.CONVERGENCE_CALLBACK_ACTIONS.value, ["SimpleTunneling"]
            ),
        }
    )
    return params


params = _get_parameters(dict())
g = Graph(
    [0, 1, 2, 3, 4],
    {
        0: [(1, 2), (2, 3)],
        1: [(0, 2), (2, 2), (4, 5)],
        2: [(0, 3), (1, 2), (3, 3)],
        3: [(2, 3)],
        4: [(1, 5)],
    },
)
alg = params[Parameter.ALGORITHM]
sol_indv = alg.run(g, params)
sol = sol_indv.chromosome
print(sol)
