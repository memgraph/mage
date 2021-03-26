import re
import logging
from typing import Dict, Any
from mage.graph_coloring_module import QA
from mage.graph_coloring_module import ConflictError
from mage.graph_coloring_module import SimpleMutation
from mage.graph_coloring_module import MultipleMutation
from mage.graph_coloring_module import LDO
from mage.graph_coloring_module import SDO
from mage.graph_coloring_module import Random
from mage.graph_coloring_module import Graph


logging.basicConfig(format="%(asctime)-15s [%(levelname)s]: %(message)s")
logger = logging.getLogger("telco")
logger.setLevel(logging.INFO)


def _demo():
    graph = Graph(
        [0, 1, 2, 3],
        {
            0: [(1, 2), (2, 3)],
            1: [(0, 2), (2, 2)],
            2: [(0, 3), (1, 2), (3, 3)],
            3: [(2, 3)],
        },
    )
    alg = QA()
    _default_parameters = {
        "no_of_processes": 5,
        "no_of_chunks": 5,
        "communication_delay": 10,
        "max_iterations": 100,
        "population_size": 10,
        "no_of_colors": 3,
        "max_steps": 10,
        "temperature": 0.025,
        "mutation": SimpleMutation(),
        "error": ConflictError(),
        "alpha": 0.1,
        "beta": 0.001,
        "algorithms": [Random(), SDO(), LDO()],
        "logging_delay": 1,
        "convergence_tolerance": 10000,
        "convergence_probability": 0.5,
        "max_attempts_tunneling": 25,
        "mutation_tunneling": MultipleMutation(),
        "multiple_mutation_no_of_nodes": 3,
        "random_mutation_probability": 0.1,
        "iteration_callbacks": [],
    }
    sol = alg.run(graph, _default_parameters)
    print(sol.chromosome)


_demo()
