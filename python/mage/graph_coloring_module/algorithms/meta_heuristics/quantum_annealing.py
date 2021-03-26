import math
import random
import logging
import multiprocessing as mp
from typing import Dict, Any
from mage.graph_coloring_module.graph import Graph
from mage.graph_coloring_module.components.population import Population
from mage.graph_coloring_module.utils.parameters_utils import param_value
from mage.graph_coloring_module.utils.validation import validate
from mage.graph_coloring_module.algorithms.meta_heuristics.parallel_algorithm import (
    ParallelAlgorithm,
)


logger = logging.getLogger("telco")


class ConvergenceCallback:
    def __init__(self):
        self._iteration = 0
        self._best_sol_error = float("inf")

    @validate("error", "convergence_tolerance")
    def update(self, graph: Graph, population: Population, parameters: Dict[str, Any]):
        error = param_value(graph, parameters, "error")
        convergence_tolerance = param_value(graph, parameters, "convergence_tolerance")

        self._iteration += 1
        if population.min_error(error.individual_err) < self._best_sol_error:
            self._best_sol_error = population.min_error(error.individual_err)
            self._iteration = 0
        if self._iteration == convergence_tolerance:
            self._convergence_detected(graph, population, parameters)

    @validate("error", "convergence_callback_actions, convergence_probability")
    def _convergence_detected(
        self, graph: Graph, population: Population, parameters: Dict[str, Any]
    ):
        error = param_value(graph, parameters, "error")
        convergence_callback_actions = param_value(
            graph, parameters, "convergence_callback_actions"
        )
        convergence_probability = param_value(
            graph, parameters, "convergence_probability"
        )

        for action in convergence_callback_actions:
            for i, _ in enumerate(population.individuals):
                if random.random() < convergence_probability:
                    action.run(i, graph, population, parameters)
        self._iteration = 0
        self._best_sol_error = population.min_error(error.individual_err)


class MatplotlibCallback:
    def __init__(self):
        pass


class QA(ParallelAlgorithm):
    """A class that represents quantum annealing algorithm."""

    def __str__(self):
        return "QA"

    @validate("max_iterations", "error", "communication_delay, iteration_adapters")
    def algorithm(
        self,
        proc_id: int,
        graph: Graph,
        population: Population,
        my_q: mp.Queue,
        prev_q: mp.Queue,
        next_q: mp.Queue,
        results: mp.Queue,
        parameters: Dict[str, Any],
    ) -> None:
        """A function that executes a QA algorithm. The resulting population
        is written to the queue results."""

        max_iterations = param_value(graph, parameters, "max_iterations")
        error = param_value(graph, parameters, "error")
        communication_delay = param_value(graph, parameters, "communication_delay")
        logging_delay = param_value(graph, parameters, "logging_delay")
        iteration_callbacks = param_value(graph, parameters, "iteration_callbacks")

        for iteration in range(max_iterations):
            if population.contains_solution or self._read_msgs(my_q, population):
                self._write_stop(prev_q, next_q, population.solution())
                break

            for i in range(len(population)):
                self._markow_chain(graph, population, i, parameters)

            self._write_msg(communication_delay, iteration, prev_q, population[0], -1)
            self._write_msg(
                communication_delay,
                iteration,
                next_q,
                population[population.size() - 1],
                1,
            )

            for callback in iteration_callbacks:
                callback.update(graph, population, parameters)

            if iteration % logging_delay == 0:
                logger.info(
                    "Id: {} Iteration: {} Error: {}".format(
                        proc_id, iteration, population.min_error(error.individual_err)
                    )
                )

        logger.info(
            "Id: {} Iteration: {} Error: {}".format(
                proc_id, iteration, population.solution_error()
            )
        )
        results.put(population)

    @validate("temperature", "max_steps", "mutation", "error")
    def _markow_chain(
        self, graph: Graph, population: Population, ind: int, parameters: Dict[str, Any]
    ) -> None:

        temperature = param_value(graph, parameters, "temperature")
        max_steps = param_value(graph, parameters, "max_steps")
        mutation = param_value(graph, parameters, "mutation")
        error = param_value(graph, parameters, "error")

        for _ in range(max_steps):
            indv = population[ind]
            pop_error_old = error.population_err(graph, population, parameters)
            new_indv, diff_nodes = mutation.mutate(graph, indv, parameters)
            delta_h_pot = error.individual_err(graph, new_indv) - error.individual_err(
                graph, indv
            )
            population.set_individual(ind, new_indv, diff_nodes)
            pop_error_new = error.population_err(graph, population, parameters)
            delta_error = pop_error_new - pop_error_old

            if delta_h_pot > 0 or delta_error > 0:
                try:
                    probability = 1 - math.exp((-1 * delta_error) / temperature)
                except OverflowError:
                    probability = 1
                if random.random() <= probability:
                    population.set_individual(ind, indv, diff_nodes)
