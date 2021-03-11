import math
import random
import logging
import multiprocessing as mp
from typing import Dict, Any, List
from graph_coloring_module.graph import Graph
from graph_coloring_module.components.population import Population
from graph_coloring_module.framework.parameters_utils import param_value
from graph_coloring_module.utils.validation import validate
from graph_coloring_module.algorithms.meta_heuristics.parallel_algorithm import ParallelAlgorithm


logger = logging.getLogger('telco')


class QA(ParallelAlgorithm):
    """A class that represents quantum annealing algorithm."""

    def __str__(self):
        return "QA"

    @validate("max_iterations", "error", "communication_delay")
    def algorithm(
            self,
            proc_id: int,
            graph: Graph,
            population: Population,
            my_q: mp.Queue,
            prev_q: mp.Queue,
            next_q: mp.Queue,
            results: mp.Queue,
            parameters: Dict[str, Any]) -> None:
        """A function that executes a QA algorithm. The resulting population
        is written to the queue results."""

        max_iterations = param_value(graph, parameters, "max_iterations")
        error = param_value(graph, parameters, "error")
        communication_delay = param_value(graph, parameters, "communication_delay")
        logging_delay = param_value(graph, parameters, "logging_delay")
        convergence_tolerance = param_value(graph, parameters, "convergence_tolerance")
        convergence_probability = param_value(graph, parameters, "convergence_probability")
        best_sol_error = population.solution_error()
        iterations_passed = 0
        for iteration in range(max_iterations):
            if population.contains_solution or self._read_msgs(my_q, population):
                self._write_msgs(communication_delay, iteration, next_q, prev_q, population.solution(), True)
                break

            for i in range(len(population)):
                self._markow_chain(graph, population, i, parameters)
                self._write_msgs(communication_delay, iteration, next_q, prev_q, population[i])

                if iterations_passed == convergence_tolerance:
                    if random.random() < convergence_probability:
                        self._tunneling(i, graph, population, parameters)

            if iterations_passed == convergence_tolerance:
                best_sol_error = population.min_error(error.individual_err)
                iterations_passed = 0

            if best_sol_error <= population.min_error(error.individual_err):
                iterations_passed += 1
            else:
                best_sol_error = population.min_error(error.individual_err)
                iterations_passed = 0

            if iteration % logging_delay == 0:
                logger.info('Id: {} Iteration: {} Error: {}'.format(
                            proc_id, iteration, population.min_error(error.individual_err)))

        logger.info('Id: {} Iteration: {} Error: {}'.format(proc_id, iteration, population.solution_error()))
        results.put(population)

    @validate("temperature", "max_steps", "mutation", "error")
    def _markow_chain(
            self,
            graph: Graph,
            population: Population,
            ind: int,
            parameters: Dict[str, Any]) -> None:

        temperature = param_value(graph, parameters, "temperature")
        max_steps = param_value(graph, parameters, "max_steps")
        mutation = param_value(graph, parameters, "mutation")
        error = param_value(graph, parameters, "error")

        for _ in range(max_steps):
            indv = population[ind]
            new_indv, diff_nodes = mutation.mutate(graph, indv, parameters)
            delta_h_pot = error.individual_err(graph, new_indv) - error.individual_err(graph, indv)
            delta_corr = population.set_individual(ind, new_indv, diff_nodes)
            delta_error = error.delta(graph, indv, new_indv, delta_corr, parameters)

            if delta_h_pot > 0 or delta_error > 0:
                try:
                    probability = 1 - math.exp((-1 * delta_error) / temperature)
                except OverflowError:
                    probability = 1
                if random.random() <= probability:
                    population.set_individual(ind, indv, diff_nodes)

    @validate("max_attempts_tunneling", "mutation_tunneling")
    def _tunneling(
            self,
            ind: int,
            graph: Graph,
            population: Population,
            parameters: Dict[str, Any] = None) -> None:

        max_attempts_tunneling = param_value(graph, parameters, "max_attempts_tunneling")
        mutation = param_value(graph, parameters, "mutation_tunneling")

        old_indv = population.individuals[ind]
        old_indv_error = old_indv.conflicts_weight

        new_indv, diff_nodes = mutation.mutate(graph, old_indv, parameters)
        population.set_individual(ind, new_indv, diff_nodes)
        self._markow_chain(graph, population, ind, parameters)

        counter = 0
        while not (population.individuals[ind].conflicts_weight <= 2 * old_indv_error):
            population.set_individual(ind, old_indv, diff_nodes)
            if counter == max_attempts_tunneling:
                new_indv, _ = mutation.mutate(graph, population.best_individuals[ind], parameters)
                diff_nodes = self._find_diff_nodes(old_indv, new_indv)
                population.set_individual(ind, new_indv, diff_nodes)
                self._markow_chain(graph, population, ind, parameters)
                break

            counter += 1
            new_indv, diff_nodes = mutation.mutate(graph, old_indv, parameters)
            population.set_individual(ind, new_indv, diff_nodes)
            self._markow_chain(graph, population, ind, parameters)

    def _tunneling_seq(
            self,
            ind: int,
            graph: Graph,
            population: Population,
            parameters: Dict[str, Any] = None) -> None:
        pass

    def _find_diff_nodes(self, old_indv, new_indv) -> List[int]:
        nodes = []
        for i in range(len(old_indv.chromosome)):
            if old_indv[i] != new_indv[i]:
                nodes.append(i)
        return nodes
