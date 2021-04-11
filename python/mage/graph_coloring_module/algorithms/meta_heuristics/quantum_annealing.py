import math
import random
import logging
import multiprocessing as mp
from typing import Dict, Any
from mage.graph_coloring_module.parameters import Parameter
from mage.graph_coloring_module.graph import Graph
from mage.graph_coloring_module.components.population import Population
from mage.graph_coloring_module.utils.parameters_utils import param_value
from mage.graph_coloring_module.utils.validation import validate
from mage.graph_coloring_module.communication.message_type import MessageType
from mage.graph_coloring_module.algorithms.meta_heuristics.parallel_algorithm import (
    ParallelAlgorithm,
)


logger = logging.getLogger("graph_coloring")


class QA(ParallelAlgorithm):
    """A class that represents the quantum annealing algorithm."""

    def __str__(self):
        return "QA"

    @validate(Parameter.MAX_ITERATIONS, Parameter.ERROR, Parameter.COMMUNICATION_DALAY)
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
        """Function that executes the QA algorithm. The resulting population
        is written to the queue named results."""

        max_iterations = param_value(graph, parameters, Parameter.MAX_ITERATIONS)
        error = param_value(graph, parameters, Parameter.ERROR)
        communication_delay = param_value(
            graph, parameters, Parameter.COMMUNICATION_DALAY
        )
        logging_delay = param_value(graph, parameters, Parameter.LOGGING_DELAY)
        iteration_callbacks = param_value(
            graph, parameters, Parameter.ITERATION_CALLBACKS
        )
        no_of_chunks = param_value(graph, parameters, Parameter.NO_OF_CHUNKS)

        for iteration in range(max_iterations):
            flag, pid = self._read_msgs(my_q, population)
            if flag:
                if self._prev_chunk(proc_id, no_of_chunks) != pid:
                    self._write_msg(
                        communication_delay,
                        iteration,
                        prev_q,
                        population.best_individual(error.individual_err),
                        MessageType.STOP,
                        proc_id,
                    )
                if self._next_chunk(proc_id, no_of_chunks) != pid:
                    self._write_msg(
                        communication_delay,
                        iteration,
                        next_q,
                        population.best_individual(error.individual_err),
                        MessageType.STOP,
                        proc_id,
                    )
                break

            for i in range(len(population)):
                self._markow_chain(graph, population, i, parameters)

            if math.fabs(population.min_error(error.individual_err)) < 1e-5:
                self._write_msg(
                    communication_delay,
                    iteration,
                    prev_q,
                    population.best_individual(error.individual_err),
                    MessageType.STOP,
                    proc_id,
                )
                self._write_msg(
                    communication_delay,
                    iteration,
                    next_q,
                    population.best_individual(error.individual_err),
                    MessageType.STOP,
                    proc_id,
                )
                break

            self._write_msg(
                communication_delay,
                iteration,
                prev_q,
                population[0],
                MessageType.FROM_PREV_CHUNK,
                proc_id,
            )

            self._write_msg(
                communication_delay,
                iteration,
                next_q,
                population[-1],
                MessageType.FROM_NEXT_CHUNK,
                proc_id,
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
                proc_id, iteration, population.min_error(error.individual_err)
            )
        )

        for callback in iteration_callbacks:
            callback.end(graph, population, parameters)

        results.put(population)

    @validate(
        Parameter.QA_TEMPERATURE,
        Parameter.QA_MAX_STEPS,
        Parameter.MUTATION,
        Parameter.ERROR,
    )
    def _markow_chain(
        self,
        graph: Graph,
        population: Population,
        index: int,
        parameters: Dict[str, Any],
    ) -> None:

        temperature = param_value(graph, parameters, Parameter.QA_TEMPERATURE)
        max_steps = param_value(graph, parameters, Parameter.QA_MAX_STEPS)
        mutation = param_value(graph, parameters, Parameter.MUTATION)
        error = param_value(graph, parameters, Parameter.ERROR)

        for _ in range(max_steps):
            individual = population[index]
            population_error_old = error.population_err(graph, population, parameters)
            new_individual, diff_nodes = mutation.mutate(graph, individual, parameters)
            delta_individual_error = error.individual_err(
                graph, new_individual
            ) - error.individual_err(graph, individual)
            population.set_individual(index, new_individual, diff_nodes)
            population_error_new = error.population_err(graph, population, parameters)
            delta_population_error = population_error_new - population_error_old

            if delta_individual_error > 0 or delta_population_error > 0:
                try:
                    probability = 1 - math.exp(
                        (-1 * delta_population_error) / temperature
                    )
                except OverflowError:
                    probability = 1
                if random.random() <= probability:
                    population.set_individual(index, individual, diff_nodes)
