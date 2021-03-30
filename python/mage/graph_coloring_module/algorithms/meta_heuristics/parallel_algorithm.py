import logging
from typing import Dict, Any, Optional, Tuple
from mage.graph_coloring_module.graph import Graph
from mage.graph_coloring_module.error_functions.error import Error
from mage.graph_coloring_module.components.individual import Individual
from mage.graph_coloring_module.components.population import Population
from mage.graph_coloring_module.algorithms.algorithm import Algorithm
from mage.graph_coloring_module.utils.parameters_utils import param_value
from mage.graph_coloring_module.utils.validation import validate
from mage.graph_coloring_module.communication.message import Message
import multiprocessing as mp
from abc import ABC, abstractmethod
from mage.graph_coloring_module.parameters import Parameter
from mage.graph_coloring_module.communication.message_type import MessageType


logger = logging.getLogger("graph_coloring")


class ParallelAlgorithm(Algorithm, ABC):
    """A class that represents abstract parallel algorithm."""

    @validate(
        Parameter.NO_OF_PROCESSES,
        Parameter.NO_OF_CHUNKS,
        Parameter.ERROR,
        Parameter.POPULATION_FACTORY,
    )
    def run(self, graph: Graph, parameters: Dict[str, Any]) -> Individual:
        """Runs the algorithm in a given number of processes and returns the best individual.

        Parameters that must be specified:
        :no_of_processes: the number of processes to run an algorithm in
        :no_of_chunks: The number of pieces into which the population is divided.
        Each population chunk has approximately population_size / no_of_chunks individuals.
        :error: a function that defines an error"""

        no_of_processes = param_value(graph, parameters, Parameter.NO_OF_PROCESSES)
        no_of_chunks = param_value(graph, parameters, Parameter.NO_OF_CHUNKS)
        error = param_value(graph, parameters, Parameter.ERROR)
        population_factory = param_value(
            graph, parameters, Parameter.POPULATION_FACTORY
        )

        populations = population_factory.create(graph, parameters)

        pool = mp.Pool(no_of_processes)
        m = mp.Manager()
        results = m.Queue()
        queues = [m.Queue() for _ in range(no_of_chunks)]

        processes = []
        for i, population in enumerate(populations):
            my_q, prev_q, next_q = self._get_queues(i, no_of_chunks, queues)
            p = mp.Process(
                target=self.algorithm,
                args=(i, graph, population, my_q, prev_q, next_q, results, parameters),
            )
            processes.append(p)
            p.start()
        for p in processes:
            p.join()

        best_individual = self._find_best(graph, results, error)
        pool.close()

        return best_individual

    @abstractmethod
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
        """A function that executes an algorithm."""
        pass

    def _get_queues(
        self, ind: int, no_of_chunks: int, queues: mp.Queue
    ) -> Tuple[Optional[mp.Queue], Optional[mp.Queue], Optional[mp.Queue]]:
        """Returns my_q, prev_q, next_q for the process on a given index."""
        if no_of_chunks == 1:
            return None, None, None
        prev_chunk = ind - 1 if ind - 1 > 0 else no_of_chunks - 1
        next_chunk = ind + 1 if ind + 1 < no_of_chunks else 0
        prev_q = queues[prev_chunk]
        next_q = queues[next_chunk]
        my_q = queues[ind]
        return my_q, prev_q, next_q

    def _write_msg(
        self,
        communication_delay: int,
        iteration: int,
        queue: mp.Queue,
        indv: Individual,
        msg_type: MessageType,
    ) -> None:
        if queue is not None:
            if iteration % communication_delay == 0:
                queue.put(Message(indv, msg_type))

    def _write_stop(self, prev_q: mp.Queue, next_q: mp.Queue, indv: Individual) -> None:
        if next_q is not None and prev_q is not None:
            next_q.put(Message(indv, MessageType.STOP))
            prev_q.put(Message(indv, MessageType.STOP))

    def _read_msgs(self, my_q: mp.Queue, population: Population) -> None:
        """Reads messages from the queue and sets the previous or next individual of the population."""
        if my_q is not None:
            while not my_q.empty():
                msg = my_q.get()
                if msg.msg_type == MessageType.FROM_NEXT_CHUNK:
                    population.set_next_individual(msg.data)
                elif msg.msg_type == MessageType.FROM_PREV_CHUNK:
                    population.set_prev_individual(msg.data)
                elif msg.msg_type == MessageType.STOP:
                    return True
        return False

    def _find_best(self, graph: Graph, results: mp.Queue, error: Error) -> Individual:
        """Finds the individual with the smallest error in the results queue."""
        individuals = []
        while not results.empty():
            pop = results.get()
            individuals.extend(pop.best_individuals)
        best_individual = min(
            individuals, key=lambda indv: error.individual_err(graph, indv)
        )
        return best_individual
