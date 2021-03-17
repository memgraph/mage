import logging
from typing import Dict, Any, Optional, Tuple
from mage.graph_coloring_module import Graph
from mage.graph_coloring_module import create
from mage.graph_coloring_module import Error
from mage.graph_coloring_module import Individual
from mage.graph_coloring_module import Population
from mage.graph_coloring_module import Algorithm
from mage.graph_coloring_module import param_value
from mage.graph_coloring_module import validate
from mage.graph_coloring_module import Message
import multiprocessing as mp
from abc import ABC, abstractmethod


logger = logging.getLogger('telco')


class ParallelAlgorithm(Algorithm, ABC):
    """A class that represents abstract parallel algorithm."""

    @validate("no_of_processes", "no_of_chunks", "error")
    def run(
            self,
            graph: Graph,
            parameters: Dict[str, Any]) -> Individual:
        """Runs the algorithm in a given number of processes and returns the best individual.

        Parameters that must be specified:
        :no_of_processes: the number of processes to run an algorithm in
        :no_of_chunks: The number of pieces into which the population is divided.
        Each population chunk has approximately population_size / no_of_chunks individuals.
        :error: a function that defines an error"""

        no_of_processes = param_value(graph, parameters, "no_of_processes")
        no_of_chunks = param_value(graph, parameters, "no_of_chunks")
        error = param_value(graph, parameters, "error")

        populations = create(graph, parameters)

        pool = mp.Pool(no_of_processes)
        m = mp.Manager()
        results = m.Queue()
        queues = [m.Queue() for _ in range(no_of_chunks)]

        processes = []
        for i, population in enumerate(populations):
            my_q, prev_q, next_q = self._get_queues(i, no_of_chunks, queues)
            p = mp.Process(
                target = self.algorithm,
                args = (i, graph, population, my_q, prev_q, next_q, results, parameters))
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
            parameters: Dict[str, Any]) -> None:
        """A function that executes an algorithm."""
        pass

    def _get_queues(
            self,
            ind: int,
            no_of_chunks: int,
            queues: mp.Queue) -> Tuple[Optional[mp.Queue], Optional[mp.Queue], Optional[mp.Queue]]:
        """Returns my_q, prev_q, next_q for the process on a given index."""
        if no_of_chunks == 1:
            return None, None, None
        prev_chunk = ind - 1 if ind - 1 > 0 else no_of_chunks - 1
        next_chunk = ind + 1 if ind + 1 < no_of_chunks else 0
        prev_q = queues[prev_chunk]
        next_q = queues[next_chunk]
        my_q = queues[ind]
        return my_q, prev_q, next_q

    def _write_msgs(
            self,
            communication_delay: int,
            iteration: int,
            next_q: mp.Queue,
            prev_q: mp.Queue,
            indv: Individual,
            stop: bool = False) -> None:
        """Sends the individual to the previous and next part of the population."""
        if next_q is not None and prev_q is not None:
            if stop:
                next_q.put(Message(indv, 0))
                prev_q.put(Message(indv, 0))
            elif iteration % communication_delay == 0:
                next_q.put(Message(indv, 1))
                prev_q.put(Message(indv, -1))

    def _read_msgs(
            self,
            my_q: mp.Queue,
            population: Population) -> None:
        """Reads messages from the queue and sets the previous or next individual of the population."""
        if my_q is not None:
            while not my_q.empty():
                msg = my_q.get()
                if msg.msg_type == 1:
                    population.set_next_individual(msg.data)
                elif msg.msg_type == -1:
                    population.set_prev_individual(msg.data)
                elif msg.msg_type == 0:
                    return True
        return False

    def _find_best(
            self,
            graph: Graph,
            results: mp.Queue,
            error: Error) -> Individual:
        """Finds the individual with the smallest error in the results queue."""
        individuals = []
        while not results.empty():
            pop = results.get()
            individuals.extend(pop.best_individuals)
        best_individual = min(individuals, key = lambda indv: error.individual_err(graph, indv))
        return best_individual
