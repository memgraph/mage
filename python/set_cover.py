import mgp
import random
import abc
from collections import defaultdict
from typing import List

from gekko import GEKKO

@mgp.read_proc
def cp_solve(context: mgp.ProcCtx,
              element_vertexes: List[mgp.Vertex],
              set_vertexes: List[mgp.Vertex]
              ) -> mgp.Record(resulting_sets=List[mgp.Vertex]):
    '''
    This set cover solver method returns 1 filed

      * `resulting_sets` is a minimal set of sets in which all the element have been contained

    The input arguments consist of

      * `element_vertexes` that is a list of element nodes
      * `set_vertexes` that is a list of set nodes those elements are contained in

    Element and set equivalents at a certain index come in pairs so mappings between sets and elements are consistent.

    The procedure can be invoked in openCypher using the following calls, e.g.:
      CALL set_cover.cp_solve([(:Point), (:Point)], [(:Set), (:Set)]) YIELD resulting_sets;

    The method uses constraint programming as a solving tool for obtaining a minimal set of sets that contain
        all the elements.
    '''

    creator = GekkoMatchingProblemCreator()
    mp = creator.create_matching_problem(element_vertexes, set_vertexes)

    solver = GekkoMPSolver()
    result = solver.solve(matching_problem=mp)

    resulting_nodes = [context.graph.get_vertex_by_id(x) for x in result]

    return [mgp.Record(resulting_sets=resulting_nodes)]


@mgp.read_proc
def greedy(context: mgp.ProcCtx,
              element_vertexes: List[mgp.Vertex],
              set_vertexes: List[mgp.Vertex]
              ) -> mgp.Record(resulting_sets=List[mgp.Vertex]):
    '''
    This set cover solver method returns 1 filed

      * `resulting_sets` is a minimal set of sets in which all the element have been contained

    The input arguments consist of

      * `element_vertexes` that is a list of element nodes
      * `set_vertexes` that is a list of set nodes those elements are contained in

    Element and set equivalents at a certain index come in pairs so mappings between sets and elements are consistent.

    The procedure can be invoked in openCypher using the following calls, e.g.:
      CALL set_cover.cp_solve([(:Point), (:Point)], [(:Set), (:Set)]) YIELD resulting_sets;

    The method uses a greedy method as a solving tool for obtaining a minimal set of sets that contain
        all the elements.
    '''

    creator = GreedyMatchingProblemCreator()
    mp = creator.create_matching_problem(element_vertexes, set_vertexes)

    solver = GreedyMPSolver()
    result = solver.solve(matching_problem=mp)

    resulting_nodes = [context.graph.get_vertex_by_id(x) for x in result]

    return [mgp.Record(resulting_sets=resulting_nodes)]


class GreedyMatchingProblem:
    '''
    Matching problem to be used with greedy solving of set cover.
    '''

    def __init__(self, elements, containing_sets, elements_by_sets):
        self.elements = elements
        self.containing_sets = containing_sets
        self.elements_by_sets = elements_by_sets


class GekkoMatchingProblem:
    '''
    Matching problem to be used with gekko constraint programming solving of set cover.
    '''

    def __init__(self, containing_sets, sets_by_elements):
        self.containing_sets = containing_sets
        self.sets_by_elements = sets_by_elements


class MatchingProblemCreator(abc.ABC):
    '''
    Creator abstract class of matching problems
    '''

    @abc.abstractmethod
    def create_matching_problem(self, element_vertexes, set_vertexes):
        '''
        Creates a matching problem
        :param element_vertexes: Element vertexes pair component list
        :param set_vertexes: Set vertexes pair component list
        :return: matching problem
        '''

        pass


class GreedyMatchingProblemCreator(MatchingProblemCreator):
    '''
    Creator class for set cover to be solved with greedy method
    '''

    def create_matching_problem(self, element_vertexes, set_vertexes):
        '''
        Creates a matching problem to be solved with greedy method
        :param element_vertexes: Element vertexes pair component list
        :param set_vertexes: Set vertexes pair component list
        :return: matching problem
        '''

        element_values = [x.id for x in element_vertexes]
        set_values = [x.id for x in set_vertexes]
        all_elements = set(element_values)
        all_sets = set(set_values)

        elements_by_sets = defaultdict(set)

        for element, contained_set in zip(element_values, set_values):
            elements_by_sets[contained_set].add(element)

        return GreedyMatchingProblem(all_elements, all_sets, elements_by_sets)


class GekkoMatchingProblemCreator(MatchingProblemCreator):
    '''
    Creator class for set cover to be solved with gekko constraint programming
    '''

    def create_matching_problem(self, element_vertexes, set_vertexes):
        '''
        Creates a matching problem to be solved with gekko constraing programming method
        :param element_vertexes: Element vertexes pair component list
        :param set_vertexes: Set vertexes pair component list
        :return: matching problem
        '''

        element_values = [x.id for x in element_vertexes]
        set_values = [x.id for x in set_vertexes]
        set_values_distinct = set(set_values)
        sets_by_elements = defaultdict(set)

        for element, contained_set in zip(element_values, set_values):
            sets_by_elements[element].add(contained_set)

        return GekkoMatchingProblem(set_values_distinct, sets_by_elements)


class MatchingProblemSolver(abc.ABC):
    '''
    Solver of set cover matching problem
    '''

    @abc.abstractmethod
    def solve(self, matching_problem):
        '''
        Solves the matching problem and returns the set indices
        :param matching_problem: matching problem
        :return: set indices
        '''
        pass


class GekkoMPSolver(MatchingProblemSolver):
    '''
    Solver of set cover with gekko constraint programming
    '''

    def solve(self, matching_problem):
        '''
        Solves the matching problem and returns the set indices
        :param matching_problem: matching problem
        :return: set indices
        '''

        m = GEKKO(remote=True)
        m.options.SOLVER = 1
        containing_const = m.Const(1, name='const')

        set_list = list(matching_problem.containing_sets)
        vars = [m.Var(lb=0, ub=1, integer=True, name=GekkoMPSolver.get_variable_name(i))
                         for i in range(len(set_list))]

        set_ordinal_map = {value: i for i, value in enumerate(set_list)}

        for element in matching_problem.sets_by_elements.keys():
            containing_sets = matching_problem.sets_by_elements[element]
            contained_sets_eq = 0

            for contained_set in containing_sets:
                ordinal_number = set_ordinal_map[contained_set]
                contained_sets_eq = contained_sets_eq + vars[ordinal_number]

            m.Equation(equation=contained_sets_eq >= containing_const)

        m.Obj(sum(vars))
        m.solve()

        resulting_sets = []
        for idx, var in enumerate(vars):
            if var.value[0] == 1.0:
                resulting_sets.append(set_list[idx])

        return resulting_sets


    @staticmethod
    def get_variable_name(set_no):
        '''
        Returns unique variable name based on the set id
        :param set_no: set id
        :return: set variable name
        '''

        return f"containing_set_{set_no}"


class GreedyMPSolver(MatchingProblemSolver):
    '''
    Solver of set cover with greedy method
    '''

    def solve(self, matching_problem):
        '''
        Solves the matching problem and returns the set indices
        :param matching_problem: matching problem
        :return: set indices
        '''

        universe = matching_problem.elements

        possible_sets = list(matching_problem.containing_sets)
        random.shuffle(possible_sets)

        picked_sets = list()
        covered_universe = set()

        while len(universe) != len(covered_universe):
            picked_set = possible_sets[0]
            picked_sets.append(picked_set)
            possible_sets = possible_sets[1:]

            covered_universe |= matching_problem.elements_by_sets[picked_set]

        return picked_sets
