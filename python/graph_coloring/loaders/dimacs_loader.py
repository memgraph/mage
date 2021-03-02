from collections import defaultdict
from typing import Optional, List
import logging
import re
from telco.graph import Graph


logger = logging.getLogger('telco')


def load_dimacs(filepath: str) -> Optional[Graph]:
    """Reads the graph from the file in DIMACS format.
    Return Graph if the file was read successfully, otherwise returns None.
    Throws IOError."""

    edge_list = []
    matches = re.search(r'[\w.]+$', filepath)
    graph_name = matches.group(0)

    with open(filepath, 'r') as f:
        logger.info('Started reading from file {}...'.format(filepath))

        if not _verify_dimacs_format(filepath):
            return None

        for line in f:
            line = line.rstrip()
            if line.startswith('p'):
                num_nodes = int(line.split()[2])
            elif line.startswith('e'):
                u = int(line.split()[1])
                v = int(line.split()[2])
                edge_list.append((u, v))
        logger.info('File {} was read successfully.'.format(filepath))

        nodes = list(range(1, num_nodes + 1))
        adj_list = defaultdict(list)
        for u, v in edge_list:
            adj_list[u].append((v, 1))
            adj_list[v].append((u, 1))

        return Graph(nodes, adj_list, graph_name)


def _verify_dimacs_format(filepath: str) -> bool:
    correct_format = True
    p_line_exist = False
    check_flag = False

    with open(filepath, 'r') as f:
        for line_no, line in enumerate(f, start=1):
            line = line.rstrip()
            args = line.split()

            if not line.startswith('c'):
                if line.startswith('p'):
                    if p_line_exist:
                        correct_format = False
                        logger.info('Line {} : '
                                    'There must be one p line.'.format(line_no))
                    else:
                        if not _verify_p_line(args, line_no):
                            correct_format = False
                        else:
                            num_nodes = int(args[2])
                            check_flag = True
                        p_line_exist = True

                elif line.startswith('e'):
                    if not _verify_e_line(args, line_no):
                        correct_format = False
                    elif check_flag and not _check_labels(args, num_nodes, line_no):
                        correct_format = False

                else:
                    correct_format = False
                    logger.info('Line {} : Wrong line type.'.format(line_no))

    if not correct_format:
        logger.error('File {} : Wrong file format.'.format(filepath))

    return correct_format


def _verify_p_line(args: List[str], line_no) -> bool:
    if len(args) != 4:
        logger.info('Line {} : '
                    'Wrong number of arguments in the line.'.format(line_no))
        return False
    if not (args[0] == 'p' and args[1] == 'edge' and args[2].isnumeric() and args[3].isnumeric()):
        logger.info('Line {} : '
                    'Wrong arguments in the line.'.format(line_no))
        return False
    return True


def _verify_e_line(args: List[str], line_no) -> bool:
    if len(args) != 3:
        logger.info('Line {} : '
                    'Wrong number of arguments in the line.'.format(line_no))
        return False
    if not (args[0] == 'e' and args[1].isnumeric() and args[2].isnumeric()):
        logger.info('Line {} : '
                    'Wrong arguments in the line.'.format(line_no))
        return False
    return True


def _check_labels(args: List[str], num_nodes, line_no) -> bool:
    if not (1 <= int(args[1]) <= num_nodes and 1 <= int(args[2]) <= num_nodes):
        logger.info('Line {} : Wrong node label.'.format(line_no))
        return False
    return True
