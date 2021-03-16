from itertools import product
from typing import Any, List, Optional, Tuple, Dict
import importlib.util
import re
import datetime
import glob
import os
import logging
from graph_coloring_module.loaders.dimacs_loader import load_dimacs
from graph_coloring_module.algorithms.algorithm import Algorithm
from graph_coloring_module.framework.output_functions import write_to_file


logger = logging.getLogger('telco')


def run(configurations_path: str, result_path: str) -> None:
    for configuration in glob.iglob(os.path.join(configurations_path, "**//*.py"), recursive=True):
        conf = _get_configuration(configuration)
        if conf is not None:
            (folders, algorithms, parameters) = conf
            graph_filepaths = _get_graph_files(folders)
            for graph_filepath in graph_filepaths:
                graph = load_dimacs(graph_filepath)
                for alg, alg_params in zip(algorithms, parameters):
                    params_list = _get_parameters(alg_params)
                    for params in params_list:
                        try:
                            sol = alg.run(graph, params)
                            matches = re.search(r'[\w.]+$', graph_filepath)
                            graph_name = matches.group(0)
                            date = re.search(r'[:\s\w-]+', str(datetime.datetime.now())).group(0)
                            date = date.replace(" ", "_")
                            result_file_name = graph_name + "_" + str(alg) + "_" + date + ".txt"
                            filename = os.path.join(result_path, result_file_name)
                            write_to_file(params, sol, graph, filename)
                        except Exception as e:
                            logger.exception(e)


def _check_configuration_file(module) -> bool:
    if not hasattr(module, 'folders'):
        return False
    if not hasattr(module, 'algorithms'):
        return False
    if not hasattr(module, 'parameters'):
        return False
    return True


def _get_configuration(
        conf_path: str) -> Optional[Tuple[List[str], List[Algorithm], List[Dict[str, Any]]]]:

    matches = re.search(r'[\w]+.py$', conf_path)
    module_name = matches.group(0)
    spec = importlib.util.spec_from_file_location(module_name, conf_path)
    conf_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(conf_module)
    if not _check_configuration_file(conf_module):
        return None

    folders = conf_module.folders
    algorithms = conf_module.algorithms
    parameters = conf_module.parameters
    return folders, algorithms, parameters


def _get_graph_files(folders: List[str]) -> List[str]:
    graph_filepaths = []
    for folder in folders:
        for filepath in glob.iglob(os.path.join(folder, "**//*"), recursive=True):
            graph_filepaths.append(filepath)
    return graph_filepaths


def _get_parameters(params: Dict[str, Any]) -> List[Dict[str, Any]]:
    params_list = []
    keys, values = zip(*params.items())
    values = list(values)
    for value_list in values:
        if isinstance(value_list[0], dict):
            new_value_list = []
            for value in value_list:
                new_value_list.extend(_get_parameters(value))
            values[values.index(value_list)] = new_value_list
    for bundle in product(*values, range(1)):
        d = dict(zip(keys, bundle[:-1]))
        params_list.append(d)
    return params_list
