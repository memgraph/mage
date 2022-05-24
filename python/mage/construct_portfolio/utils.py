from typing import List
import numpy as np
import igraph
import random
from scipy.stats import spearmanr


def split_data(data: List, number_of_elements_in_each_bin: int) -> List[List[float]]:
    """Function used for splitting data.

    Args:
        data (List): data
        number_of_elements_in_each_bin (int): number of elements that will be in each bin after splitting

    Returns:
        List[List[float]]: splitted data

    """

    return np.array(np.array_split(data, number_of_elements_in_each_bin))


def create_igraph_from_matrix(matrix: List[List[float]]):
    """Create igraph graph from weighted 2D matrix
    Args:
        matrix (List[List[float]]): weighted matrix
    Returns:
        Igraph graph
    """

    random.seed(0)
    igraph.set_random_number_generator(random)
    graph = igraph.Graph.Weighted_Adjacency(
        matrix, mode=igraph.ADJ_UNDIRECTED, attr="weight", loops=False
    )

    return graph


def get_sorted_indices(stock_tickers: List[str]) -> List[List[int]]:
    """Returns sorted indices

    Args:
        stock_tickers (List[str]): stock tickers on each day

    Returns:
       List[List[int]]: sorted indices on each day

    """

    sorted_indices = []
    for i in range(stock_tickers.shape[0]):
        sorted_indices.append(np.argsort(stock_tickers[i]).tolist())

    return sorted_indices


def get_last_n_days(data: List, number_of_stocks: int, n: int) -> List:
    """Function for extracting only last n days

    Args:
        data (List): data
        number_of_stocks (int): number of stocks
        n (int): number of days

    Returns:
        List: list containing on last n days

    """

    return data[-number_of_stocks * n :]


def split_data_and_sort(
    data: List, number_of_days: int, sorted_indices: List[List[int]]
) -> List[List]:
    """Function used for splitting and sorting data.

    Args:
        data (List): data
        number_of_days (int): number of days for splitting
        sorted_indices (List[List[int]]): indices used for sorting in each day

    Returns:
        List: splitted and sorted 2D-array

    """
    x = split_data(data, number_of_days)
    for i in range(number_of_days):
        x[i] = x[i][sorted_indices[i]]

    return x.transpose()


def calculate_correlations(
    values: List[List[float]], measure: str = "pearson"
) -> List[List[float]]:
    """Calculate correlations between values

    Args:
        values (List[List[float]]): values
        measure (str, optional): measure. Defaults to 'pearson'.

    Returns:
        List[List[float]]: correlations

    """

    if measure == "pearson":
        return abs(np.corrcoef(values))
    elif measure == "spearman":
        return abs(spearmanr(values, axis=1))


def get_n_best_performing(
    stock_trading_values: List[List[float]], members: List[int], n_best_performing: int
) -> List[int]:
    """From each community pick number of best performing

    Args:
        stock_trading_values (List[List[float]]): values of each stock
        members (List[int]): indicies which are in community
        n_best_performing (int): number of best performing stocks

    Returns:
        List[int]: indicies of best performing stocks in community

    """

    members = np.array(members)
    current_community_stock_trading_values = stock_trading_values[members]
    mean_current_community_stock_trading_values = np.mean(
        current_community_stock_trading_values, axis=1
    )
    sorted_indices = np.argsort(mean_current_community_stock_trading_values)

    return members[sorted_indices[-n_best_performing:]]
