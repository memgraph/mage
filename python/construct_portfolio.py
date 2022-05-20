import mgp
from typing import List
import numpy as np
import igraph
import random
from scipy.stats import spearmanr

@mgp.read_proc
def get(
    context: mgp.ProcCtx,
    stocks: List[str],
    values: List[float],
    number_of_last_n_trading_days:int = 3,
    n_best_performing:int = -1,
    resolution_parameter:float = 0.6,
    correlation_measure:str = 'pearson',
) -> mgp.Record(community_index = int, community = str):

    """Procedure for constructing portfolio and getting communities detected by leiden algorithm.

    Args:
        stocks: stock node tickers
        values: values used for calculating correlations
        number_of_last_n_trading_days: number of days taking in consideration while calculating correlations
        n_best_performing: number of best perfroming stocks to pick from each community
        resolution_parameter: the resolution parameter to use. Higher resolutions lead to more smaller communities, while lower resolutions lead to fewer larger communities.
        correlation_measure: measure to use for calculating correlations between stocks

    Returns:
        Community indexes and communities

    The procedure can be invoked in openCypher using the following call:

    MATCH (s:Stock)-[r:Traded_On]->(d:TradingDay)
    WHERE d.date < "2022-04-27"
    WITH collect(s.ticker) as stocks,collect(r.close - r.open) as daily_returns 
    CALL construct_portfolio.get(stocks,daily_returns,5,5,0.7)
    YIELD community_index, community
    RETURN community_index, community;

    """

    if number_of_last_n_trading_days <= 2:
        raise InvalidNumberOfTradingDaysException("Number of last trading days must be greater than 2.")

    if not correlation_measure in ['pearson','spearman']:
        raise InvalidCorrelationMeasureException("Correlation measure can only be either pearson or spearman")


    stock_tickers = np.sort(np.unique(stocks))

    if len(values) == 0 or len(values) < len(stock_tickers) * 3:
        raise NotEnoughOfDataException("There need to be atleast three entries of data for each stock")

    stock_nodes = get_last_n_days(stocks, len(stock_tickers), number_of_last_n_trading_days)
    stock_trading_values = get_last_n_days(values, len(stock_tickers), number_of_last_n_trading_days)

    stock_nodes = split_data(stock_nodes, number_of_last_n_trading_days)

    sorted_indices = get_sorted_indices(stock_nodes)
    
    stock_trading_values = split_data_and_sort(stock_trading_values, number_of_last_n_trading_days, sorted_indices)
      
    correlations = calculate_correlations(stock_trading_values, correlation_measure)

    graph = create_igraph_from_matrix(correlations)

    communities = graph.community_leiden(weights=graph.es['weight'], resolution_parameter=resolution_parameter, n_iterations = -1)

    return get_records(communities, stock_tickers, stock_trading_values, n_best_performing)
    

def split_data(data: List, number_of_elements_in_each_bin: int) -> List[List[float]]:
    """Function used for splitting data.

    Args:
        data (List): data
        number_of_elements_in_each_bin (int): number of elements that will be in each bin after splitting

    Returns:
        List[List[float]]: splitted data

    """

    return np.array(np.array_split(data,number_of_elements_in_each_bin))

def create_igraph_from_matrix(matrix: List[List[float]]):
    """Create igraph graph from weighted 2D matrix 
    Args:
        matrix (List[List[float]]): weighted matrix
    Returns:
        Igraph graph
    """

    random.seed(0)
    igraph.set_random_number_generator(random)
    graph = igraph.Graph.Weighted_Adjacency(matrix, mode = igraph.ADJ_UNDIRECTED,attr = 'weight',loops = False)

    return graph


def get_sorted_indices(stock_tickers: List[str]) -> List[List[int]]:
    """Returns sorted indices

    Args:
        stock_tickers (List[str]): stock tickers on each day

    Returns:
       List[List[float]]: sorted indices on each day

    """

    sorted_indices = []
    for i in range(stock_tickers.shape[0]):
        sorted_indices.append(np.argsort(stock_tickers[i]))
    
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

    return data[-number_of_stocks*n:]

def split_data_and_sort(data: List, number_of_days: int, sorted_indices: List[List[int]]) -> List[List]:
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

def calculate_correlations(values: List[List[float]], measure: str = 'pearson') -> List[List[float]]:
    """Calculate correlations between values

    Args:
        values (List[List[float]]): values
        measure (str, optional): measure. Defaults to 'pearson'.

    Returns:
        List[List[float]]: correlations

    """

    if measure == 'pearson':
        return abs(np.corrcoef(values))
    elif measure == 'spearman':
        return abs(spearmanr(values,axis = 1))

def get_n_best_performing(stock_trading_values:List[List[float]], members:List[int], n_best_performing:int) -> List[int]:
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
    mean_current_community_stock_trading_values = np.mean(current_community_stock_trading_values, axis = 1)
    sorted_indices = np.argsort(mean_current_community_stock_trading_values)

    return members[sorted_indices[-n_best_performing:]]

    
def get_records(communities: List[List[int]], stock_tickers: List[str], stock_trading_values: List[float], n_best_performing: int):
    """Function for creating returning mgp.Record data

    Args:
        communities (List[List[int]]): List of communities with belonging members indexes
        stock_tickers (List[str]): stock tickers 
        stock_trading_values (List[float]): stock trading values
        n_best_performing (int): number of best performing stocks to pick from each community

    Returns:
        List[mgp.Record(community_index = int, community = str)]: list of mgp.Record(community_index = int, community = str)

    """

    records = []
    for i,members in enumerate(communities):
        if n_best_performing > 0:
            members = get_n_best_performing(stock_trading_values,members,n_best_performing)

        stocks_in_community = ", ".join(stock_tickers[members])
        records.append(mgp.Record(community_index = i, community = stocks_in_community ))

    return records

class InvalidNumberOfTradingDaysException(Exception):
    pass

class InvalidCorrelationMeasureException(Exception):
    pass

class NotEnoughOfDataException(Exception):
    pass
