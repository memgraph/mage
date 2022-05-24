import mgp
from typing import List
import numpy as np
import mage.construct_portfolio.utils as utils


@mgp.read_proc
def get(
    context: mgp.ProcCtx,
    stocks: List[str],
    values: List[float],
    number_of_last_n_trading_days: int = 3,
    n_best_performing: int = -1,
    resolution_parameter: float = 0.6,
    correlation_measure: str = "pearson",
) -> mgp.Record(community_index=int, community=str):
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
        raise InvalidNumberOfTradingDaysException(
            "Number of last trading days must be greater than 2."
        )

    if correlation_measure not in ["pearson", "spearman"]:
        raise InvalidCorrelationMeasureException(
            "Correlation measure can only be either pearson or spearman"
        )

    stock_tickers = np.sort(np.unique(stocks))

    if len(values) == 0 or len(values) < len(stock_tickers) * 3:
        raise NotEnoughOfDataException(
            "There need to be atleast three entries of data for each stock"
        )

    stock_nodes = utils.get_last_n_days(
        stocks, len(stock_tickers), number_of_last_n_trading_days
    )
    stock_trading_values = utils.get_last_n_days(
        values, len(stock_tickers), number_of_last_n_trading_days
    )

    stock_nodes = utils.split_data(stock_nodes, number_of_last_n_trading_days)

    sorted_indices = utils.get_sorted_indices(stock_nodes)

    stock_trading_values = utils.split_data_and_sort(
        stock_trading_values, number_of_last_n_trading_days, sorted_indices
    )

    correlations = utils.calculate_correlations(
        stock_trading_values, correlation_measure
    )

    graph = utils.create_igraph_from_matrix(correlations)

    communities = graph.community_leiden(
        weights=graph.es["weight"],
        resolution_parameter=resolution_parameter,
        n_iterations=-1,
    )

    return get_records(
        communities, stock_tickers, stock_trading_values, n_best_performing
    )


def get_records(
    communities: List[List[int]],
    stock_tickers: List[str],
    stock_trading_values: List[float],
    n_best_performing: int,
):
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
    for i, members in enumerate(communities):
        if n_best_performing > 0:
            members = utils.get_n_best_performing(
                stock_trading_values, members, n_best_performing
            )

        stocks_in_community = ", ".join(stock_tickers[members])
        records.append(mgp.Record(community_index=i, community=stocks_in_community))

    return records


class InvalidNumberOfTradingDaysException(Exception):
    pass


class InvalidCorrelationMeasureException(Exception):
    pass


class NotEnoughOfDataException(Exception):
    pass
