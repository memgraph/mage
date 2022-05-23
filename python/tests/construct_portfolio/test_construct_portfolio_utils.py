import mage.construct_portfolio.utils as utils
import pytest
import numpy as np
from typing import List

STOCKS_TICKER_VALUES = np.array([['AAPL', 'TSLA', 'ZION'], [['ZION', 'TSLA', 'AAPL']]], dtype=object)
STOCKS_TICKER_UNIQUE = np.array(['AAPL', 'TSLA', 'ZION'], dtype=object)
STOCK_VALUES = np.array([1, 2, 3, 1, 2, 3, 1, 2, 3], dtype=np.int32)
STOCK_TRADING_VALUES_SPLITTED = np.array([[1, 1, 1], [2, 2, 2], [3, 3, 3]], dtype=np.int32)


def same_array_values(
    array_1: List[float], array_2: List[float], absolute_tolerance=1e-5
) -> bool:
    return np.all(
        np.isclose(
            array_1,
            array_2,
            atol=absolute_tolerance,
        )
    )


@pytest.mark.parametrize(
    "data, number_of_stocks,number_of_days",
    [
        (STOCK_VALUES, len(STOCKS_TICKER_UNIQUE), 2),
    ],
)
def test_get_last_n_days(data, number_of_stocks, number_of_days):
    last_n_days_array = utils.get_last_n_days(data, number_of_stocks, number_of_days)

    assert same_array_values(last_n_days_array, [1, 2, 3, 1, 2, 3])


@pytest.mark.parametrize(
    "data",
    [
        (STOCKS_TICKER_VALUES),
    ],
)
def test_get_sorted_indices(data):
    sorted_indices = utils.get_sorted_indices(data)

    assert sorted_indices == [[0, 1, 2], [[2, 1, 0]]]


@pytest.mark.parametrize(
    "data, number_of_elements_each_bin",
    [
        (STOCK_VALUES, 3),
    ],
)
def test_split_data(data, number_of_elements_each_bin):
    splitted_array = utils.split_data(data, number_of_elements_each_bin)

    assert splitted_array.shape[0] == len(data) / number_of_elements_each_bin


@pytest.mark.parametrize(
    "data, number_of_days,sorted_indices",
    [
        (STOCK_VALUES, 3, [[0, 1, 2], [2, 1, 0], [0, 1, 2]]),
    ],
)
def test_split_sort_data(data, number_of_days, sorted_indices):
    splitted_sorted_array = utils.split_data_and_sort(data, number_of_days, sorted_indices)

    assert same_array_values(splitted_sorted_array, [[1, 3, 1], [2, 2, 2], [3, 1, 3]])


@pytest.mark.parametrize(
    "data, members, n_best_performing",
    [
        (STOCK_TRADING_VALUES_SPLITTED, [0, 1, 2], 2),
    ],
)
def test_get_n_best_performing(data, members, n_best_performing):
    best_performing = utils.get_n_best_performing(data, members, n_best_performing)

    assert same_array_values(best_performing, [1, 2])
