import pytest

from mage.date.unit_conversion import toInt, toTimedelta
from mage.date.constants import Units

POSITIVE_VALUE = 12345
NEGATIVE_VALUE = -12345
UNIT_NAMES = list(Units.MILLISECOND | Units.SECOND | Units.MINUTE | Units.HOUR | Units.DAY)


@pytest.mark.parametrize("unit", UNIT_NAMES)
def test_roundtrip_positive(unit):
    assert POSITIVE_VALUE == toInt(toTimedelta(POSITIVE_VALUE, unit), unit)


@pytest.mark.parametrize("unit", UNIT_NAMES)
def test_roundtrip_negative(unit):
    assert NEGATIVE_VALUE == toInt(toTimedelta(NEGATIVE_VALUE, unit), unit)


def test_incorrect_unit_to_int():
    incorrect_unit = "year"
    with pytest.raises(Exception, match=f"The unit {incorrect_unit} is not correct.") as _:
        toInt(POSITIVE_VALUE, incorrect_unit)


def test_incorrect_unit_to_timedelta():
    incorrect_unit = "year"
    with pytest.raises(Exception, match=f"The unit {incorrect_unit} is not correct.") as _:
        toTimedelta(POSITIVE_VALUE, incorrect_unit)
