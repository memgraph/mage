import mgp
import pytz
import datetime

from mage.date.constants import Conversion, Epoch
from mage.date.unit_conversion import to_int, to_timedelta


MILLISECOND = {"ms", "milli", "millis", "milliseconds"}
SECOND = {"s", "second", "seconds"}
MINUTE = {"m", "minute", "minutes"}
HOUR = {"h", "hour", "hours"}
DAY = {"d", "day", "days"}


def getOffset(timezone, date):
    offset = pytz.timezone(timezone).utcoffset(date)
    if offset.days == 1:
        return (
            datetime.timedelta(
                minutes=offset.seconds // Conversion.SECONDS_IN_MINUTE
                + Conversion.HOURS_IN_DAY * Conversion.MINUTES_IN_HOUR
            ),
            False,
        )
    elif offset.days == -1:
        return (
            datetime.timedelta(
                minutes=Conversion.HOURS_IN_DAY * Conversion.MINUTES_IN_HOUR
                - offset.seconds // Conversion.SECONDS_IN_MINUTE
            ),
            True,
        )
    return (
        datetime.timedelta(minutes=offset.seconds // Conversion.SECONDS_IN_MINUTE),
        False,
    )


@mgp.read_proc
def parse(
    time: str,
    unit: str = "ms",
    format: str = "%Y-%m-%d %H:%M:%S",
    timezone: str = "UTC",
) -> mgp.Record(parsed=int):
    first_date = Epoch.UNIX_EPOCH
    input_date = datetime.datetime.strptime(time, format)

    if timezone not in pytz.all_timezones:
        raise Exception(
            "Timezone doesn't exist. Check documentation to see available timezones."
        )

    offset, add = getOffset(timezone, input_date)
    tz_input = input_date + offset if add else input_date - offset

    time_since = tz_input - first_date

    if unit == "ms":
        parsed = (
            time_since.days
            * Conversion.HOURS_IN_DAY
            * Conversion.MINUTES_IN_HOUR
            * Conversion.SECONDS_IN_MINUTE
            * Conversion.MILLISECONDS_IN_SECOND
            + time_since.seconds * Conversion.MILLISECONDS_IN_SECOND
        )
    elif unit == "s":
        parsed = (
            time_since.days
            * Conversion.HOURS_IN_DAY
            * Conversion.MINUTES_IN_HOUR
            * Conversion.SECONDS_IN_MINUTE
            + time_since.seconds
        )
    elif unit == "m":
        parsed = (
            time_since.days * Conversion.HOURS_IN_DAY * Conversion.MINUTES_IN_HOUR
            + time_since.seconds // Conversion.SECONDS_IN_MINUTE
        )
    elif unit == "h":
        parsed = (
            time_since.days * Conversion.HOURS_IN_DAY
            + time_since.seconds
            // Conversion.SECONDS_IN_MINUTE
            // Conversion.MINUTES_IN_HOUR
        )
    elif unit == "d":
        parsed = time_since.days
    else:
        raise Exception(
            "Unit doesn't exist. Check documentation to see available units."
        )

    return mgp.Record(parsed=parsed)


@mgp.read_proc
def format(
    time: int,
    unit: str = "ms",
    format: str = "%Y-%m-%d %H:%M:%S %Z",
    timezone: str = "UTC",
) -> mgp.Record(formatted=str):
    first_date = Epoch.UNIX_EPOCH

    if unit == "ms":
        new_date = first_date + datetime.timedelta(milliseconds=time)
    elif unit == "s":
        new_date = first_date + datetime.timedelta(seconds=time)
    elif unit == "m":
        new_date = first_date + datetime.timedelta(minutes=time)
    elif unit == "h":
        new_date = first_date + datetime.timedelta(hours=time)
    elif unit == "d":
        new_date = first_date + datetime.timedelta(days=time)
    else:
        raise Exception(
            "Unit doesn't exist. Check documentation to see available units."
        )

    if timezone not in pytz.all_timezones:
        raise Exception(
            "Timezone doesn't exist. Check documentation to see available timezones."
        )
    offset, subtract = getOffset(timezone, new_date)
    tz_new = new_date - offset if subtract else new_date + offset

    return mgp.Record(
        formatted=pytz.timezone(timezone).localize(tz_new).strftime(format)
    )


@mgp.function
def add(
    time: int = None,
    unit: str = None,
    add_value: int = None,
    add_unit: str = None,
) -> int:
    return to_int(
        to_timedelta(time=time, unit=unit) + to_timedelta(time=add_value, unit=add_unit),
        unit=unit,
    )
