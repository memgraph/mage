import mgp
import pytz
import datetime

MINUTES_IN_HOUR = 60
SECONDS_IN_MINUTE = 60
MILLISECONDS_IN_SECOND = 1000
HOURS_IN_DAY = 24
UNIX_EPOCH = datetime.datetime(1970, 1, 1, 0, 0, 0)

def getOffset(timezone, date):
    offset = pytz.timezone(timezone).utcoffset(date)
    if (offset.days == 1):
        return datetime.timedelta(minutes=offset.seconds // SECONDS_IN_MINUTE + HOURS_IN_DAY*MINUTES_IN_HOUR), False
    elif (offset.days == -1):
        return datetime.timedelta(minutes=HOURS_IN_DAY*MINUTES_IN_HOUR - offset.seconds // SECONDS_IN_MINUTE), True
    return datetime.timedelta(minutes=offset.seconds // SECONDS_IN_MINUTE), False

@mgp.read_proc
def parse(
    time: str,
    unit: str = "ms",
    format: str = "%Y-%m-%d %H:%M:%S",
    timezone: str = "UTC",
) -> mgp.Record(parsed=int):
    first_date = UNIX_EPOCH
    input_date = datetime.datetime.strptime(time, format)

    if (timezone in pytz.all_timezones):
        offset, add = getOffset(timezone, input_date)
        if (add):
            tz_input = input_date + offset
        else:
            tz_input = input_date - offset
    else:
        raise Exception("Timezone doesn't exist. Check documentation to see available timezones.")

    time_since = tz_input - first_date

    if (unit == "ms"):
        parsed = time_since.days * HOURS_IN_DAY * MINUTES_IN_HOUR * SECONDS_IN_MINUTE * MILLISECONDS_IN_SECOND + time_since.seconds * MILLISECONDS_IN_SECOND
    elif (unit == "s"):
        parsed = time_since.days * HOURS_IN_DAY * MINUTES_IN_HOUR * SECONDS_IN_MINUTE + time_since.seconds
    elif (unit == "m"):
        parsed = time_since.days * HOURS_IN_DAY * MINUTES_IN_HOUR + time_since.seconds // SECONDS_IN_MINUTE
    elif (unit == "h"):
        parsed = time_since.days * HOURS_IN_DAY + time_since.seconds // SECONDS_IN_MINUTE // MINUTES_IN_HOUR
    elif (unit == "d"):
        parsed = time_since.days
    else:
        raise Exception("Unit doesn't exist. Check documentation to see available units.")

    return mgp.Record(parsed=parsed)


@mgp.read_proc
def format(
    time: int,
    unit: str = "ms",
    format: str = "%Y-%m-%d %H:%M:%S %Z",
    timezone: str = "UTC",
) -> mgp.Record(formatted=str):
    first_date = UNIX_EPOCH

    if (unit == "ms"):
        new_date = first_date + datetime.timedelta(milliseconds=time)
    elif (unit == "s"):
        new_date = first_date + datetime.timedelta(seconds=time)
    elif (unit == "m"):
        new_date = first_date + datetime.timedelta(minutes=time)
    elif (unit == "h"):
        new_date = first_date + datetime.timedelta(hours=time)
    elif (unit == "d"):
        new_date = first_date + datetime.timedelta(days=time)
    else:
        raise Exception("Unit doesn't exist. Check documentation to see available units.")

    if (timezone in pytz.all_timezones):
        offset, subtract = getOffset(timezone, new_date)
        if (subtract):
            tz_new = new_date - offset
        else:
            tz_new = new_date + offset
    else:
        raise Exception("Timezone doesn't exist. Check documentation to see available timezones.")

    return mgp.Record(formatted=pytz.timezone(timezone).localize(tz_new).strftime(format))
