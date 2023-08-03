import mgp
import pytz
import datetime

def getOffset(timezone, input_date):
    offset = pytz.timezone(timezone).utcoffset(input_date)
    if (offset.days == 1):
        return datetime.timedelta(minutes=offset.seconds // 60 + 24*60), False
    elif (offset.days == -1):
        return datetime.timedelta(minutes=24*60 - offset.seconds // 60), True
    return datetime.timedelta(minutes=offset.seconds // 60), False

@mgp.read_proc
def parse(
    time: str,
    unit: str = "ms",
    format: str = "%Y/%m/%d %H:%M:%S",
    timezone: str = "UTC",
) -> mgp.Record(parsed=int):
    first_date = datetime.datetime(1970, 1, 1, 0, 0, 0)
    input_date = datetime.datetime.strptime(time, format)

    if (timezone in pytz.all_timezones):
        offset, subtract = getOffset(timezone, input_date)
        if (subtract):
            tz_input = input_date + offset
        else:
            tz_input = input_date - offset
    else:
        raise Exception("Timezone doesn't exist. Check documentation to see available timezones.")

    time_since = tz_input - first_date

    if (unit == "ms"):
        parsed = time_since.days * 24 * 60 * 60 * 1000 + time_since.seconds * 1000
    elif (unit == "s"):
        parsed = time_since.days * 24 * 60 * 60 + time_since.seconds
    elif (unit == "m"):
        parsed = time_since.days * 24 * 60 + time_since.seconds // 60
    elif (unit == "h"):
        parsed = time_since.days * 24 + time_since.seconds // 60 // 60
    elif (unit == "d"):
        parsed = time_since.days
    else:
        raise Exception("Unit doesn't exist. Check documentation to see available units.")

    return mgp.Record(parsed=parsed)
