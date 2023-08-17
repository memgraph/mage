import datetime


class Conversion(int):
    MINUTES_IN_HOUR = 60
    SECONDS_IN_MINUTE = 60
    MILLISECONDS_IN_SECOND = 1000
    HOURS_IN_DAY = 24


class Epoch(datetime.datetime):
    UNIX_EPOCH = datetime.datetime(1970, 1, 1, 0, 0, 0)
