import mgp
from json import loads


@mgp.function
def str2object(string: str) -> mgp.Any:
    if string:
        return loads(string)
    else:
        return None
