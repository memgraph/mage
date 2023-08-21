import mgp
from ast import literal_eval

@mgp.function
def str2object(ctx: mgp.FuncCtx, string: str) -> mgp.Any:
    string = str(string)
    return literal_eval(string)