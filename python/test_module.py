import mgp

from mage.test_submodule.test_functions import test_function


@mgp.read_proc
def calculate(
    ctx: mgp.ProcCtx, a: mgp.Number, b: mgp.Number
) -> mgp.Record(result=mgp.Number):
    return mgp.Record(result=int(test_function(a, b)))
