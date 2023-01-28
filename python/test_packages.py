import mgp


@mgp.read_proc
def calculate2(
    ctx: mgp.ProcCtx, a: mgp.Number, b: mgp.Number
) -> mgp.Record(result1=mgp.Number, result2=mgp.Number):
    return mgp.Record(
        result1=10,
        result2=8,
    )
