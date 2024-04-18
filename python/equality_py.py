import mgp

@mgp.function
def is_equal(ctx: mgp.FuncCtx, output: str, input: str):
  return output == input
