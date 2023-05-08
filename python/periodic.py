import mgp

from gqlalchemy import Memgraph

@mgp.read_proc
def iterate(
  context: mgp.ProcCtx,
  input_query: str,
  running_query: str,
  config: mgp.Map
) -> mgp.Record(success=bool):

  if "batch_size" not in config:
    raise Exception("Batch size is not specified in periodic.iterate!")
  
  batch_size = config["batch_size"]
  if not isinstance(batch_size, int):
    raise Exception("Batch size is not an integer!")

  memgraph = Memgraph()

  input_results = memgraph.execute_and_fetch(input_query)
  input_results = list(input_results)

  unwind_query = "UNWIND $periodic_inputs as periodic_input"

  offset = 0
  while True:
    start = offset
    end = offset + batch_size if offset + batch_size <= len(input_results) else len(input_results)

    input_results_batch = input_results[start : end]
    input_results_batch = {"periodic_inputs": input_results_batch}

    memgraph.execute(f"{unwind_query} {running_query}", input_results_batch)

    if end == len(input_results):
      break
    
    offset = end

  return mgp.Record(success=True)
