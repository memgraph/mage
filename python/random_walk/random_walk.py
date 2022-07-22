from click import pass_context
import mgp  # wrapper around the C API
import random


@mgp.read_proc  # read proc is because procedure will only read from the database
def procedure(context: mgp.ProcCtx,  # ProcCtx is actually a reference to the graph
              starting_vertex: mgp.Vertex,  # starting vertex
              length: int
              ) -> mgp.Record(result=mgp.Path):

    # Create a path that will be returned
    path = mgp.Path(starting_vertex)
     
    # Create a hash set for efficiently checking if some node is already added
    visited_cities = set()
    visited_cities.add(starting_vertex.properties.get("name"))
    # You can also do: visited_cities = {starting_vertex.properties.get("name")}
    # Iterate for at most length times


    source_city = starting_vertex
    for i in range(length):
        curr_out_edges = list(source_city.out_edges)  # for sure not the best way when you already have generator but 

        # for vertex in path.vertices:
        #    print(vertex.properties.get("name"), end=' ')

        if len(curr_out_edges) == 0:
            break
        else:
            target_ind = random.randint(0, len(curr_out_edges)- 1)

        target_city = curr_out_edges[target_ind].to_vertex

        if target_city.properties.get("name") in visited_cities:
            break  # break before time

        # Append to the set
        visited_cities.add(target_city.properties.get("name"))
        # Append to the path
        path.expand(curr_out_edges[target_ind])
        source_city = target_city

    # Multiple rows can be produced by returning an iterable of mgp.Record
    return mgp.Record(result=path)
