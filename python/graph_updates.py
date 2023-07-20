import mgp


@mgp.write_proc
def set_property(ctx:mgp.ProcCtx, vertex:mgp.Vertex, property_name:str, value:mgp.Any)->mgp.Record(vertex=mgp.Vertex):
    vertex.properties.set(property_name, value)
    return mgp.Record(vertex=vertex)
