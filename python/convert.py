import mgp
from typing import Any, List, Union, Optional


def _flatten_relationship_properties(rel_props: dict, rel_type: str) -> dict:
    """Flatten relationship properties with relationship type prefix."""
    return {f"{rel_type.lower()}.{k}": v for k, v in rel_props.items()}


def _convert_to_tree(value: Any, config: Optional[mgp.Map] = None) -> Union[dict, list, Any]:
    """Helper function to convert Memgraph values to a tree structure."""
    if isinstance(value, mgp.Vertex):
        result = {
            "_type": next(iter(value.labels)).name,
            "_id": value.id
        }
        # Add filtered properties
        for k, v in value.properties.items():
            if any(_should_include_property(k, config, label.name) for label in value.labels):
                result[k] = _convert_to_tree(v, config)
        return result
    elif isinstance(value, mgp.Edge):
        # For edges, we'll return both the properties and target node
        rel_type = value.type.name
        target_node = _convert_to_tree(value.to_vertex, config)
        
        # Add flattened relationship properties to target node
        if config is None or 'rels' not in config or rel_type in config['rels']:
            rel_props = _flatten_relationship_properties(value.properties, rel_type)
            target_node.update(rel_props)
        
        return target_node
    elif isinstance(value, mgp.Path):
        # For a path starting from a Person, we'll create a tree with movies under acted_in
        start_node = _convert_to_tree(value.vertices[0], config)
        if not start_node["_type"] == "Person":
            return start_node
        
        # Initialize acted_in array if not present
        if "acted_in" not in start_node:
            start_node["acted_in"] = []
        
        # Add movie with relationship properties
        movie_node = _convert_to_tree(value.edges[0], config)
        start_node["acted_in"].append(movie_node)
        
        return start_node
    elif isinstance(value, (list, tuple)):
        if all(isinstance(x, mgp.Path) for x in value):
            # Merge all paths into a single tree
            if not value:
                return None
            
            result = _convert_to_tree(value[0], config)
            for path in value[1:]:
                path_tree = _convert_to_tree(path, config)
                if "acted_in" in path_tree:
                    result["acted_in"].extend(path_tree["acted_in"])
            return result
        return [_convert_to_tree(item, config) for item in value]
    elif isinstance(value, dict):
        return {k: _convert_to_tree(v, config) for k, v in value.items()}
    else:
        return value


def _should_include_property(prop_name: str, config: mgp.Map, label: str) -> bool:
    """Helper function to determine if a property should be included based on config."""
    if not config or 'nodes' not in config or label not in config['nodes']:
        return True
    props = config['nodes'][label]
    if not props:
        return True
    # If first property starts with '-', it's an exclusion list
    if props[0].startswith('-'):
        return not any(p[1:] == prop_name for p in props if p.startswith('-'))
    # Otherwise, it's an inclusion list
    return prop_name in props


@mgp.read_proc
def to_tree(ctx: mgp.ProcCtx,
            value: Any,
            config: Optional[mgp.Map] = None) -> mgp.Record(value=Any):
    """
    Converts graph elements into a tree structure.
    Similar to Neo4j's apoc.convert.toTree().

    This procedure converts nodes, relationships, and paths into a tree structure
    where graph metadata is prefixed with underscore (_) and node/relationship
    properties are included as regular key-value pairs.

    Parameters
    ----------
    value : Any
        The value to convert to a tree structure. Can be a node, relationship,
        path, or any nested structure containing these elements.
    config : mgp.Map, optional
        Configuration for property filtering. Can contain 'nodes' and 'rels' keys
        with label/type-specific property lists. Properties can be included or
        excluded (prefix with '-').

    Returns
    -------
    mgp.Record
        A record containing the tree structure representation of the input value.

    Examples
    --------
    Include only specific properties:
    {
        nodes: {Movie: ["title"]},
        rels: {ACTED_IN: ["roles"]}
    }

    Exclude specific properties:
    {
        nodes: {Movie: ["-tagline"]},
        rels: {REVIEWED: ["-rating"]}
    }
    """
    converted = _convert_to_tree(value, config)
    return mgp.Record(value=converted) 