import mgp
from enum import Enum


class Parameter(Enum):
    END = "end"
    NODE_PROPS = "node_props"
    PROPERTY = "property"
    RELATIONSHIPS = "relationships"
    REL_PROPS = "rel_props"
    START = "start"
    TYPE = "type"


@mgp.read_proc
def schema(
    context: mgp.ProcCtx, verbose: bool = False
) -> mgp.Record(simple_schema=mgp.Map, verbose_schema=str):
    """
    Procedure to generate the graph database schema for LangChain.

    Args:
        context (mgp.ProcCtx): Reference to the context execution.
        verbose (bool): If set to True, the graph schema will include additional context.

    Returns:
        mgp.Record containing a mgp.Map of node properties, relationship properties and all relationships in the database and a string representing verbose schema especially useful for LangChain.

    Example:
        Get a simple schema without verbose schema:
            `CALL llm_util.schema() YIELD simple_schema, verbose_schema RETURN simple_schema, verbose_schema;`
            or
            `CALL llm_util.schema() YIELD simple_schema, verbose_schema RETURN simple_schema;`
        Get graph schema with simple and verbose schema:
            `CALL llm_util.schema(true) YIELD simple_schema, verbose_schema RETURN simple_schema, verbose_schema;`
        Get only verbose schema:
            `CALL llm_util.schema(true) YIELD simple_schema, verbose_schema RETURN verbose_schema;`
    """

    node_counter = 0
    all_node_properties_dict = {}
    all_relationship_properties_dict = {}
    all_relationships_list = []
    for node in context.graph.vertices:
        node_counter += 1
        labels = ":".join(tuple(sorted(label.name for label in node.labels)))

        all_node_properties_dict = get_properties_dict(
            node, all_node_properties_dict, labels
        )

        for relationship in node.out_edges:
            target_labels = tuple(
                sorted(label.name for label in relationship.to_vertex.labels)
            )

            full_relationship = {
                Parameter.START.value: labels,
                Parameter.TYPE.value: relationship.type.name,
                Parameter.END.value: ":".join(target_labels),
            }
            if full_relationship not in all_relationships_list:
                all_relationships_list.append(full_relationship)
            all_relationship_properties_dict = get_properties_dict(
                relationship, all_relationship_properties_dict, relationship.type.name
            )

    if node_counter == 0:
        raise Exception(
            "Can't generate a graph schema since there is no data in the database."
        )

    if verbose:
        verbose_schema = "Node properties are the following:\n"
        for label in all_node_properties_dict.keys():
            verbose_schema += f"Node name: '{label}', Node properties: {all_node_properties_dict.get(label)}\n"

        verbose_schema += "\nRelationship properties are the following:\n"
        for type in all_relationship_properties_dict.keys():
            verbose_schema += f"Relationship Name: '{type}', Relationship Properties: {all_relationship_properties_dict.get(type)}\n"

        verbose_schema += "\nThe relationships are the following:\n"

        for relationship in all_relationships_list:
            verbose_schema += f"['(:{relationship[Parameter.START.value]})-[:{relationship[Parameter.TYPE.value]}]->(:{relationship[Parameter.END.value]})']\n"

        return mgp.Record(
            simple_schema={
                Parameter.NODE_PROPS.value: all_relationship_properties_dict,
                Parameter.REL_PROPS.value: all_relationship_properties_dict,
                Parameter.RELATIONSHIPS.value: all_relationships_list,
            },
            verbose_schema=verbose_schema,
        )

    else:
        return mgp.Record(
            simple_schema={
                Parameter.NODE_PROPS.value: all_relationship_properties_dict,
                Parameter.REL_PROPS.value: all_relationship_properties_dict,
                Parameter.RELATIONSHIPS.value: all_relationships_list,
            },
            verbose_schema="",
        )


def get_properties_dict(graph_object, all_properties_dict, key):
    for property_name in graph_object.properties.keys():
        if not all_properties_dict.get(key):
            all_properties_dict[key] = [
                {
                    Parameter.PROPERTY.value: property_name,
                    Parameter.TYPE.value: type(
                        graph_object.properties.get(property_name)
                    ).__name__,
                }
            ]
        else:
            if property_name not in [
                d[Parameter.PROPERTY.value] for d in all_properties_dict[key]
            ]:
                all_properties_dict[key].append(
                    {
                        Parameter.PROPERTY.value: property_name,
                        Parameter.TYPE.value: type(
                            graph_object.properties.get(property_name)
                        ).__name__,
                    }
                )
    return all_properties_dict
