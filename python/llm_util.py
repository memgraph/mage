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


class OutputType(Enum):
    RAW = "raw"
    PROMPT_READY = "prompt_ready"


class SchemaGenerator(object):
    def __init__(self, context, output_type):
        self._type = output_type.lower()
        self._node_counter = 0
        self._all_node_properties_dict = {}
        self._all_relationship_properties_dict = {}
        self._all_relationships_list = []

        self._generate_schema(context)

    @property
    def type(self):
        return self._type

    @property
    def node_counter(self):
        return self._node_counter

    def get_schema(self):
        if self._type == OutputType.RAW.value:
            return self._get_raw_schema()
        elif self._type == OutputType.PROMPT_READY.value:
            return self._get_prompt_ready_schema()
        else:
            raise Exception(
                f"Can't generate a graph schema since the provided output_type is not correct. Please choose {OutputType.RAW.value} or {OutputType.PROMPT_READY.value}."
            )

    def _generate_schema(self, context):
        for node in context.graph.vertices:
            self._node_counter += 1
            labels = ":".join(tuple(sorted(label.name for label in node.labels)))

            self._update_properties_dict(node, self._all_node_properties_dict, labels)

            for relationship in node.out_edges:
                target_labels = tuple(
                    sorted(label.name for label in relationship.to_vertex.labels)
                )

                full_relationship = {
                    Parameter.START.value: labels,
                    Parameter.TYPE.value: relationship.type.name,
                    Parameter.END.value: ":".join(target_labels),
                }
                if full_relationship not in self._all_relationships_list:
                    self._all_relationships_list.append(full_relationship)

                self._update_properties_dict(
                    relationship,
                    self._all_relationship_properties_dict,
                    relationship.type.name,
                )

    def _get_raw_schema(self):
        return {
            Parameter.NODE_PROPS.value: self._all_node_properties_dict,
            Parameter.REL_PROPS.value: self._all_relationship_properties_dict,
            Parameter.RELATIONSHIPS.value: self._all_relationships_list,
        }

    def _get_prompt_ready_schema(self):
        prompt_ready_schema = "Node properties are the following:\n"
        for label in self._all_node_properties_dict.keys():
            prompt_ready_schema += f"Node name: '{label}', Node properties: {self._all_node_properties_dict.get(label)}\n"

        prompt_ready_schema += "\nRelationship properties are the following:\n"
        for rel in self._all_relationship_properties_dict.keys():
            prompt_ready_schema += f"Relationship Name: '{rel}', Relationship Properties: {self._all_relationship_properties_dict.get(rel)}\n"

        prompt_ready_schema += "\nThe relationships are the following:\n"

        for relationship in self._all_relationships_list:
            prompt_ready_schema += f"['(:{relationship[Parameter.START.value]})-[:{relationship[Parameter.TYPE.value]}]->(:{relationship[Parameter.END.value]})']\n"

        return prompt_ready_schema

    def _update_properties_dict(self, graph_object, all_properties_dict, key):
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


@mgp.read_proc
def schema(
    context: mgp.ProcCtx,
    output_type: str = OutputType.PROMPT_READY.value,
) -> mgp.Record(schema=mgp.Any):
    """
    Procedure to generate the graph database schema in raw or prompt-ready format.

    Args:
        context (mgp.ProcCtx): Reference to the context execution.
        output_type (str): By default (set to 'prompt_ready'), the graph schema will include additional context and it will be prompt-ready. If set to 'raw', it will produce a simple version which can be adjusted for prompt.

    Returns:
        mgp.Record containing a mgp.Map of node properties, relationship properties and all relationships in the database and a string representing prompt-ready schema.

    Example:
        Get raw graph schema:
            `CALL llm_util.schema('raw') YIELD schema RETURN schema;`
        Get prompt-ready graph schema:
            `CALL llm_util.schema() YIELD schema RETURN schema;`
            or
            `CALL llm_util.schema('prompt_ready') YIELD schema RETURN schema;`
    """

    schema_generator = SchemaGenerator(context, output_type)

    if schema_generator.node_counter == 0:
        raise Exception(
            "Can't generate a graph schema since there is no data in the database."
        )

    return mgp.Record(schema=schema_generator.get_schema())
