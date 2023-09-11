#pragma once

#include <mgp.hpp>

namespace Refactor {

/*invert constants*/
constexpr std::string_view kProcedureInvert = "invert";
constexpr std::string_view kArgumentRelationship = "relationship";
constexpr std::string_view kReturnRelationshipInvert = "relationship";
constexpr std::string_view kReturnIdInvert = "id_inverted";

void Invert(mgp_list *args, mgp_graph *memgraph_graph, mgp_result *result, mgp_memory *memory);
void InvertRel(mgp::Graph &graph, mgp::Relationship &rel);
}  // namespace Refactor
