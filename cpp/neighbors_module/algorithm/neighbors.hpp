#pragma once

#include <mgp.hpp>
#include <string_view>

namespace Neighbors {

enum class RelDirection { kNone = -1, kAny = 0, kIncoming = 1, kOutgoing = 2, kBoth = 3 };

struct Config {
  explicit Config(const mgp::List &list_of_relationships);

  RelDirection GetDirection(std::string_view rel_type);

  bool any_incoming{false};
  bool any_outgoing{true};
  std::unordered_map<std::string_view, RelDirection> rel_direction;
};

constexpr std::string_view kReturnAtHop = "nodes";
constexpr std::string_view kProcedureAtHop = "at_hop";

constexpr std::string_view kReturnByHop = "nodes";
constexpr std::string_view kProcedureByHop = "by_hop";

constexpr std::string_view kArgumentsNode = "node";
constexpr std::string_view kArgumentsRelType = "rel_type";
constexpr std::string_view kArgumentsDistance = "distance";

constexpr std::string_view kResultAtHop = "nodes";
constexpr std::string_view kResultByHop = "nodes";

/* to_hop constants */
constexpr const std::string_view kProcedureToHop = "to_hop";
constexpr const std::string_view kToHopArg1 = "node";
constexpr const std::string_view kToHopArg2 = "types";
constexpr const std::string_view kToHopArg3 = "distance";
constexpr const std::string_view kResultToHop = "node";

void AtHop(mgp_list *args, mgp_graph *memgraph_graph, mgp_result *result, mgp_memory *memory);

void ByHop(mgp_list *args, mgp_graph *memgraph_graph, mgp_result *result, mgp_memory *memory);

void ToHop(mgp_list *args, mgp_graph *memgraph_graph, mgp_result *result, mgp_memory *memory);

}  // namespace Neighbors
