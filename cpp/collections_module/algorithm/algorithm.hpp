#include <mgp.hpp>
#include <string>

namespace Collections {

constexpr std::string_view kReturnToSet = "result";
constexpr std::string_view kProcedureToSet = "to_set";
constexpr std::string_view kArgumentListToSet = "list";
void toSet(mgp_list *args, mgp_graph *memgraph_graph, mgp_result *result, mgp_memory *memory);
}  // namespace Collections
