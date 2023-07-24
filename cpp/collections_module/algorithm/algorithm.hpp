#include <mgp.hpp>
#include <string>

namespace Collections {
constexpr std::string_view kReturnValueContains = "output";
constexpr std::string_view kProcedureContains = "contains";
constexpr std::string_view kArgumentListContains = "list";
constexpr std::string_view kArgumentValueContains = "value";
void Contains(mgp_list *args, mgp_graph *memgraph_graph, mgp_result *result, mgp_memory *memory);
}  // namespace Collections
