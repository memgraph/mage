#include <mgp.hpp>
#include <sstream>
#include <string>

namespace Collections{
    constexpr std::string_view kReturnValueMin = "min";
    constexpr std::string_view kProcedureMin = "min";
    constexpr std::string_view kArgumentListMin = "list";
    void Min(mgp_list *args, mgp_graph *memgraph_graph, mgp_result *result, mgp_memory *memory);
  
}
