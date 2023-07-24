#include <mgp.hpp>
#include <string>


namespace Collections{
    constexpr std::string_view kReturnValuePartition = "result";
    constexpr std::string_view  kProcedurePartition = "partition";
    constexpr std::string_view  kArgumentListPartition = "list";
    constexpr std::string_view  kArgumentSizePartition = "partition_size";
    
    void Partition(mgp_list *args, mgp_graph *memgraph_graph, mgp_result *result, mgp_memory *memory);

}