#include <cstdint>
#include <mg_exceptions.hpp>
#include <mg_utils.hpp>

#include "algorithm/leiden.hpp"


namespace {

const char *kProcedureGet = "get";
const char *kFieldNode = "node";
const char *kFieldCommunity = "community_id";
const double kDefaultWeight = 1.0;


void InsertLeidenRecord(mgp_graph *graph, mgp_result *result, mgp_memory *memory, const std::uint64_t node_id,
                        const std::uint64_t community) {
    auto *vertex = mg_utility::GetNodeForInsertion(node_id, graph, memory);
    if (!vertex) return;

    mgp_result_record *record = mgp::result_new_record(result);
    if (record == nullptr) throw mg_exception::NotEnoughMemoryException();

    mg_utility::InsertNodeValueResult(record, kFieldNode, vertex, memory);
    mg_utility::InsertIntValueResult(record, kFieldCommunity, community, memory);
}


void OnGraph(mgp_list *args, mgp_graph *memgraph_graph, mgp_result *result, mgp_memory *memory) {
    auto graph = mg_utility::GetGraphView(memgraph_graph, result, memory, mg_graph::GraphType::kUndirectedGraph);
    try {
        auto communities = leiden_alg::GetCommunities(*graph);

        for (std::size_t i = 0; i < communities.size(); i++) {
            InsertLeidenRecord(memgraph_graph, result, memory, i, communities[i]);
        }
    } catch (const std::exception &e) {
        mgp::result_set_error_msg(result, e.what());
        return;
    }
}


extern "C" int mgp_init_module(mgp_module *module, mgp_memory *memory) {
    try {
        auto *proc = mgp::module_add_read_procedure(module, kProcedureGet, OnGraph);

        mgp::proc_add_result(proc, kFieldNode, mgp::type_node());
        mgp::proc_add_result(proc, kFieldCommunity, mgp::type_int());
    } catch (const std::exception &e) {
        return 1;
    }
    
    return 0;
} 



extern "C" int mgp_shutdown_module() { return 0; }

}  // namespace