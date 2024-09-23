#include <cstdint>
#include <mg_exceptions.hpp>
#include <mg_utils.hpp>

#include "_mgp.hpp"
#include "algorithm/leiden.hpp"
#include "mg_procedure.h"


namespace {

const char *kProcedureGet = "get";
const char *kFieldNode = "node";
const char *kFieldCommunity = "community_id";
const char *kFieldCommunities = "communities";

void InsertLeidenRecord(mgp_graph *graph, mgp_result *result, mgp_memory *memory, const std::uint64_t node_id,
                        const std::vector<int> &community) {
    auto *vertex = mg_utility::GetNodeForInsertion(node_id, graph, memory);
    if (!vertex) return;

    mgp_result_record *record = mgp::result_new_record(result);
    if (record == nullptr) throw mg_exception::NotEnoughMemoryException();

    mg_utility::InsertNodeValueResult(record, kFieldNode, vertex, memory);
    mg_utility::InsertIntValueResult(record, kFieldCommunity, community.back(), memory);

    auto *community_list = mgp::list_make_empty(0, memory);
    for (const auto &community_id : community) {
        mgp::list_append_extend(community_list, mgp::value_make_int(community_id, memory));
    }

    mg_utility::InsertListValueResult(record, kFieldCommunities, community_list, memory);
}


void OnGraph(mgp_list *args, mgp_graph *memgraph_graph, mgp_result *result, mgp_memory *memory) {
    auto graph = mg_utility::GetGraphView(memgraph_graph, result, memory, mg_graph::GraphType::kUndirectedGraph);
    try {
        auto communities = leiden_alg::getCommunities(*graph);

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
        mgp::proc_add_result(proc, kFieldCommunities, mgp::type_list(mgp::type_int()));
    } catch (const std::exception &e) {
        return 1;
    }
    
    return 0;
} 



extern "C" int mgp_shutdown_module() { return 0; }

}  // namespace