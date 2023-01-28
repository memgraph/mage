#include <mg_utils.hpp>
#include <iostream>

// Methods
constexpr char const *jaccardAll = "jaccard";
// Return parameters
constexpr char const *node1 = "node1";
constexpr char const *node2 = "node2";
constexpr char const *similarity = "similarity";

void Jaccard(mgp_list *args, mgp_graph *memgraph_graph, mgp_result *result, mgp_memory *memory) {
    auto graph = mg_utility::GetGraphView(memgraph_graph, result, memory, mg_graph::GraphType::kUndirectedGraph);
    const auto record_factory = mgp::RecordFactory(result);

    auto record = record_factory.NewRecord();
    record.insert(node1, 1.1);
    record.insert(node2, 1.34);
    record.insert(similarity, 0.8);

}

extern "C" int mgp_init_module(struct mgp_module *module, struct mgp_memory *memory) {
    mgp::memory = memory;

    try {
        mgp::AddProcedure(Jaccard,
        jaccardAll,
        mgp::ProcedureType::Read,
        {}, // no input parameters
        {
            mgp::Return(node1, mgp::Type::Double),
            mgp::Return(node1, mgp::Type::Double),
            mgp::Return(similarity, mgp::Type::Double)
        },
        module,
        memory
        );
    } catch (const std::exception &e) {
        return 1;
    }
    return 0;
}

extern "C" int mgp_shutdown_module() { return 0; }