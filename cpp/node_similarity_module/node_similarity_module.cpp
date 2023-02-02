#include <cstdint>
#include <unordered_map>
#include <unordered_set>
#include <mgp.hpp>
#include <omp.h>
#include "algorithms/node_similarity.hpp"


namespace {

// Methods
constexpr char const *jaccardAll = "jaccard";
// Return parameters
constexpr char const *node1_name = "node1";
constexpr char const *node2_name = "node2";
constexpr char const *similarity = "similarity";

void Jaccard(mgp_list *args, mgp_graph *memgraph_graph, mgp_result *result, mgp_memory *memory) {
    mgp::memory = memory;
    const auto record_factory = mgp::RecordFactory(result);
    const auto graph = mgp::Graph(memgraph_graph);
    std::unordered_set<std::pair<uint64_t, uint64_t>, node_similarity_util::pair_hash> visited_node_pairs;
    // Cache neighbors
    std::unordered_map<uint64_t, std::unordered_set<uint64_t>> neighbors;
    for (const auto &node1: graph.Nodes()) {
        // uint64_t node1_id = node1.Id().AsUint();
        // Process neighboura
        // std::unordered_set<uint64_t> ns1;
        // if (neighbors.count(node1_id)) {
        //     ns1 = neighbors[node1_id];
        // } else {
        //     for (const auto n1: node1.OutRelationships()) {
        //         ns1.insert(n1.To().Id().AsUint());
        //     }
        //     neighbors[node1_id] = ns1;
        // }
        for (const auto &node2: graph.Nodes()) {
            // uint64_t node2_id = node2.Id().AsUint();
            // if (node1 == node2 || visited_node_pairs.count(std::make_pair<>(node2_id, node1_id))) {
            //     continue;
            // }
            // visited_node_pairs.insert(std::make_pair<>(node1_id, node2_id));
            // // Process neighbours
            // std::unordered_set<uint64_t> ns2;
            // if (neighbors.count(node2_id)) {
            //     ns2 = neighbors[node2_id];
            // } else {
            //     for (const auto n2: node2.OutRelationships()) {
            //         ns2.insert(n2.To().Id().AsUint());
            //     }
            //     neighbors[node2_id] = ns2;
            // }
            // calculate intersection and union
            // std::unordered_set<uint64_t> elem_union, elem_intersection;
            // std::set_union(ns1.begin(), ns1.end(), ns2.begin(), ns2.end(), std::inserter(elem_union, elem_union.begin()));
            // std::set_intersection(ns1.begin(), ns1.end(), ns2.begin(), ns2.end(), std::inserter(elem_intersection, elem_intersection.begin()));
            // Construct a record
            auto record = record_factory.NewRecord();
            record.Insert(node1_name, node1);
            record.Insert(node2_name, node2);
            // if (elem_union.size() == 0) {
            record.Insert(similarity, 0.0);
            // } else {
            //     double sim = elem_intersection.size() / (double) elem_union.size();
            //     record.Insert(similarity, sim);
            // }
        }
    }
    // auto record = record_factory.NewRecord();
    // record.Insert(node1_name, 2.0);
    // record.Insert(node2_name, 2.0);
    // record.Insert(similarity, 2.0);
}

} // namespace

extern "C" int mgp_init_module(mgp_module *module, mgp_memory *memory) {
    mgp::memory = memory;

    try {
        mgp::AddProcedure(Jaccard,
        jaccardAll,
        mgp::ProcedureType::Read,
        {}, // no input parameters
        {
            mgp::Return(node1_name, mgp::Type::Node),
            mgp::Return(node2_name, mgp::Type::Node),
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