#include <cstdint>
#include <unordered_map>
#include <unordered_set>
#include <algorithm>
#include <mgp.hpp>
#include <mg_exceptions.hpp>

#include "algorithms/node_similarity.hpp"


// Methods
constexpr char const *jaccardAll = "jaccard";
constexpr char const *jaccardPairwise = "jaccard_pairwise";
constexpr char const *overlapAll = "overlap";
constexpr char const *overlapPairwise = "overlap_pairwise";
constexpr char const *cosineAll = "cosine";
constexpr char const *cosinePairwise = "cosine_pairwise";
// Parameter object names
constexpr char const *src_nodes = "src_nodes";
constexpr char const *dst_nodes = "dst_nodes";
constexpr char const *prop_vector = "property";
// Return object names
char const *node1_name = "node1";
char const *node2_name = "node2";
char const *similarity_name = "similarity";


void insert_results(const std::vector<std::tuple<mgp::Node, mgp::Node, double>> &results, const mgp::RecordFactory &record_factory) {
    for (const auto &[node1, node2, similarity]: results) {
        auto new_record = record_factory.NewRecord();
        new_record.Insert(node1_name, node1);
        new_record.Insert(node2_name, node2);
        new_record.Insert(similarity_name, similarity);
    }
}

/*
Calculates Jaccard similarity between given pairs of nodes.
*/
void JaccardPairwise(mgp_list *args, mgp_graph *memgraph_graph, mgp_result *result, mgp_memory *memory) {
    mgp::memory = memory;
    const auto record_factory = mgp::RecordFactory(result);
    const auto &arguments = mgp::List(args);
    try {
        insert_results(node_similarity_algs::CalculateSimilarityPairwise(arguments[0].ValueList(), arguments[1].ValueList(), node_similarity_util::Similarity::jaccard), record_factory);
    } catch (const mgp::ValueException &e) {
        record_factory.SetErrorMessage(e.what());
    }
}

/*
Calculates overlap similarity between given pairs of nodes.
*/
void OverlapPairwise(mgp_list *args, mgp_graph *memgraph_graph, mgp_result *result, mgp_memory *memory) {
    mgp::memory = memory;
    const auto &arguments = mgp::List(args);
    const auto record_factory = mgp::RecordFactory(result);
    try {
        insert_results(node_similarity_algs::CalculateSimilarityPairwise(arguments[0].ValueList(), arguments[1].ValueList(), node_similarity_util::Similarity::overlap), record_factory);
    } catch (const mgp::ValueException &e) {
        record_factory.SetErrorMessage(e.what());
    }
}

/*
Calculates Jaccard similarity between all pairs of nodes.
Jacc. similarity of two nodes can be calculated as len(intersection(neighbours(node1), neighbours(node2))) / len(union(neighbours(node1), neighbours(node2))) 
*/
void Jaccard(mgp_list *args, mgp_graph *memgraph_graph, mgp_result *result, mgp_memory *memory) {
    mgp::memory = memory;
    const auto record_factory = mgp::RecordFactory(result);
    try {
        insert_results(node_similarity_algs::CalculateSimilarityCartesian(mgp::Graph(memgraph_graph), node_similarity_util::Similarity::jaccard), record_factory);
    } catch (const mgp::ValueException &e) {
        record_factory.SetErrorMessage(e.what());
    }
}

/*
Calculates overlap similarity between all pairs of nodes.
Overlap similarity of two nodes can be calculated as len(intersection(neighbours(node1), neighbours(node2))) / min(len(neighbours(node1), len(node2))) 
*/
void Overlap(mgp_list *args, mgp_graph *memgraph_graph, mgp_result *result, mgp_memory *memory) {
    mgp::memory = memory;
    const auto record_factory = mgp::RecordFactory(result);
    try {
        insert_results(node_similarity_algs::CalculateSimilarityCartesian(mgp::Graph(memgraph_graph), node_similarity_util::Similarity::overlap), record_factory);
    } catch (const mgp::ValueException &e) {
        record_factory.SetErrorMessage(e.what());
    }

}

/*
Calculates cosine similarity between all pairs of nodes.
*/
void Cosine(mgp_list *args, mgp_graph *memgraph_graph, mgp_result *result, mgp_memory *memory) {
    mgp::memory = memory;
    const auto record_factory = mgp::RecordFactory(result);
    const auto &arguments = mgp::List(args);
    try {
        insert_results(node_similarity_algs::CalculateSimilarityCartesian(mgp::Graph(memgraph_graph), node_similarity_util::Similarity::cosine, arguments[0].ValueString().data()), record_factory);
    } catch (const mgp::ValueException &e) {
        record_factory.SetErrorMessage(e.what());
    }
}

/*
Calculates overlap similarity between given pairs of nodes.
*/
void CosinePairwise(mgp_list *args, mgp_graph *memgraph_graph, mgp_result *result, mgp_memory *memory) {
    mgp::memory = memory;
    const auto record_factory = mgp::RecordFactory(result);
    const auto &arguments = mgp::List(args);
    try {
        insert_results(node_similarity_algs::CalculateSimilarityPairwise(arguments[1].ValueList(), arguments[2].ValueList(), node_similarity_util::Similarity::cosine, arguments[0].ValueString().data()), record_factory);
    } catch (const mgp::ValueException &e) {
        record_factory.SetErrorMessage(e.what());
    }
}

extern "C" int mgp_init_module(struct mgp_module *module, struct mgp_memory *memory) {
    try {
        mgp::memory = memory;
        // method objects
        std::vector<mgp::Return> returns = {
            mgp::Return(node1_name, mgp::Type::Node),
            mgp::Return(node2_name, mgp::Type::Node),
            mgp::Return(similarity_name, mgp::Type::Double)
        };
        // Normal params
        std::vector<mgp::Parameter> params {mgp::Parameter(src_nodes, {mgp::Type::List, mgp::Type::Node}), mgp::Parameter(dst_nodes, {mgp::Type::List, mgp::Type::Node})};
        // Cosine params
        std::vector<mgp::Parameter> cosine_params_pairwise {mgp::Parameter(prop_vector, mgp::Type::String), mgp::Parameter(src_nodes, {mgp::Type::List, mgp::Type::Node}), mgp::Parameter(dst_nodes, {mgp::Type::List, mgp::Type::Node})};
        // Add Jaccard algorithm
        mgp::AddProcedure(Jaccard, jaccardAll, mgp::ProcedureType::Read, {}, returns, module, memory);
        mgp::AddProcedure(JaccardPairwise, jaccardPairwise, mgp::ProcedureType::Read, params, returns, module, memory);
        // Add Overlap algorithm
        mgp::AddProcedure(Overlap, overlapAll, mgp::ProcedureType::Read, {}, returns, module, memory);
        mgp::AddProcedure(OverlapPairwise, overlapPairwise, mgp::ProcedureType::Read, params, returns, module, memory);
        // // Add Cosine algorithm
        mgp::AddProcedure(Cosine, cosineAll, mgp::ProcedureType::Read, {mgp::Parameter(prop_vector, mgp::Type::String)}, 
                returns, module, memory);
        mgp::AddProcedure(CosinePairwise, cosinePairwise, mgp::ProcedureType::Read, cosine_params_pairwise, returns, module, memory);
    } catch(const std::exception &e) {
        return 1;
    } 
    return 0;
}

extern "C" int mgp_shutdown_module() { return 0; }
