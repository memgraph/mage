#include <cstdint>
#include <unordered_map>
#include <unordered_set>
#include <algorithm>
#include <mgp.hpp>
#include <mg_exceptions.hpp>
#include "algorithms/node_similarity.hpp"

void insert_results(const std::vector<std::tuple<mgp::Node, mgp::Node, double>> &results, const mgp::RecordFactory &record_factory) {
    for (const auto &[node1, node2, similarity]: results) {
        auto new_record = record_factory.NewRecord();
        new_record.Insert(node_similarity_util::node1_name, node1);
        new_record.Insert(node_similarity_util::node2_name, node2);
        new_record.Insert(node_similarity_util::similarity, similarity);
    }
}

/*
Calculates Jaccard similarity between given pairs of nodes.
*/
void JaccardPairwise(mgp_list *args, mgp_graph *memgraph_graph, mgp_result *result, mgp_memory *memory) {
    mgp::memory = memory;
    const auto &arguments = mgp::List(args);
    insert_results(node_similarity_algs::CalculateSimilarityPairwise(arguments[0].ValueList(), arguments[1].ValueList(), node_similarity_util::Similarity::jaccard), mgp::RecordFactory(result));
}

/*
Calculates overlap similarity between given pairs of nodes.
*/
void OverlapPairwise(mgp_list *args, mgp_graph *memgraph_graph, mgp_result *result, mgp_memory *memory) {
    mgp::memory = memory;
    const auto &arguments = mgp::List(args);
    insert_results(node_similarity_algs::CalculateSimilarityPairwise(arguments[0].ValueList(), arguments[1].ValueList(), node_similarity_util::Similarity::overlap), mgp::RecordFactory(result));
}

/*
Calculates Jaccard similarity between all pairs of nodes.
Jacc. similarity of two nodes can be calculated as len(intersection(neighbours(node1), neighbours(node2))) / len(union(neighbours(node1), neighbours(node2))) 
*/
void Jaccard(mgp_list *args, mgp_graph *memgraph_graph, mgp_result *result, mgp_memory *memory) {
    mgp::memory = memory;
    insert_results(node_similarity_algs::CalculateSimilarityCartesian(mgp::Graph(memgraph_graph), node_similarity_util::Similarity::jaccard), mgp::RecordFactory(result));
}

/*
Calculates overlap similarity between all pairs of nodes.
Overlap similarity of two nodes can be calculated as len(intersection(neighbours(node1), neighbours(node2))) / min(len(neighbours(node1), len(node2))) 
*/
void Overlap(mgp_list *args, mgp_graph *memgraph_graph, mgp_result *result, mgp_memory *memory) {
    mgp::memory = memory;
    insert_results(node_similarity_algs::CalculateSimilarityCartesian(mgp::Graph(memgraph_graph), node_similarity_util::Similarity::overlap), mgp::RecordFactory(result));

}

// /*
// Calculates cosine similarity between all pairs of nodes.
// */
void Cosine(mgp_list *args, mgp_graph *memgraph_graph, mgp_result *result, mgp_memory *memory) {
    mgp::memory = memory;
    const auto& arguments = mgp::List(args);
    insert_results(node_similarity_algs::CalculateSimilarityCartesian(mgp::Graph(memgraph_graph), node_similarity_util::Similarity::cosine, std::string(arguments[0].ValueString())), mgp::RecordFactory(result));
}

/*
Calculates overlap similarity between given pairs of nodes.
*/
void CosinePairwise(mgp_list *args, mgp_graph *memgraph_graph, mgp_result *result, mgp_memory *memory) {
    mgp::memory = memory;
    const auto &arguments = mgp::List(args);
    insert_results(node_similarity_algs::CalculateSimilarityPairwise(arguments[1].ValueList(), arguments[2].ValueList(), node_similarity_util::Similarity::cosine, std::string(arguments[0].ValueString())), mgp::RecordFactory(result));
}

extern "C" int mgp_init_module(struct mgp_module *module, struct mgp_memory *memory) {
    try {
        mgp::memory = memory;
        // method objects
        std::vector<mgp::Return> returns = {
            mgp::Return(node_similarity_util::node1_name, mgp::Type::Node),
            mgp::Return(node_similarity_util::node2_name, mgp::Type::Node),
            mgp::Return(node_similarity_util::similarity, mgp::Type::Double)
        };
        // Normal params
        std::vector<mgp::Parameter> params {mgp::Parameter(node_similarity_util::src_nodes, {mgp::Type::List, mgp::Type::Node}), mgp::Parameter(node_similarity_util::dst_nodes, {mgp::Type::List, mgp::Type::Node})};
        // Cosine params
        std::vector<mgp::Parameter> cosine_params_pairwise {mgp::Parameter(node_similarity_util::prop_vector, mgp::Type::String), mgp::Parameter(node_similarity_util::src_nodes, {mgp::Type::List, mgp::Type::Node}), mgp::Parameter(node_similarity_util::dst_nodes, {mgp::Type::List, mgp::Type::Node})};
        // Add Jaccard algorithm
        mgp::AddProcedure(Jaccard, node_similarity_util::jaccardAll, mgp::ProcedureType::Read, {}, returns, module, memory);
        mgp::AddProcedure(JaccardPairwise, node_similarity_util::jaccardPairwise, mgp::ProcedureType::Read, params, returns, module, memory);
        // Add Overlap algorithm
        mgp::AddProcedure(Overlap, node_similarity_util::overlapAll, mgp::ProcedureType::Read, {}, returns, module, memory);
        mgp::AddProcedure(OverlapPairwise, node_similarity_util::overlapPairwise, mgp::ProcedureType::Read, params, returns, module, memory);
        // // Add Cosine algorithm
        mgp::AddProcedure(Cosine, node_similarity_util::cosineAll, mgp::ProcedureType::Read, {mgp::Parameter(node_similarity_util::prop_vector, mgp::Type::String)}, 
                returns, module, memory);
        mgp::AddProcedure(CosinePairwise, node_similarity_util::cosinePairwise, mgp::ProcedureType::Read, cosine_params_pairwise, returns, module, memory);
    } catch(const mgp::ValueException &e) {
        std::cout << "catched 1" << std::endl;
        return 1;
    }
    catch(const std::exception &e) {
        std::cout << "catched 2" << std::endl;
        return 1;
    } catch(...){
        std::cout << "Catched somewhere" << std::endl;
    } 
    return 0;
}

extern "C" int mgp_shutdown_module() { return 0; }