#include "../mg_utility/mg_utils.hpp"
#include "hits.hpp"

namespace {
    constexpr char const *kProcedureGet = "get";

    constexpr char const *kFieldNode = "node";
    constexpr char const *kFieldHub = "rank";
    constexpr char const *kFieldAuth = "rank";

    constexpr char const *kArgumentMaxIterations = "max_iterations";
    constexpr char const *kArgumentStopEpsilon = "stop_epsilon";


    void InsertHitsRecord(mgp_graph *graph, mgp_result *result, mgp_memory *memory, const std::uint64_t node_id,
                          double hub, double auth) {
        auto *record = mgp::result_new_record(result);

        mg_utility::InsertNodeValueResult(graph, record, kFieldNode, node_id, memory);
        mg_utility::InsertDoubleValueResult(record, kFieldHub, hub, memory);
        mg_utility::InsertDoubleValueResult(record, kFieldAuth, auth, memory);
    }

    void HitsWrapper(mgp_list *args, mgp_graph *memgraph_graph, mgp_result *result, mgp_memory *memory) {
        try {
            auto max_iterations = mgp::value_get_int(mgp::list_at(args, 0));
            auto stop_epsilon = mgp::value_get_double(mgp::list_at(args, 1));

            auto graph = mg_utility::GetGraphView(memgraph_graph, result, memory, mg_graph::GraphType::kDirectedGraph);

            const auto &graph_edges = graph->Edges();
            std::vector<hits_alg::EdgePair> hits_edges;
            std::transform(graph_edges.begin(), graph_edges.end(), std::back_inserter(hits_edges),
                           [](const mg_graph::Edge<std::uint64_t> &edge) -> hits_alg::EdgePair {
                               return {edge.from, edge.to};
                           });

            auto number_of_nodes = graph->Nodes().size();

            auto hits_graph = hits_alg::HitsGraph(number_of_nodes, hits_edges.size(), hits_edges);
            auto hits =
                    hits_alg::ParallelIterativeHits(hits_graph, max_iterations, stop_epsilon);
            auto hubs = std::get<0>(hits);
            auto auth =std::get<1>(hits);
            for (std::uint64_t node_id = 0; node_id < number_of_nodes; ++node_id) {
                InsertHitsRecord(memgraph_graph, result, memory, graph->GetMemgraphNodeId(node_id), hubs[node_id],
                                 auth[node_id]);
            }
        } catch (const std::exception &e) {
            // We must not let any exceptions out of our module.
            mgp::result_set_error_msg(result, e.what());
            return;
        }
    }
}

extern "C" int mgp_init_module(struct mgp_module *module, struct mgp_memory *memory) {
    mgp_value *default_max_iterations;
    mgp_value *default_stop_epsilon;
    try {
        auto *hits_proc = mgp::module_add_read_procedure(module, kProcedureGet, HitsWrapper);
        default_max_iterations = mgp::value_make_int(100, memory);
        default_stop_epsilon = mgp::value_make_double(1e-5, memory);

        mgp::proc_add_opt_arg(hits_proc, kArgumentMaxIterations, mgp::type_int(), default_max_iterations);
        mgp::proc_add_opt_arg(hits_proc, kArgumentStopEpsilon, mgp::type_float(), default_stop_epsilon);

        // Query module output record
        mgp::proc_add_result(hits_proc, kFieldNode, mgp::type_node());
        mgp::proc_add_result(hits_proc, kFieldHub, mgp::type_float());
        mgp::proc_add_result(hits_proc, kFieldAuth, mgp::type_float());

    } catch (const std::exception &e) {
        // Destroy values if exception occurs earlier
        mgp_value_destroy(default_max_iterations);
        mgp_value_destroy(default_stop_epsilon);
        return 1;
    }

    mgp_value_destroy(default_max_iterations);
    mgp_value_destroy(default_stop_epsilon);

    return 0;
}

extern "C" int mgp_shutdown_module() { return 0; }