#include <iostream>
#include <map>
#include <optional>
#include <sstream>
#include <unordered_set>

#include "algorithms/algorithms.hpp"
#include "data_structures/graph.hpp"
#include "mg_interface.hpp"
#include "mg_procedure.h"

namespace {

const char *fieldEdgeID = "edge_id";
const char *fieldNodeFrom = "node_from";
const char *fieldNodeTo = "node_to";

bool InsertBridgeRecord(const mgp_graph *graph, mgp_result *result,
                        mgp_memory *memory, const int edge_id,
                        const int node_from_id, const int node_to_id) {
  mgp_result_record *record = mgp_result_new_record(result);
  if (record == nullptr) return false;

  bool result_inserted =
      mg_interface::InsertIntValue(record, fieldEdgeID, edge_id, memory);
  if (!result_inserted) return false;

  result_inserted = mg_interface::InsertNodeValue(graph, record, fieldNodeFrom,
                                                  node_from_id, memory);
  if (!result_inserted) return false;

  result_inserted = mg_interface::InsertNodeValue(graph, record, fieldNodeTo,
                                                  node_to_id, memory);
  if (!result_inserted) return false;

  return true;
}

void GetBridges(const mgp_list *args, const mgp_graph *graph,
                mgp_result *result, mgp_memory *memory) {
  try {
    graphdata::Graph g;
    std::map<uint32_t, uint32_t> node_mapping;
    std::map<uint32_t, uint32_t> edge_mapping;
    mg_interface::GetGraphView(&g, node_mapping, edge_mapping, graph, result,
                               memory);
    std::vector<graphdata::Edge> bridges = algorithms::GetBridges(g);

    for (const auto &bridge_edge : bridges) {
      bool record_inserted = InsertBridgeRecord(
          graph, result, memory, edge_mapping[bridge_edge.id], bridge_edge.from,
          bridge_edge.to);

      if (!record_inserted) {
        mg_interface::NotEnoughMemory(result);
        return;
      }
    }
  } catch (const std::exception &e) {
    // We must not let any exceptions out of our module.
    mgp_result_set_error_msg(result, e.what());
    return;
  }
}
}  // namespace

// Each module needs to define mgp_init_module function.
// Here you can register multiple procedures your module supports.
extern "C" int mgp_init_module(mgp_module *module, mgp_memory *memory) {
  mgp_proc *proc = mgp_module_add_read_procedure(module, "bridges", GetBridges);

  if (!proc) return 1;

  if (!mgp_proc_add_result(proc, fieldEdgeID, mgp_type_int())) return 1;
  if (!mgp_proc_add_result(proc, fieldNodeFrom, mgp_type_node())) return 1;
  if (!mgp_proc_add_result(proc, fieldNodeTo, mgp_type_node())) return 1;

  return 0;
}

// This is an optional function if you need to release any resources before the
// module is unloaded. You will probably need this if you acquired some
// resources in mgp_init_module.
extern "C" int mgp_shutdown_module() {
  // Return 0 to indicate success.
  return 0;
}