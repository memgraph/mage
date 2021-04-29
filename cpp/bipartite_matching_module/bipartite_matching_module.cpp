#include <mg_procedure.h>
#include <mg_exceptions.hpp>
#include <mg_utils.hpp>

#include "algorithm/bipartite_matching.hpp"

namespace {

constexpr char const *kFieldMatchingNumber = "maximum_bipartite_matching";

void InsertBipartiteMatchingRecord(mgp_result *result, mgp_memory *memory, const int matching_number) {
  auto *record = mgp_result_new_record(result);
  if (record == nullptr) {
    throw mg_exception::NotEnoughMemoryException();
  }

  mg_utility::InsertIntValueResult(record, kFieldMatchingNumber, matching_number, memory);
}

void GetMaximumBipartiteMatching(const mgp_list *args, const mgp_graph *memgraph_graph, mgp_result *result,
                                 mgp_memory *memory) {
  try {
    auto graph = mg_utility::GetGraphView(memgraph_graph, result, memory);
    auto maximum_bipartite_matching = bipartite_matching_alg::BipartiteMatching(*graph);

    InsertBipartiteMatchingRecord(result, memory, maximum_bipartite_matching);
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
  mgp_proc *proc = mgp_module_add_read_procedure(module, "max", GetMaximumBipartiteMatching);
  if (!proc) return 1;

  if (!mgp_proc_add_result(proc, kFieldMatchingNumber, mgp_type_int())) return 1;

  return 0;
}

// This is an optional function if you need to release any resources before the
// module is unloaded. You will probably need this if you acquired some
// resources in mgp_init_module.
extern "C" int mgp_shutdown_module() {
  // Return 0 to indicate success.
  return 0;
}