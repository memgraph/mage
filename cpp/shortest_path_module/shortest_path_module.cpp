#include <mg_utils.hpp>

void KShortestPath(mgp_list *args, mgp_graph *memgraph_graph, mgp_result *result, mgp_memory *memory);

extern "C" int mgp_init_module(struct mgp_module *module, struct mgp_memory *memory);

extern "C" int mgp_shutdown_module() { return 0; }