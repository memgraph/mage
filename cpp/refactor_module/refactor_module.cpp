#include <mgp.hpp>

#include "algorithm/refactor.hpp"

extern "C" int mgp_init_module(struct mgp_module *module, struct mgp_memory *memory) {
  try {
    mgp::memory = memory;
     AddProcedure(Refactor::CollapseNode, "collapse_node", mgp::ProcedureType::Write,
                 {mgp::Parameter("node",  mgp::Type::Node)},
                 {}, module, memory);

  } catch (const std::exception &e) {
    return 1;
  }

  return 0;
}

extern "C" int mgp_shutdown_module() { return 0; }
