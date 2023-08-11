#include <mgp.hpp>

#include "algorithm/nodes.hpp"

extern "C" int mgp_init_module(struct mgp_module *module, struct mgp_memory *memory) {
  try {
    mgp::memory = memory;

    AddProcedure(Nodes::Delete, Nodes::kProcedureDelete, mgp::ProcedureType::Write,
                 {mgp::Parameter(Nodes::kDeleteArg1, mgp::Type::Any)}, {}, module, memory);

  } catch (const std::exception &e) {
    return 1;
  }

  return 0;
}

extern "C" int mgp_shutdown_module() { return 0; }
