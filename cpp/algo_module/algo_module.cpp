#include <mgp.hpp>

#include "algorithm/algo.hpp"

extern "C" int mgp_init_module(struct mgp_module *module, struct mgp_memory *memory) {
  try {
    mgp::MemoryDispatcherGuard guard{memory};

    AddProcedure(Algo::AStar, "astar", mgp::ProcedureType::Read,
                 {mgp::Parameter("start", mgp::Type::Node),
                 mgp::Parameter("target", mgp::Type::Node),
                 mgp::Parameter("config", mgp::Type::Map)},
                 {mgp::Return("result", mgp::Type::Path),
                 mgp::Return("weight", mgp::Type::Double)}, module, memory);

  } catch (const std::exception &e) {
    return 1;
  }

  return 0;
}

extern "C" int mgp_shutdown_module() { return 0; }
