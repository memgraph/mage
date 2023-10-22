#include <mgp.hpp>

#include "algorithm/algo.hpp"

extern "C" int mgp_init_module(struct mgp_module *module, struct mgp_memory *memory) {
  try {
    mgp::MemoryDispatcherGuard guard{memory};

    AddProcedure(Algo::AStar, std::string(Algo::kProcedureAStar).c_str(), mgp::ProcedureType::Read,
                 {mgp::Parameter(std::string(Algo::kAStarStart).c_str(), mgp::Type::Node),
                  mgp::Parameter(std::string(Algo::kAStarTarget).c_str(), mgp::Type::Node),
                  mgp::Parameter(std::string(Algo::kAStarConfig).c_str(), mgp::Type::Map)},
                 {mgp::Return(std::string(Algo::kAStarPath).c_str(), mgp::Type::Path),
                  mgp::Return(std::string(Algo::kAStarWeight).c_str(), mgp::Type::Double)},
                 module, memory);

  } catch (const std::exception &e) {
    return 1;
  }

  return 0;
}

extern "C" int mgp_shutdown_module() { return 0; }
