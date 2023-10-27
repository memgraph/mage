#include <mgp.hpp>

#include "algorithm/algo.hpp"

extern "C" int mgp_init_module(struct mgp_module *module, struct mgp_memory *memory) {
  try {
    mgp::MemoryDispatcherGuard guard{memory};
    AddProcedure(Algo::Cover, std::string(Algo::kProcedureCover).c_str(), mgp::ProcedureType::Read,
                 {mgp::Parameter(std::string(Algo::kCoverArg1).c_str(), {mgp::Type::List, mgp::Type::Node})},
                 {mgp::Return(std::string(Algo::kCoverRet1).c_str(), mgp::Type::Relationship)}, module, memory);

  } catch (const std::exception &e) {
    return 1;
  }

  return 0;
}

extern "C" int mgp_shutdown_module() { return 0; }
