#include <mgp.hpp>

#include "algorithm/path.hpp"

extern "C" int mgp_init_module(struct mgp_module *module, struct mgp_memory *memory) {
  try {
    mgp::memory = memory;

    AddProcedure(Path::Create, Path::kProcedureCreate, mgp::ProcedureType::Read,
                {mgp::Parameter(Path::kCreateArg1, mgp::Type::Node),
                mgp::Parameter(Path::kCreateArg2, {mgp::Type::List, mgp::Type::Relationship}, mgp::Value(mgp::List{}))},
                {mgp::Return(Path::kResultCreate, mgp::Type::Path)}, module, memory);

  } catch (const std::exception &e) {
    return 1;
  }

  return 0;
}

extern "C" int mgp_shutdown_module() { return 0; }