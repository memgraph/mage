#include <mgp.hpp>

#include "algorithm/path.hpp"

extern "C" int mgp_init_module(struct mgp_module *module, struct mgp_memory *memory) {
  try {
    mgp::memory = memory;

    auto empty_list = mgp::Value(mgp::List{});
    auto empty_map = mgp::Map{};
    empty_map.Insert("key", empty_list);

    AddProcedure(Path::Create, Path::kProcedureCreate, mgp::ProcedureType::Read,
                 {mgp::Parameter(Path::kCreateArg1, mgp::Type::Node),
                  mgp::Parameter(Path::kCreateArg2, {mgp::Type::Map, mgp::Type::List}, mgp::Value(empty_map))},
                 {mgp::Return(Path::kResultCreate, mgp::Type::Path)}, module, memory);

  } catch (const std::exception &e) {
    return 1;
  }

  return 0;
}

extern "C" int mgp_shutdown_module() { return 0; }