#include <mgp.hpp>

#include "algorithm/path.hpp"

extern "C" int mgp_init_module(struct mgp_module *module, struct mgp_memory *memory) {
  try {
    mgp::memory = memory;
    AddProcedure(
        Path::Expand, std::string(Path::kProcedureExpand).c_str(), mgp::ProcedureType::Read,
        {mgp::Parameter(std::string(Path::kArgumentStartExpand).c_str(), mgp::Type::Any),
         mgp::Parameter(std::string(Path::kArgumentRelationshipsExpand).c_str(), {mgp::Type::List, mgp::Type::String}),
         mgp::Parameter(std::string(Path::kArgumentLabelsExpand).c_str(), {mgp::Type::List, mgp::Type::String}),
         mgp::Parameter(std::string(Path::kArgumentMinHopsExpand).c_str(), mgp::Type::Int),
         mgp::Parameter(std::string(Path::kArgumentMaxHopsExpand).c_str(), mgp::Type::Int)},
        {mgp::Return(std::string(Path::kResultExpand).c_str(), mgp::Type::Path)}, module, memory);

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
