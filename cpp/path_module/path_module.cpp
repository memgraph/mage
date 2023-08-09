#include <mgp.hpp>

#include "algorithm/path.hpp"

extern "C" int mgp_init_module(struct mgp_module *module, struct mgp_memory *memory) {
  try {
    mgp::memory = memory;
    AddProcedure(Path::Expand, std::string(Path::kProcedureExpand).c_str(),
                 mgp::ProcedureType::Read,
                 {mgp::Parameter(std::string("Node").c_str(),mgp::Type::Node), 
                 mgp::Parameter(std::string("relationships").c_str(),{mgp::Type::List, mgp::Type::String}),
                 mgp::Parameter(std::string("labels").c_str(),{mgp::Type::List, mgp::Type::String}),
                 mgp::Parameter(std::string("min_hops").c_str(),mgp::Type::Int),
                 mgp::Parameter(std::string("max_hops").c_str(),mgp::Type::Int)},
                 {mgp::Return(std::string("result").c_str(), mgp::Type::Path)},
                 module, memory);

  } catch (const std::exception &e) {
    return 1;
  }

  return 0;
}

extern "C" int mgp_shutdown_module() { return 0; }