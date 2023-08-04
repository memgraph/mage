#include <mgp.hpp>

#include "algorithm/nodes.hpp"

extern "C" int mgp_init_module(struct mgp_module *module, struct mgp_memory *memory) {
  try {
    mgp::memory = memory;
    AddProcedure(Nodes::Link, std::string(Nodes::kProcedureLink).c_str(),
                 mgp::ProcedureType::Write,
                 {mgp::Parameter(std::string(Nodes::kArgumentNodesLink).c_str(), {mgp::Type::List, mgp::Type::Node}),
                 mgp::Parameter(std::string(Nodes::kArgumentTypeLink).c_str(),mgp::Type::String)}, {},module, memory);

  } catch (const std::exception &e) {
    return 1;
  }

  return 0;
}

extern "C" int mgp_shutdown_module() { return 0; }