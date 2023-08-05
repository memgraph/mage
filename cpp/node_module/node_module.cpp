#include <mgp.hpp>

#include "algorithm/node.hpp"

extern "C" int mgp_init_module(struct mgp_module *module, struct mgp_memory *memory) {
  try {
    mgp::memory = memory;
    AddProcedure(Node::RelExists, Node::kProcedureRelExists, mgp::ProcedureType::Read,
                 {mgp::Parameter(Node::kArgumentsNode, mgp::Type::Node),
                  mgp::Parameter(Node::kArgumentsPattern, mgp::Type::String, "")},
                 {mgp::Return(Node::kReturnRelExists, mgp::Type::Bool)}, module, memory);

  } catch (const std::exception &e) {
    return 1;
  }

  return 0;
}

extern "C" int mgp_shutdown_module() { return 0; }
