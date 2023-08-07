#include <mgp.hpp>

#include "algorithm/node.hpp"

extern "C" int mgp_init_module(struct mgp_module *module, struct mgp_memory *memory) {
  try {
    mgp::memory = memory;
    AddProcedure(Node::RelationshipsExist, std::string(Node::kProcedureRelationshipsExist).c_str(),
                 mgp::ProcedureType::Read,
                 {mgp::Parameter(std::string(Node::kArgumentNodesRelationshipsExist).c_str(), mgp::Type::Node),
                  mgp::Parameter(std::string(Node::kArgumentRelationshipsRelationshipsExist).c_str(),
                                 {mgp::Type::List, mgp::Type::String})},
                 {mgp::Return(std::string(Node::kReturnRelationshipsExist).c_str(), {mgp::Type::Map, mgp::Type::Any})},
                 module, memory);

  } catch (const std::exception &e) {
    return 1;
  }

  return 0;
}

extern "C" int mgp_shutdown_module() { return 0; }
