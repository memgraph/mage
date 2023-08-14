#include <mgp.hpp>

#include "algorithm/node.hpp"

extern "C" int mgp_init_module(struct mgp_module *module, struct mgp_memory *memory) {
  try {
    mgp::memory = memory;
    AddProcedure(
        Node::RelExists, Node::kProcedureRelExists, mgp::ProcedureType::Read,
        {mgp::Parameter(Node::kArgumentsNode, mgp::Type::Node),
         mgp::Parameter(Node::kArgumentsPattern, {mgp::Type::List, mgp::Type::String}, mgp::Value(mgp::List{}))},
        {mgp::Return(Node::kReturnRelExists, mgp::Type::Bool)}, module, memory);

    AddProcedure(
        Node::RelationshipTypes, Node::kProcedureRelationshipTypes, mgp::ProcedureType::Read,
        {mgp::Parameter(Node::kRelationshipTypesArg1, mgp::Type::Node),
         mgp::Parameter(Node::kRelationshipTypesArg2, {mgp::Type::List, mgp::Type::String}, mgp::Value(mgp::List{}))},
        {mgp::Return(Node::kResultRelationshipTypes, {mgp::Type::List, mgp::Type::String})}, module, memory);

  } catch (const std::exception &e) {
    return 1;
  }

  return 0;
}

extern "C" int mgp_shutdown_module() { return 0; }
