#include <cstddef>
#include <mgp.hpp>

#include "algorithm/nodes.hpp"

extern "C" int mgp_init_module(struct mgp_module *module, struct mgp_memory *memory) {
  try {
    mgp::memory = memory;
    AddProcedure(Nodes::Link, std::string(Nodes::kProcedureLink).c_str(), mgp::ProcedureType::Write,
                 {mgp::Parameter(std::string(Nodes::kArgumentNodesLink).c_str(), {mgp::Type::List, mgp::Type::Node}),
                  mgp::Parameter(std::string(Nodes::kArgumentTypeLink).c_str(), mgp::Type::String)},
                 {}, module, memory);

    AddProcedure(
        Nodes::RelationshipTypes, Nodes::kProcedureRelationshipTypes, mgp::ProcedureType::Read,
        {mgp::Parameter(Nodes::kRelationshipTypesArg1, mgp::Type::Any),
         mgp::Parameter(Nodes::kRelationshipTypesArg2, {mgp::Type::List, mgp::Type::String}, mgp::Value(mgp::List{}))},
        {mgp::Return(Nodes::kResultRelationshipTypes, {mgp::Type::List, mgp::Type::Map})}, module, memory);

    AddProcedure(Nodes::Delete, Nodes::kProcedureDelete, mgp::ProcedureType::Write,
                 {mgp::Parameter(Nodes::kDeleteArg1, mgp::Type::Any)}, {}, module, memory);

  } catch (const std::exception &e) {
    return 1;
  }

  return 0;
}

extern "C" int mgp_shutdown_module() { return 0; }
