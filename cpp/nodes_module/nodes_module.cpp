#include <mgp.hpp>

#include "algorithm/nodes.hpp"

extern "C" int mgp_init_module(struct mgp_module *module, struct mgp_memory *memory) {
  try {
    mgp::memory = memory;
    AddProcedure(Nodes::RelationshipsExist, std::string(Nodes::kProcedureRelationshipsExist).c_str(),
                 mgp::ProcedureType::Read,
                 {mgp::Parameter(std::string(Nodes::kArgumentNodesRelationshipsExist).c_str(),
                                 {mgp::Type::List, mgp::Type::Any}),
                  mgp::Parameter(std::string(Nodes::kArgumentRelationshipsRelationshipsExist).c_str(),
                                 {mgp::Type::List, mgp::Type::String})},
                 {mgp::Return(std::string(Nodes::kReturnRelationshipsExist).c_str(), {mgp::Type::Map, mgp::Type::Any})},
                 module, memory);

  } catch (const std::exception &e) {
    return 1;
  }

  return 0;
}

extern "C" int mgp_shutdown_module() { return 0; }
