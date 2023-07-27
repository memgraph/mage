#include <mgp.hpp>

#include "algorithm/create.hpp"

extern "C" int mgp_init_module(struct mgp_module *module, struct mgp_memory *memory) {
  try {
    mgp::memory = memory;

    AddProcedure(Create::Relationship, Create::kProcedureRelationship, mgp::ProcedureType::Write,
                 {mgp::Parameter(Create::kRelationshipArg1, mgp::Type::Node),
                  mgp::Parameter(Create::kRelationshipArg2, mgp::Type::String),
                  mgp::Parameter(Create::kRelationshipArg3, mgp::Type::Map),
                  mgp::Parameter(Create::kRelationshipArg4, mgp::Type::Node)},
                 {mgp::Return(Create::kResultRelationship, mgp::Type::Relationship)}, module, memory);

  } catch (const std::exception &e) {
    return 1;
  }

  return 0;
}

extern "C" int mgp_shutdown_module() { return 0; }
