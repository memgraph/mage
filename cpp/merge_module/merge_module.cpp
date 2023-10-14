#include <mgp.hpp>

#include "algorithm/merge.hpp"

extern "C" int mgp_init_module(struct mgp_module *module, struct mgp_memory *memory) {
  try {
    mgp::MemoryDispatcherGuard guard{memory};

    AddProcedure(Merge::Relationship, Merge::kProcedureRelationship, mgp::ProcedureType::Write,
                 {mgp::Parameter(Merge::kRelationshipArg1, mgp::Type::Node),
                  mgp::Parameter(Merge::kRelationshipArg2, mgp::Type::String),
                  mgp::Parameter(Merge::kRelationshipArg3, mgp::Type::Map),
                  mgp::Parameter(Merge::kRelationshipArg4, mgp::Type::Map),
                  mgp::Parameter(Merge::kRelationshipArg5, mgp::Type::Node),
                  mgp::Parameter(Merge::kRelationshipArg6, mgp::Type::Map)},
                 {mgp::Return(Merge::kRelationshipResult, mgp::Type::Relationship)}, module, memory);

  } catch (const std::exception &e) {
    return 1;
  }

  return 0;
}

extern "C" int mgp_shutdown_module() { return 0; }