#include <mgp.hpp>

#include "algorithm_online/meta.hpp"

extern "C" int mgp_init_module(struct mgp_module *module, struct mgp_memory *memory) {
  try {
    mgp::memory = memory;

    AddProcedure(Meta::Update, Meta::kProcedureUpdate, mgp::ProcedureType::Read,
                 {mgp::Parameter(Meta::kStatsArg1, {mgp::Type::List, mgp::Type::Map}),
                  mgp::Parameter(Meta::kStatsArg2, {mgp::Type::List, mgp::Type::Map}),
                  mgp::Parameter(Meta::kStatsArg3, {mgp::Type::List, mgp::Type::Map}),
                  mgp::Parameter(Meta::kStatsArg4, {mgp::Type::List, mgp::Type::Map}),
                  mgp::Parameter(Meta::kStatsArg5, {mgp::Type::List, mgp::Type::Map}),
                  mgp::Parameter(Meta::kStatsArg6, {mgp::Type::List, mgp::Type::Map})},
                 {}, module, memory);

    AddProcedure(Meta::Stats, Meta::kProcedureStats, mgp::ProcedureType::Read, {},
                 {mgp::Return(Meta::kReturnStats1, mgp::Type::Int), mgp::Return(Meta::kReturnStats2, mgp::Type::Int),
                  mgp::Return(Meta::kReturnStats3, mgp::Type::Int), mgp::Return(Meta::kReturnStats4, mgp::Type::Int),
                  mgp::Return(Meta::kReturnStats5, mgp::Type::Int), mgp::Return(Meta::kReturnStats6, mgp::Type::Map),
                  mgp::Return(Meta::kReturnStats7, mgp::Type::Map), mgp::Return(Meta::kReturnStats8, mgp::Type::Map),
                  mgp::Return(Meta::kReturnStats9, mgp::Type::Map)},
                 module, memory);

    AddProcedure(Meta::Reset, Meta::kProcedureReset, mgp::ProcedureType::Read, {}, {}, module, memory);

  } catch (const std::exception &e) {
    return 1;
  }

  return 0;
}

extern "C" int mgp_shutdown_module() { return 0; }
