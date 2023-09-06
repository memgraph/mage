#include <mgp.hpp>

#include "algorithm/meta.hpp"

extern "C" int mgp_init_module(struct mgp_module *module, struct mgp_memory *memory) {
  try {
    mgp::memory = memory;

    AddProcedure(Meta::Update, Meta::kProcedureUpdate, mgp::ProcedureType::Read,
                 {mgp::Parameter(Meta::kUpdateArg1, {mgp::Type::List, mgp::Type::Map}),
                  mgp::Parameter(Meta::kUpdateArg2, {mgp::Type::List, mgp::Type::Map}),
                  mgp::Parameter(Meta::kUpdateArg3, {mgp::Type::List, mgp::Type::Map}),
                  mgp::Parameter(Meta::kUpdateArg4, {mgp::Type::List, mgp::Type::Map}),
                  mgp::Parameter(Meta::kUpdateArg5, {mgp::Type::List, mgp::Type::Map}),
                  mgp::Parameter(Meta::kUpdateArg6, {mgp::Type::List, mgp::Type::Map})},
                 {}, module, memory);

    AddProcedure(Meta::StatsOnline, Meta::kProcedureStatsOnline, mgp::ProcedureType::Read,
                 {mgp::Parameter(Meta::kStatsOnlineArg1, mgp::Type::Bool, false)},
                 {mgp::Return(Meta::kReturnStats1, mgp::Type::Int), mgp::Return(Meta::kReturnStats2, mgp::Type::Int),
                  mgp::Return(Meta::kReturnStats3, mgp::Type::Int), mgp::Return(Meta::kReturnStats4, mgp::Type::Int),
                  mgp::Return(Meta::kReturnStats5, mgp::Type::Int),
                  mgp::Return(Meta::kReturnStats6, {mgp::Type::Map, mgp::Type::Int}),
                  mgp::Return(Meta::kReturnStats7, {mgp::Type::Map, mgp::Type::Int}),
                  mgp::Return(Meta::kReturnStats8, {mgp::Type::Map, mgp::Type::Int}),
                  mgp::Return(Meta::kReturnStats9, {mgp::Type::Map, mgp::Type::Int})},
                 module, memory);

    AddProcedure(Meta::StatsOffline, Meta::kProcedureStatsOffline, mgp::ProcedureType::Read, {},
                 {mgp::Return(Meta::kReturnStats1, mgp::Type::Int), mgp::Return(Meta::kReturnStats2, mgp::Type::Int),
                  mgp::Return(Meta::kReturnStats3, mgp::Type::Int), mgp::Return(Meta::kReturnStats4, mgp::Type::Int),
                  mgp::Return(Meta::kReturnStats5, mgp::Type::Int),
                  mgp::Return(Meta::kReturnStats6, {mgp::Type::Map, mgp::Type::Int}),
                  mgp::Return(Meta::kReturnStats7, {mgp::Type::Map, mgp::Type::Int}),
                  mgp::Return(Meta::kReturnStats8, {mgp::Type::Map, mgp::Type::Int}),
                  mgp::Return(Meta::kReturnStats9, {mgp::Type::Map, mgp::Type::Int})},
                 module, memory);

    AddProcedure(Meta::Reset, Meta::kProcedureReset, mgp::ProcedureType::Read, {}, {}, module, memory);

  } catch (const std::exception &e) {
    return 1;
  }

  return 0;
}

extern "C" int mgp_shutdown_module() { return 0; }
