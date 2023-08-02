#include <mgp.hpp>

#include "algorithm/date.hpp"

extern "C" int mgp_init_module(struct mgp_module *module, struct mgp_memory *memory) {
  try {
    mgp::memory = memory;
    AddProcedure(Date::Parse, Date::kProcedureParse, mgp::ProcedureType::Read,
                 {mgp::Parameter(Date::kArgumentsTime, mgp::Type::String),
                  mgp::Parameter(Date::kArgumentsUnit, mgp::Type::String),
                  mgp::Parameter(Date::kArgumentsFormat, mgp::Type::String),
                  mgp::Parameter(Date::kArgumentsTimezone, mgp::Type::String)},
                 {mgp::Return(Date::kReturnParsed, mgp::Type::Int)}, module, memory);

  } catch (const std::exception &e) {
    return 1;
  }

  return 0;
}

extern "C" int mgp_shutdown_module() { return 0; }
