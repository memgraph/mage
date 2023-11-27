#include <mgp.hpp>

#include "algorithm/text.hpp"

extern "C" int mgp_init_module(struct mgp_module *module, struct mgp_memory *memory) {
  try {
    mgp::MemoryDispatcherGuard guard{memory};;

    AddProcedure(Text::Join, Text::kProcedureJoin, mgp::ProcedureType::Read,
                 {mgp::Parameter(Text::kJoinArg1, {mgp::Type::List, mgp::Type::String}),
                  mgp::Parameter(Text::kJoinArg2, mgp::Type::String)},
                 {mgp::Return(Text::kResultJoin, mgp::Type::String)}, module, memory);

    AddProcedure(Text::Format, Text::kProcedureFormat, mgp::ProcedureType::Read,
                 {mgp::Parameter(Text::kStringToFormat, mgp::Type::String),
                  mgp::Parameter(Text::kParameters, {mgp::Type::List, mgp::Type::Any})},
                 {mgp::Return(Text::kResultFormat, mgp::Type::String)}, module, memory);

  } catch (const std::exception &e) {
    return 1;
  }

  return 0;
}

extern "C" int mgp_shutdown_module() { return 0; }
