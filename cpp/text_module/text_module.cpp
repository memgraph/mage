#include <mgp.hpp>

#include "algorithm/text.hpp"

extern "C" int mgp_init_module(struct mgp_module *module, struct mgp_memory *memory) {
  try {
    mgp::MemoryDispatcherGuard guard{memory};

    AddProcedure(Text::Join, Text::kProcedureJoin, mgp::ProcedureType::Read,
                 {mgp::Parameter(Text::kJoinArg1, {mgp::Type::List, mgp::Type::String}),
                  mgp::Parameter(Text::kJoinArg2, mgp::Type::String)},
                 {mgp::Return(Text::kResultJoin, mgp::Type::String)}, module, memory);

    AddProcedure(Text::Format, Text::kProcedureFormat, mgp::ProcedureType::Read,
                 {mgp::Parameter(Text::kStringToFormat, mgp::Type::String),
                  mgp::Parameter(Text::kParameters, {mgp::Type::List, mgp::Type::Any})},
                 {mgp::Return(Text::kResultFormat, mgp::Type::String)}, module, memory);

    AddProcedure(Text::RegexGroups, Text::kProcedureRegexGroups, mgp::ProcedureType::Read,
                 {mgp::Parameter(Text::kInput, mgp::Type::String),
                  mgp::Parameter(Text::kRegex, mgp::Type::String)},
                 {mgp::Return(Text::kResultRegexGroups, {mgp::Type::List, mgp::Type::Any})}, module, memory);

    mgp::AddFunction(Text::Replace, Text::kProcedureReplace,
                {mgp::Parameter(Text::kText, mgp::Type::String),
                 mgp::Parameter(Text::kRegex, mgp::Type::String),
                 mgp::Parameter(Text::kReplacement, mgp::Type::String)},
                module, memory);

    mgp::AddFunction(Text::RegReplace, Text::kProcedureRegReplace,
                {mgp::Parameter(Text::kText, mgp::Type::String),
                 mgp::Parameter(Text::kRegex, mgp::Type::String),
                 mgp::Parameter(Text::kReplacement, mgp::Type::String)},
                module, memory);

    mgp::AddFunction(Text::Distance, Text::kProcedureDistance,
                {mgp::Parameter(Text::kText1, mgp::Type::String),
                 mgp::Parameter(Text::kText2, mgp::Type::String)},
                module, memory);

    AddProcedure(Text::IndexOf, Text::kProcedureIndexOf, mgp::ProcedureType::Read,
                 {mgp::Parameter(Text::kIndexOfText, mgp::Type::String),
                  mgp::Parameter(Text::kIndexOfLookup, mgp::Type::String),
                  mgp::Parameter(Text::kIndexOfFrom, mgp::Type::Int, int64_t(0)),
                  mgp::Parameter(Text::kIndexOfTo, mgp::Type::Int, int64_t(-1))},
                 {mgp::Return(Text::kResultIndexOf, mgp::Type::Int)}, module, memory);
  } catch (const std::exception &e) {
    return 1;
  }

  return 0;
}

extern "C" int mgp_shutdown_module() { return 0; }
