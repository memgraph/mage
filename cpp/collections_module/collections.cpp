#include <mgp.hpp>

#include "algorithms/split.hpp"

constexpr std::string_view kProcedureSplit = "split";

constexpr std::string_view kReturnSplit = "splitted";

constexpr std::string_view kArgumentInputList = "inputList";
constexpr std::string_view kArgumentDelimiter = "delimiter";

extern "C" int mgp_init_module(struct mgp_module *module, struct mgp_memory *memory) {
  try {
    mgp::memory = memory;

    AddProcedure(Split, kProcedureSplit, mgp::ProcedureType::Read,
                 {mgp::Parameter(kArgumentInputList, {mgp::Type::List, mgp::Type::Any}),
                  mgp::Parameter(kArgumentDelimiter, mgp::Type::Any)},
                 {mgp::Return(kReturnSplit, {mgp::Type::List, mgp::Type::Any})}, module, memory);

  } catch (const std::exception &e) {
    return 1;
  }

  return 0;
}

extern "C" int mgp_shutdown_module() { return 0; }
