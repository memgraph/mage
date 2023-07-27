#include <mgp.hpp>

#include "algorithm/algorithm.hpp"

extern "C" int mgp_init_module(struct mgp_module *module, struct mgp_memory *memory) {
  try {
    mgp::memory = memory;

    const auto parameter_list_of_any =
        mgp::Parameter(Collections::kNumbersList, std::make_pair(mgp::Type::List, mgp::Type::Any));

    AddProcedure(Collections::SumLongs, Collections::kProcedureSumLongs, mgp::ProcedureType::Read,
                 {parameter_list_of_any}, {mgp::Return(Collections::kReturnSumLongs, mgp::Type::Int)}, module, memory);

    AddProcedure(Collections::Avg, Collections::kProcedureAvg, mgp::ProcedureType::Read, {parameter_list_of_any},
                 {mgp::Return(Collections::kReturnAvg, mgp::Type::Double)}, module, memory);

    AddProcedure(Collections::ContainsAll, Collections::kProcedureContainsAll, mgp::ProcedureType::Read,
                 {mgp::Parameter(Collections::kAnyList, {mgp::Type::List, mgp::Type::Any}),
                  mgp::Parameter(Collections::kAnyList, {mgp::Type::List, mgp::Type::Any})},
                 {mgp::Return(Collections::kReturnContainsAll, mgp::Type::Bool)}, module, memory);

  } catch (const std::exception &e) {
    return 1;
  }

  return 0;
}

extern "C" int mgp_shutdown_module() { return 0; }
