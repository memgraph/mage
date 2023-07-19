#include <mgp.hpp>

#include "algorithm/algorithm.hpp"

constexpr const char *kProcedureSumLongs = "sumLongs";

constexpr const char *kReturnSumLongs = "sum";

constexpr const char *kNumbersList = "list_of_numbers";

extern "C" int mgp_init_module(struct mgp_module *module, struct mgp_memory *memory) {
  try {
    mgp::memory = memory;

    const auto parameter_list_of_any = mgp::Parameter(kNumbersList, std::make_pair(mgp::Type::List, mgp::Type::Any));

    AddProcedure(Collections::SumLongs, kProcedureSumLongs, mgp::ProcedureType::Read,
                 {parameter_list_of_any}, {mgp::Return(kReturnSumLongs, mgp::Type::Int)},
                 module, memory);

  } catch (const std::exception &e) {
    return 1;
  }

  return 0;
}

extern "C" int mgp_shutdown_module() { return 0; }