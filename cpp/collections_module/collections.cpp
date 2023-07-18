#include <mgp.hpp>

#include "algorithm/algorithm.hpp"

constexpr const char *kProcedureAvg = "avg";

constexpr const char *kReturnAvg = "average";

constexpr const char *kNumbersList = "list_of_numbers";

extern "C" int mgp_init_module(struct mgp_module *module, struct mgp_memory *memory) {
  try {
    mgp::memory = memory;

    const auto avg_return = mgp::Type::Double;
    const auto avg_parameter_type = mgp::Parameter(kNumbersList, std::make_pair(mgp::Type::List, mgp::Type::Any));

    AddProcedure(Collections::Avg, kProcedureAvg, mgp::ProcedureType::Read,
                 {avg_parameter_type}, {mgp::Return(kReturnAvg, avg_return)},
                 module, memory);

  } catch (const std::exception &e) {
    return 1;
  }

  return 0;
}

extern "C" int mgp_shutdown_module() { return 0; }
