#include <mgp.hpp>
#include "algorithm/algorithm.hpp"

extern "C" int mgp_init_module(struct mgp_module *module, struct mgp_memory *memory) {
  try {
    mgp::memory = memory;
    AddProcedure(
        Collections::Partition, std::string(Collections::kProcedurePartition).c_str(), mgp::ProcedureType::Read,
        {mgp::Parameter(std::string(Collections::kArgumentListPartition).c_str(), {mgp::Type::List, mgp::Type::Any}),
        mgp::Parameter(std::string(Collections::kArgumentListPartition).c_str(), mgp::Type::Int)},
        {mgp::Return(std::string(Collections::kReturnValuePartition).c_str(), {mgp::Type::List, mgp::Type::Any})}, module, memory);
   

  } catch (const std::exception &e) {
    return 1;
  }

  return 0;
}

extern "C" int mgp_shutdown_module() { return 0; }
