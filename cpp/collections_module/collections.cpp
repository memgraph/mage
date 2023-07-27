#include <mgp.hpp>
#include "algorithm/algorithm.hpp"

extern "C" int mgp_init_module(struct mgp_module *module, struct mgp_memory *memory) {
  try {
    mgp::memory = memory;

    AddProcedure(
        Collections::Contains, std::string(Collections::kProcedureContains).c_str(), mgp::ProcedureType::Read,
        {mgp::Parameter(std::string(Collections::kArgumentListContains).c_str(), {mgp::Type::List, mgp::Type::Any}),
         mgp::Parameter(std::string(Collections::kArgumentValueContains).c_str(), mgp::Type::Any)},
        {mgp::Return(std::string(Collections::kReturnValueContains).c_str(), mgp::Type::Bool)}, module, memory);
 
    AddProcedure(
        Collections::Min, std::string(Collections::kProcedureMin).c_str(), mgp::ProcedureType::Read,
        {mgp::Parameter(std::string(Collections::kArgumentListMin).c_str(), {mgp::Type::List, mgp::Type::Any})},
        {mgp::Return(std::string(Collections::kReturnValueMin).c_str(), mgp::Type::Any)}, module, memory);
    AddProcedure(
        Collections::UnionAll, std::string(Collections::kProcedureUnionAll).c_str(), mgp::ProcedureType::Read,
        {mgp::Parameter(std::string(Collections::kArgumentList1UnionAll).c_str(), {mgp::Type::List, mgp::Type::Any}),
         mgp::Parameter(std::string(Collections::kArgumentList2UnionAll).c_str(), {mgp::Type::List, mgp::Type::Any})},
        {mgp::Return(std::string(Collections::kReturnValueUnionAll).c_str(), {mgp::Type::List, mgp::Type::Any})},
        module, memory);
      
    AddProcedure(Collections::toSet, Collections::kProcedureToSet, mgp::ProcedureType::Read,
        {mgp::Parameter(Collections::kArgumentListToSet, {mgp::Type::List, mgp::Type::Any})},
        {mgp::Return(Collections::kReturnToSet, {mgp::Type::List, mgp::Type::Any})}, module, memory);

  } catch (const std::exception &e) {
    return 1;
  }

  return 0;
}

extern "C" int mgp_shutdown_module() { return 0; }
