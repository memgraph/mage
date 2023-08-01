#include <mgp.hpp>

#include "algorithm/collections.hpp"

extern "C" int mgp_init_module(struct mgp_module *module, struct mgp_memory *memory) {
  try {
    mgp::memory = memory;

    AddProcedure(Collections::RemoveAll, Collections::kProcedureRemoveAll, mgp::ProcedureType::Read,
                 {mgp::Parameter(Collections::kArgumentsInputList, {mgp::Type::List, mgp::Type::Any}),
                  mgp::Parameter(Collections::kArgumentsRemoveList, {mgp::Type::List, mgp::Type::Any})},
                 {mgp::Return(Collections::kReturnRemoveAll, {mgp::Type::List, mgp::Type::Any})}, module, memory);

    AddProcedure(Collections::Sum, Collections::kProcedureSum, mgp::ProcedureType::Read,
                 {mgp::Parameter(Collections::kInputList, {mgp::Type::List, mgp::Type::Any})},
                 {mgp::Return(Collections::kReturnSum, mgp::Type::Double)}, module, memory);

    AddProcedure(Collections::Union, Collections::kProcedureUnion, mgp::ProcedureType::Read,
                 {mgp::Parameter(Collections::kArgumentsInputList1, {mgp::Type::List, mgp::Type::Any}),
                  mgp::Parameter(Collections::kArgumentsInputList2, {mgp::Type::List, mgp::Type::Any})},
                 {mgp::Return(Collections::kReturnUnion, {mgp::Type::List, mgp::Type::Any})}, module, memory);

    AddProcedure(Collections::Sort, Collections::kProcedureSort, mgp::ProcedureType::Read,
                 {mgp::Parameter(Collections::kArgumentsInputList, {mgp::Type::List, mgp::Type::Any})},
                 {mgp::Return(Collections::kReturnSort, {mgp::Type::List, mgp::Type::Any})}, module, memory);

    AddProcedure(Collections::ContainsSorted, Collections::kProcedureCS, mgp::ProcedureType::Read,
                 {mgp::Parameter(Collections::kArgumentInputList, {mgp::Type::List, mgp::Type::Any}),
                  mgp::Parameter(Collections::kArgumentElement, mgp::Type::Any)},
                 {mgp::Return(Collections::kReturnCS, mgp::Type::Bool)}, module, memory);

    AddProcedure(Collections::Max, Collections::kProcedureMax, mgp::ProcedureType::Read,
                 {mgp::Parameter(Collections::kArgumentsInputList, {mgp::Type::List, mgp::Type::Any})},
                 {mgp::Return(Collections::kReturnMax, mgp::Type::Any)}, module, memory);

    AddProcedure(Collections::Split, Collections::kProcedureSplit, mgp::ProcedureType::Read,
                 {mgp::Parameter(Collections::kArgumentInputList, {mgp::Type::List, mgp::Type::Any}),
                  mgp::Parameter(Collections::kArgumentDelimiter, mgp::Type::Any)},
                 {mgp::Return(Collections::kReturnSplit, {mgp::Type::List, mgp::Type::Any})}, module, memory);

    AddProcedure(Collections::Pairs, Collections::kProcedurePairs, mgp::ProcedureType::Read,
                 {mgp::Parameter(Collections::kInputList, {mgp::Type::List, mgp::Type::Any})},
                 {mgp::Return(Collections::kReturnPairs, {mgp::Type::List, mgp::Type::Any})}, module, memory);
    AddProcedure(Collections::Contains, std::string(Collections::kProcedureContains).c_str(), mgp::ProcedureType::Read,
                {mgp::Parameter(std::string(Collections::kArgumentListContains).c_str(), {mgp::Type::List, mgp::Type::Any}),
                mgp::Parameter(std::string(Collections::kArgumentValueContains).c_str(), mgp::Type::Any)},
                {mgp::Return(std::string(Collections::kReturnValueContains).c_str(), mgp::Type::Bool)}, module, memory);
 
    AddProcedure(Collections::Min, std::string(Collections::kProcedureMin).c_str(), mgp::ProcedureType::Read,
                {mgp::Parameter(std::string(Collections::kArgumentListMin).c_str(), {mgp::Type::List, mgp::Type::Any})},
                {mgp::Return(std::string(Collections::kReturnValueMin).c_str(), mgp::Type::Any)}, module, memory);
    AddProcedure(Collections::UnionAll, std::string(Collections::kProcedureUnionAll).c_str(), mgp::ProcedureType::Read,
                {mgp::Parameter(std::string(Collections::kArgumentList1UnionAll).c_str(), {mgp::Type::List, mgp::Type::Any}),
                mgp::Parameter(std::string(Collections::kArgumentList2UnionAll).c_str(), {mgp::Type::List, mgp::Type::Any})},
                {mgp::Return(std::string(Collections::kReturnValueUnionAll).c_str(), {mgp::Type::List, mgp::Type::Any})},
                module, memory);
      
    AddProcedure(Collections::ToSet, Collections::kProcedureToSet, mgp::ProcedureType::Read,
              {mgp::Parameter(Collections::kArgumentListToSet, {mgp::Type::List, mgp::Type::Any})},
              {mgp::Return(Collections::kReturnToSet, {mgp::Type::List, mgp::Type::Any})}, module, memory);

    AddProcedure(Collections::Partition, std::string(Collections::kProcedurePartition).c_str(), mgp::ProcedureType::Read,
              {mgp::Parameter(std::string(Collections::kArgumentListPartition).c_str(), {mgp::Type::List, mgp::Type::Any}),
              mgp::Parameter(std::string(Collections::kArgumentSizePartition).c_str(), mgp::Type::Int)},
              {mgp::Return(std::string(Collections::kReturnValuePartition).c_str(), {mgp::Type::List, mgp::Type::Any})},
              module, memory);

  } catch (const std::exception &e) {
    return 1;
  }

  return 0;
}

extern "C" int mgp_shutdown_module() { return 0; }
