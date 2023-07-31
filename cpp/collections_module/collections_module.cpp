#include <mgp.hpp>

#include "algorithm/collections.hpp"

extern "C" int mgp_init_module(struct mgp_module *module, struct mgp_memory *memory) {
  try {
    mgp::memory = memory;

    AddProcedure(Collections::SumLongs, Collections::kProcedureSumLongs, mgp::ProcedureType::Read,
                 {mgp::Parameter(Collections::kSumLongsArg1, {mgp::Type::List, mgp::Type::Any})},
                 {mgp::Return(Collections::kResultSumLongs, mgp::Type::Int)}, module, memory);

    AddProcedure(Collections::Avg, Collections::kProcedureAvg, mgp::ProcedureType::Read,
                 {mgp::Parameter(Collections::kAvgArg1, {mgp::Type::List, mgp::Type::Any})},
                 {mgp::Return(Collections::kResultAvg, mgp::Type::Double)}, module, memory);

    AddProcedure(Collections::ContainsAll, Collections::kProcedureContainsAll, mgp::ProcedureType::Read,
                 {mgp::Parameter(Collections::kContainsAllArg1, {mgp::Type::List, mgp::Type::Any}),
                  mgp::Parameter(Collections::kContainsAllArg2, {mgp::Type::List, mgp::Type::Any})},
                 {mgp::Return(Collections::kResultContainsAll, mgp::Type::Bool)}, module, memory);

    AddProcedure(Collections::Intersection, Collections::kProcedureIntersection, mgp::ProcedureType::Read,
                 {mgp::Parameter(Collections::kIntersectionArg1, {mgp::Type::List, mgp::Type::Any}),
                  mgp::Parameter(Collections::kIntersectionArg2, {mgp::Type::List, mgp::Type::Any})},
                 {mgp::Return(Collections::kResultIntersection, {mgp::Type::List, mgp::Type::Any})}, module, memory);

  } catch (const std::exception &e) {
    return 1;
  }

  return 0;
}

extern "C" int mgp_shutdown_module() { return 0; }
