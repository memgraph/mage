#include <mgp.hpp>

#include "algorithm/path.hpp"

extern "C" int mgp_init_module(struct mgp_module *module, struct mgp_memory *memory) {
  try {
    mgp::memory = memory;
    AddProcedure(
        Path::SubgraphNodes, Path::kProcedureSubgraphNodes, mgp::ProcedureType::Read,
        {mgp::Parameter(Path::kArgumentsStart, mgp::Type::Any), mgp::Parameter(Path::kArgumentsConfig, mgp::Type::Map)},
        {mgp::Return(Path::kReturnSubgraphNodes, mgp::Type::Node)}, module, memory);

    AddProcedure(
        Path::SubgraphAll, Path::kProcedureSubgraphAll, mgp::ProcedureType::Read,
        {mgp::Parameter(Path::kArgumentsStart, mgp::Type::Any), mgp::Parameter(Path::kArgumentsConfig, mgp::Type::Map)},
        {mgp::Return(Path::kReturnNodesSubgraphAll, {mgp::Type::List, mgp::Type::Node}),
         mgp::Return(Path::kReturnRelsSubgraphAll, {mgp::Type::List, mgp::Type::Relationship})},
        module, memory);

  } catch (const std::exception &e) {
    return 1;
  }

  return 0;
}

extern "C" int mgp_shutdown_module() { return 0; }
