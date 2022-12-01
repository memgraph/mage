#include <mg_utils.hpp>

const char *kProcedureGet = "get";
const char *kParameterStart = "start";
const char *kParameterSteps = "steps";
const char *kReturnStep = "step";
const char *kReturnNode = "node";

void RandomWalk(mgp_list *args, mgp_graph *memgraph_graph, mgp_result *result, mgp_memory *memory) {
  mgp::memory = memory;

  const auto arguments = mgp::List(args);
  const auto record_factory = mgp::RecordFactory(result);

  const auto start = arguments[0].ValueNode();
  const auto n_steps = arguments[1].ValueInt();

  srand(time(NULL));

  auto current_nodes = mgp::List();
  current_nodes.AppendExtend(mgp::Value(start));

  std::int64_t step = 0;
  while (step <= n_steps) {
    auto current_node = current_nodes[current_nodes.Size() - 1].ValueNode();

    auto neighbours = mgp::List();
    for (const auto relationship : current_node.OutRelationships()) {
      neighbours.AppendExtend(mgp::Value(relationship));
    }

    if (neighbours.Size() == 0) {
      break;
    }

    const auto next_node = neighbours[rand() % neighbours.Size()].ValueRelationship().To();

    current_nodes.AppendExtend(mgp::Value(next_node));
    step++;
  }

  for (std::int64_t i = 0; i < current_nodes.Size(); i++) {
    auto record = record_factory.NewRecord();
    record.Insert(kReturnStep, i);
    record.Insert(kReturnNode, current_nodes[i].ValueNode());
  }
}

extern "C" int mgp_init_module(struct mgp_module *module, struct mgp_memory *memory) {
  mgp::memory = memory;

  std::int64_t default_steps = 10;
  try {
    mgp::AddProcedure(RandomWalk, kProcedureGet, mgp::ProdecureType::Read,
                      {mgp::Parameter(kParameterStart, mgp::Type::Node),
                       mgp::Parameter(kParameterSteps, mgp::Type::Int, default_steps)},
                      {mgp::Return(kReturnStep, mgp::Type::Int), mgp::Return(kReturnNode, mgp::Type::Node)}, module,
                      memory);
  } catch (const std::exception &e) {
    return 1;
  }
  return 0;
}

extern "C" int mgp_shutdown_module() { return 0; }
