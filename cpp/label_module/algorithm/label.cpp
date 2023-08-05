#include "label.hpp"

void Label::Exists(mgp_list *args, mgp_graph *memgraph_graph, mgp_result *result, mgp_memory *memory) {
  mgp::memory = memory;
  const auto arguments = mgp::List(args);
  const auto record_factory = mgp::RecordFactory(result);
  try {
    bool exists = false;

    const auto label = arguments[1].ValueString();
    if (arguments[0].IsNode()) {
      const auto node = arguments[0].ValueNode();
      exists = node.HasLabel(label);
    }

    auto record = record_factory.NewRecord();
    record.Insert(std::string(kResultExists).c_str(), std::move(exists));

  } catch (const std::exception &e) {
    record_factory.SetErrorMessage(e.what());
    return;
  }
}
