#include <mgp.hpp>
#include <string>

constexpr std::string_view kReturnValueContains = "output";
constexpr std::string_view kProcedureContains = "contains";
constexpr std::string_view kArgumentListContains = "list";
constexpr std::string_view kArgumentValueContains = "value";

void contains(mgp_list *args, mgp_graph *memgraph_graph, mgp_result *result, mgp_memory *memory) {
  mgp::memory = memory;
  const auto arguments = mgp::List(args);
  const auto record_factory = mgp::RecordFactory(result);
  try {
    const mgp::List &list = arguments[0].ValueList();
    const mgp::Value &value = arguments[1];

    bool contains_value{false};
    auto record = record_factory.NewRecord();

    if (list.Empty()) {
      record.Insert(std::string(kReturnValueContains).c_str(), contains_value);
      return;
    }
    for (size_t i = 0; i < list.Size(); i++) {
      if (list[i] == value) {
        contains_value = true;
        break;
      }
    }
    record.Insert(std::string(kReturnValueContains).c_str(), contains_value);

  } catch (const std::exception &e) {
    record_factory.SetErrorMessage(e.what());
    return;
  }
}
