#include <queue>
#include <unordered_map>

#include <mage.hpp>
#include <mg_exceptions.hpp>
#include <mg_utils.hpp>

namespace {

constexpr char const *kArgumentBool = "bool";
constexpr char const *kArgumentInt = "int";
constexpr char const *kArgumentDouble = "double";
constexpr char const *kArgumentString = "string";
constexpr char const *kArgumentList = "list";
constexpr char const *kArgumentMap = "map";
constexpr char const *kArgumentNode = "vertex";
constexpr char const *kArgumentRelationship = "edge";
constexpr char const *kArgumentPath = "path";
constexpr char const *kArgumentDate = "date";
constexpr char const *kArgumentLocalTime = "local_time";
constexpr char const *kArgumentLocalDateTime = "local_date_time";
constexpr char const *kArgumentDuration = "duration";

constexpr char const *kProcedureRun = "run";

constexpr char const *kFieldNode = "node";
constexpr char const *kFieldComponentId = "component_id";

constexpr char const *kProcedurePathCheck = "path_check";
constexpr char const *kFieldOut = "out";
}  // namespace

void TestProc(std::vector<mage::Value> arguments, mage::Graph graph, mage::RecordFactory record_factory) {
  auto path = arguments[0].ValuePath();

  auto first_id = path.GetNodeAt(0).id().AsInt();

  auto record = record_factory.NewRecord();
  record.Insert("out", (int64_t)first_id);
}

void TestProcWrapper(mgp_list *args, mgp_graph *memgraph_graph, mgp_result *result, mgp_memory *memory) {
  try {
    mage::memory = memory;

    std::vector<mage::Value> arguments;
    for (size_t i = 0; i < mgp::list_size(args); i++) {
      auto arg = mage::Value(mgp::list_at(args, i));
      arguments.push_back(arg);
    }

    TestProc(arguments, mage::Graph(memgraph_graph), mage::RecordFactory(result));
  } catch (const std::exception &e) {
    mgp::result_set_error_msg(result, e.what());
    return;
  }
}

extern "C" int mgp_init_module(struct mgp_module *module, struct mgp_memory *memory) {
  try {
    mage::memory = memory;
    auto wrapper = mage::ProcedureWrapper();

    auto x = mage::Value(0);
    wrapper.AddQueryProcedure(TestProcWrapper, "path_check", {mage::Parameter("path", mage::Type::Path)},
                              {mage::Return("out", mage::Type::Int)}, module, memory);

  } catch (const std::exception &e) {
    return 1;
  }

  return 0;
}

extern "C" int mgp_shutdown_module() { return 0; }