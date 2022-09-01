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

std::string_view type_to_str(mage::Type type) {
  switch (type) {
    case mage::Type::Null:
      return "Null";
    case mage::Type::Bool:
      return "Bool";
    case mage::Type::Int:
      return "Int";
    case mage::Type::Double:
      return "Double";
    case mage::Type::String:
      return "String";
    case mage::Type::List:
      return "List";
    case mage::Type::Map:
      return "Map";
    case mage::Type::Node:
      return "Node";
    case mage::Type::Relationship:
      return "Relationship";
    case mage::Type::Path:
      return "Path";
    case mage::Type::Date:
      return "Date";
    case mage::Type::LocalTime:
      return "LocalTime";
    case mage::Type::LocalDateTime:
      return "LocalDateTime";
    case mage::Type::Duration:
      return "Duration";
    default:
      return "Type unknown";
  }
}
void TestProc2(mgp_list *args, mgp_graph *memgraph_graph, mgp_result *result, mgp_memory *memory) {
  mage::memory = memory;

  try {
    if (true) {
      // Test Id:
      if (true) {
        std::cout << "Testing ID\n";

        int64_t int_1 = 8;
        uint64_t int_2 = 8;

        auto id_1 = mage::Id::FromInt(int_1);
        auto id_2 = mage::Id::FromUint(int_2);

        auto id_0 = mage::Id::FromInt(mgp::value_get_int(mgp::list_at(args, 1)));

        std::cout << (id_1.AsInt() == 8) << "\n";
        std::cout << (id_1.AsUint() == 8) << "\n";

        std::cout << (id_2.AsInt() == 8) << "\n";
        std::cout << (id_2.AsUint() == 8) << "\n";

        std::cout << (id_1 == id_2) << "\n";
        std::cout << (!(id_1 != id_2)) << "\n";

        int64_t int_3 = 7;
        uint64_t int_4 = 7;

        auto id_3 = mage::Id::FromInt(int_3);
        auto id_4 = mage::Id::FromUint(int_4);

        std::cout << (!(id_1 == id_3)) << "\n";
        std::cout << (!(id_1 == id_4)) << "\n";
        std::cout << (!(id_2 == id_3)) << "\n";
        std::cout << (!(id_2 == id_4)) << "\n";
        std::cout << (id_1 != id_3) << "\n";
        std::cout << (id_1 != id_4) << "\n";
        std::cout << (id_2 != id_3) << "\n";
        std::cout << (id_2 != id_4) << "\n";
      }

      // Test Value:
      if (true) {
        std::cout << "Testing Value\n";

        for (size_t i = 0; i < mgp::list_size(args); i++) {
          std::cout << type_to_str(mage::Value(mgp::list_at(args, i)).type()) << "\n";
        }

        std::vector<mage::Value> values;
        for (size_t i = 0; i < mgp::list_size(args); i++) {
          if (i != 3 && i != 4) {
            values.push_back(mage::Value(mgp::list_at(args, i)));
            std::cout << "Added " << type_to_str(mage::Value(mgp::list_at(args, i)).type()) << "\n";
          }
        }
      }

      // Test List:
      if (false) {
        std::cout << "Testing List\n";

        auto list_1 = mage::List(mgp::value_get_list(mgp::list_at(args, 4)));

        bool first = false;
        std::cout << "[";
        for (const auto element : list_1) {
          if (!first) std::cout << ", ";
          std::cout << element.ValueString();
          first = false;
        }
        std::cout << "]\n";
      }

      // Test Map:
      if (true) {
        std::cout << "Testing Map\n";
        auto map_1 = mage::Map(mgp::value_get_map(mgp::list_at(args, 5)));
        std::map<std::string_view, mage::Value> m;
        for (const auto e : map_1) {
          std::cout << e.key << " : " << e.value.ValueString() << "\n";
          m.insert(std::pair<std::string_view, mage::Value>(e.key, e.value));
        }

        for (const auto [k, v] : m) {
          std::cout << k << " : " << v.ValueString() << "\n";
        }
      }

      // Test Node:
      if (false) {
        std::cout << "Testing Node\n";
        auto node_1 = mage::Node(mgp::value_get_vertex(mgp::list_at(args, 6)));

        std::vector<mage::Node> x;
        x.push_back(node_1);
        std::set<mage::Node> y;
        y.insert(node_1);
        std::unordered_set<mage::Node> z;
        z.insert(node_1);
      }

      // Test Relationship:
      if (false) {
        auto edge_1 = mage::Relationship(mgp::value_get_edge(mgp::list_at(args, 7)));

        std::vector<mage::Relationship> x;
        x.push_back(edge_1);
        std::set<mage::Relationship> y;
        y.insert(edge_1);
        std::unordered_set<mage::Relationship> z;
        z.insert(edge_1);
      }

      // Test Path:
      if (false) {
        auto path_1 = mage::Path(mgp::value_get_path(mgp::list_at(args, 8)));

        std::vector<mage::Path> x;
        x.push_back(path_1);
      }

      // Test Date:
      if (false) {
        auto date_1 = mage::Date(mgp::value_get_date(mgp::list_at(args, 9)));

        std::vector<mage::Date> x;
        x.push_back(date_1);
        std::set<mage::Date> y;
        y.insert(date_1);
        std::unordered_set<mage::Date> z;
        z.insert(date_1);
      }

      // Test LocalTime:
      if (false) {
        auto local_time_1 = mage::LocalTime(mgp::value_get_local_time(mgp::list_at(args, 10)));

        std::vector<mage::LocalTime> x;
        x.push_back(local_time_1);
        std::set<mage::LocalTime> y;
        y.insert(local_time_1);
        std::unordered_set<mage::LocalTime> z;
        z.insert(local_time_1);
      }

      // Test LocalDateTime:
      if (false) {
        auto local_date_time_1 = mage::LocalDateTime(mgp::value_get_local_date_time(mgp::list_at(args, 11)));

        std::vector<mage::LocalDateTime> x;
        x.push_back(local_date_time_1);
        std::set<mage::LocalDateTime> y;
        y.insert(local_date_time_1);
        std::unordered_set<mage::LocalDateTime> z;
        z.insert(local_date_time_1);
      }

      // Test Duration:
      if (false) {
        auto duration_1 = mage::Duration(mgp::value_get_duration(mgp::list_at(args, 12)));

        std::vector<mage::Duration> x;
        x.push_back(duration_1);
        std::set<mage::Duration> y;
        y.insert(duration_1);
        std::unordered_set<mage::Duration> z;
        z.insert(duration_1);
      }
    }
  } catch (const std::exception &e) {
    // We must not let any exceptions out of our module.
    mgp::result_set_error_msg(result, e.what());
    return;
  }
}

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
    auto *test_proc = mgp::module_add_read_procedure(module, kProcedureRun, TestProc2);

    mgp::proc_add_arg(test_proc, kArgumentBool, mgp::type_bool());
    mgp::proc_add_arg(test_proc, kArgumentInt, mgp::type_int());
    mgp::proc_add_arg(test_proc, kArgumentDouble, mgp::type_float());
    mgp::proc_add_arg(test_proc, kArgumentString, mgp::type_string());
    mgp::proc_add_arg(test_proc, kArgumentList, mgp::type_list(mgp::type_string()));
    mgp::proc_add_arg(test_proc, kArgumentMap, mgp::type_map());
    mgp::proc_add_arg(test_proc, kArgumentNode, mgp::type_node());
    mgp::proc_add_arg(test_proc, kArgumentRelationship, mgp::type_relationship());
    mgp::proc_add_arg(test_proc, kArgumentPath, mgp::type_path());
    mgp::proc_add_arg(test_proc, kArgumentDate, mgp::type_date());
    mgp::proc_add_arg(test_proc, kArgumentLocalTime, mgp::type_local_time());
    mgp::proc_add_arg(test_proc, kArgumentLocalDateTime, mgp::type_local_date_time());
    mgp::proc_add_arg(test_proc, kArgumentDuration, mgp::type_duration());

    mgp::proc_add_result(test_proc, kFieldNode, mgp::type_node());
    mgp::proc_add_result(test_proc, kFieldComponentId, mgp::type_int());
  } catch (const std::exception &e) {
    return 1;
  }

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