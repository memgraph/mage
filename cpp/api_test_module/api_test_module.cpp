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

void PrintList(mage::List &list) {
  auto first = true;
  std::cout << "[";
  for (const auto &element : list) {
    if (!first) std::cout << ", ";
    std::cout << element.ValueString();
    first = false;
  }
  std::cout << "]\n";
}

void PrintMap(mage::Map &map) {
  auto first = true;
  std::cout << "{";
  for (const auto &item : map) {
    if (!first) std::cout << ", ";
    std::cout << item.key << ": " << item.value.ValueString();
    first = false;
  }
  std::cout << "}\n";
}

void TestProc(std::vector<mage::Value> arguments, mage::Graph graph, mage::RecordFactory record_factory) {
  auto test_graph = true;
  auto test_id = true;
  auto test_list = true;
  auto test_map = true;
  auto test_node = true;
  auto test_relationship = true;
  auto test_path = true;
  auto test_date = true;
  auto test_local_time = true;
  auto test_local_date_time = true;
  auto test_duration = true;

  if (test_graph) {
    std::cout << graph.Order() << "\n";
    std::cout << graph.Size() << "\n";
    for (const auto &node : graph.Nodes()) {
      std::cout << node.Id().AsInt() << "\t";
    }
    std::cout << "\n";
    for (const auto &rel : graph.Relationships()) {
      std::cout << rel.Id().AsInt() << "\t";
    }
    std::cout << "\n";

    std::cout << (graph.ContainsNode(mage::Id::FromInt(0))) << "\n";
    std::cout << (graph.ContainsNode(mage::Id::FromInt(2))) << "\n";
    std::cout << (graph.ContainsNode(arguments[0].ValueNode())) << "\n";
    std::cout << (graph.ContainsRelationship(arguments[1].ValueRelationship())) << "\n";
  }

  if (test_id) {
    std::cout << "Testing ID\n";

    auto id_0 = mage::Id::FromInt(arguments[4].ValueInt());
    std::cout << (id_0 == mage::Id::FromInt(2)) << "\n";

    int64_t int_1 = 8;
    uint64_t int_2 = 8;
    int64_t int_3 = 7;
    uint64_t int_4 = 7;

    auto id_1 = mage::Id::FromInt(int_1);
    auto id_2 = mage::Id::FromUint(int_2);
    auto id_3 = mage::Id::FromInt(int_3);
    auto id_4 = mage::Id::FromUint(int_4);

    std::cout << (id_1.AsInt() == 8) << "\n";
    std::cout << (id_1.AsUint() == 8) << "\n";
    std::cout << (id_2.AsInt() == 8) << "\n";
    std::cout << (id_2.AsUint() == 8) << "\n";

    std::cout << (id_1 == id_2) << "\n";
    std::cout << (!(id_1 != id_2)) << "\n";

    std::cout << (!(id_1 == id_3)) << "\n";
    std::cout << (!(id_1 == id_4)) << "\n";
    std::cout << (!(id_2 == id_3)) << "\n";
    std::cout << (!(id_2 == id_4)) << "\n";
    std::cout << (id_1 != id_3) << "\n";
    std::cout << (id_1 != id_4) << "\n";
    std::cout << (id_2 != id_3) << "\n";
    std::cout << (id_2 != id_4) << "\n";
  }

  if (test_list) {
    std::cout << "Testing List\n";

    auto list_1 = mage::List(arguments[7].ValueList());
    PrintList(list_1);

    auto list_2 = mage::List();
    PrintList(list_2);

    std::cout << (list_1 == list_2) << "\n";

    auto list_3 = mage::List(10);
    PrintList(list_3);
    std::cout << list_3.Size() << "\n";

    auto a = mage::Value("a");
    list_3.Append(a);
    list_3.AppendExtend(a);
    PrintList(list_3);

    std::cout << !(list_1 == list_3) << "\n";

    std::vector<mage::Value> values{mage::Value("a"), mage::Value("b"), mage::Value("c")};
    auto list_4 = mage::List(values);
    PrintList(list_4);

    auto list_5 = mage::List({mage::Value("d"), mage::Value("e"), mage::Value("f")});
    PrintList(list_5);
  }

  if (test_map) {
    std::cout << "Testing Map\n";
    auto map_1 = mage::Map(arguments[8].ValueMap());

    std::map<std::string_view, mage::Value> map_1a;
    PrintMap(map_1);
    for (const auto &e : map_1) {
      map_1a.insert(std::pair<std::string_view, mage::Value>(e.key, e.value));
    }

    for (const auto &[k, v] : map_1a) {
      std::cout << k << " : " << v.ValueString() << "\n";
    }

    auto map_2 = mage::Map();
    PrintMap(map_2);

    std::cout << (map_1 == map_2) << "\n";

    auto y = mage::Value("y");
    map_2.Insert("x", y);
    PrintMap(map_2);

    std::cout << !(map_1 == map_2) << "\n";

    auto v_1 = mage::Value("1");
    auto v_2 = mage::Value("2");
    auto p_1 = std::pair{"a", v_1};
    auto p_2 = std::pair{"b", v_2};
    auto map_3 = mage::Map({p_1, p_2});
    PrintMap(map_3);
  }

  if (test_node) {
    std::cout << "Testing Node\n";
    auto node_1 = mage::Node(arguments[0].ValueNode());

    std::vector<mage::Node> x;
    x.push_back(node_1);
    std::set<mage::Node> y;
    y.insert(node_1);
    std::unordered_set<mage::Node> z;
    z.insert(node_1);

    std::cout << node_1.Id().AsInt() << "\n";
    std::cout << node_1.HasLabel("Node") << "\n";
    std::cout << node_1.HasLabel("Vertex") << "\n";

    auto first = true;
    std::cout << "Labels: [";
    for (const auto &label : node_1.Labels()) {
      if (!first) std::cout << ", ";
      std::cout << label;
      first = false;
    }
    std::cout << "]\n";

    first = true;
    std::cout << "Properties: {";
    for (const auto &[name, value] : node_1.Properties()) {
      if (!first) std::cout << ", ";
      std::cout << name << ": " << value.ValueInt();
      first = false;
    }
    std::cout << "}\n";

    first = true;
    std::cout << "Out-neighbors: [";
    for (const auto &neighbor : node_1.OutRelationships()) {
      if (!first) std::cout << ", ";
      std::cout << neighbor.Id().AsInt();
      first = false;
    }
    std::cout << "]\n";

    // first = true;
    // auto node_2 = graph.GetNodeById(mage::Id::FromInt(1));
    // std::cout << "In-neighbors: [";
    // for (const auto &neighbor : node_2.in_Relationships()) {
    //   if (!first) std::cout << ", ";
    //   std::cout << neighbor.Id().AsInt();
    //   first = false;
    // }
    // std::cout << "]\n";
  }

  if (test_relationship) {
    std::cout << "Testing Relationship\n";
    auto edge_1 = mage::Relationship(arguments[1].ValueRelationship());

    std::vector<mage::Relationship> x;
    x.push_back(edge_1);
    std::set<mage::Relationship> y;
    y.insert(edge_1);
    std::unordered_set<mage::Relationship> z;
    z.insert(edge_1);

    std::cout << edge_1.Id().AsInt() << "\n";
    std::cout << edge_1.Type() << "\n";

    for (const auto &[name, value] : edge_1.Properties()) {
      std::cout << name << ": " << value.ValueInt() << ", ";
    }
    std::cout << "\n";

    std::cout << edge_1.From().Id().AsInt() << "\n";
    std::cout << edge_1.To().Id().AsInt() << "\n";
  }

  if (test_path) {
    std::cout << "Testing Path\n";
    auto path_1 = mage::Path(arguments[2].ValuePath());

    std::vector<mage::Path> x;
    x.push_back(path_1);

    // auto node_0 = graph.GetNodeById(mage::Id::FromInt(0));
    // auto path_2 = mage::Path(node_0);

    std::cout << path_1.Length() << "\n";
    std::cout << path_1.GetNodeAt(0).Id().AsInt() << "\n";
    std::cout << path_1.GetRelationshipAt(0).Id().AsInt() << "\n";

    // std::cout << !(path_1 == path_2) << "\n";

    // path_2.Expand(*(node_0.out_Relationships().begin()));
    // std::cout << (path_1 == path_2) << "\n";
  }

  if (test_date) {
    std::cout << "Testing Date\n";
    auto date_1 = mage::Date(arguments[9].ValueDate());

    std::vector<mage::Date> x;
    x.push_back(date_1);
    std::set<mage::Date> y;
    y.insert(date_1);
    std::unordered_set<mage::Date> z;
    z.insert(date_1);

    auto date_2 = mage::Date("2022-04-09");
    auto date_3 = mage::Date(2022, 4, 9);

    auto date_4 = mage::Date::Now();

    std::cout << date_1.Year() << "\n";
    std::cout << date_1.Month() << "\n";
    std::cout << date_1.Day() << "\n";
    std::cout << date_2.Timestamp() << "\n";

    std::cout << !(date_1 == date_2) << "\n";
    std::cout << (date_2 == date_3) << "\n";

    auto duration_1 = mage::Duration(arguments[12].ValueDuration());
    auto date_5 = date_1 + duration_1;
    auto date_6 = date_1 - duration_1;
    auto date_7 = date_1 - date_2;
  }

  if (test_local_time) {
    std::cout << "Testing LocalTime\n";
    auto lt_1 = mage::LocalTime(arguments[10].ValueLocalTime());

    std::vector<mage::LocalTime> x;
    x.push_back(lt_1);
    std::set<mage::LocalTime> y;
    y.insert(lt_1);
    std::unordered_set<mage::LocalTime> z;
    z.insert(lt_1);

    auto lt_2 = mage::LocalTime("09:15:00");
    auto lt_3 = mage::LocalTime(9, 15, 0, 0, 0);

    auto lt_4 = mage::LocalTime::Now();

    std::cout << lt_1.Hour() << "\n";
    std::cout << lt_1.Minute() << "\n";
    std::cout << lt_1.Second() << "\n";
    std::cout << lt_1.Millisecond() << "\n";
    std::cout << lt_1.Microsecond() << "\n";
    std::cout << lt_2.Timestamp() << "\n";

    std::cout << !(lt_1 == lt_2) << "\n";
    std::cout << (lt_2 == lt_3) << "\n";

    auto duration_1 = mage::Duration(arguments[12].ValueDuration());
    auto lt_5 = lt_1 + duration_1;
    auto lt_6 = lt_1 - duration_1;
    auto lt_7 = lt_1 - lt_2;
  }

  if (test_local_date_time) {
    std::cout << "Testing LocalDateTime\n";
    auto ldt_1 = mage::LocalDateTime(arguments[11].ValueLocalDateTime());

    std::vector<mage::LocalDateTime> x;
    x.push_back(ldt_1);
    std::set<mage::LocalDateTime> y;
    y.insert(ldt_1);
    std::unordered_set<mage::LocalDateTime> z;
    z.insert(ldt_1);
    auto ldt_2 = mage::LocalDateTime("2021-10-05T14:15:00");
    auto ldt_3 = mage::LocalDateTime(2021, 10, 5, 14, 15, 0, 0, 0);

    auto ldt_4 = mage::LocalTime::Now();

    std::cout << ldt_1.Year() << "\n";
    std::cout << ldt_1.Month() << "\n";
    std::cout << ldt_1.Day() << "\n";
    std::cout << ldt_1.Hour() << "\n";
    std::cout << ldt_1.Minute() << "\n";
    std::cout << ldt_1.Second() << "\n";
    std::cout << ldt_1.Millisecond() << "\n";
    std::cout << ldt_1.Microsecond() << "\n";
    std::cout << ldt_2.Timestamp() << "\n";

    std::cout << !(ldt_1 == ldt_2) << "\n";
    std::cout << (ldt_2 == ldt_3) << "\n";

    auto duration_1 = mage::Duration(arguments[12].ValueDuration());
    auto ldt_5 = ldt_1 + duration_1;
    auto ldt_6 = ldt_1 - duration_1;
    auto ldt_7 = ldt_1 - ldt_2;
  }

  if (test_duration) {
    std::cout << "Testing Duration\n";
    auto duration_1 = mage::Duration(arguments[12].ValueDuration());

    std::vector<mage::Duration> x;
    x.push_back(duration_1);
    std::set<mage::Duration> y;
    y.insert(duration_1);
    std::unordered_set<mage::Duration> z;
    z.insert(duration_1);

    auto duration_2 = mage::Duration("PT2M2.33S");
    auto duration_3 = mage::Duration(1465355);
    auto duration_4 = mage::Duration(5, 14, 15, 0, 0, 0);

    std::cout << duration_2.Microseconds() << "\n";

    std::cout << !(duration_1 == duration_3) << "\n";
    std::cout << (duration_1 == duration_2) << "\n";
  }

  auto record = record_factory.NewRecord();
  record.Insert("out_0", arguments[0].ValueNode());
  record.Insert("out_1", arguments[1].ValueRelationship());
  record.Insert("out_2", arguments[2].ValuePath());
  record.Insert("out_3", true);
  record.Insert("out_4", (int64_t)1);
  record.Insert("out_5", 1.0);
  record.Insert("out_6", arguments[6].ValueString());
  record.Insert("out_7", arguments[7].ValueList());
  record.Insert("out_8", arguments[8].ValueMap());
  record.Insert("out_9", arguments[9].ValueDate());
  record.Insert("out_10", arguments[10].ValueLocalTime());
  record.Insert("out_11", arguments[11].ValueLocalDateTime());
  record.Insert("out_12", arguments[12].ValueDuration());
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

void WriteProc(mgp_list *args, mgp_graph *memgraph_graph, mgp_result *result, mgp_memory *memory) {
  mage::memory = memory;
  auto graph = mage::Graph(memgraph_graph);

  for (int i = 0; i < 10; i++) {
    graph.CreateNode();
  }

  int i = 5;
  for (const auto &node : graph.Nodes()) {
    graph.DeleteNode(node);
    i -= 1;
    if (i == 0) break;
  }

  for (const auto &node_1 : graph.Nodes()) {
    for (const auto &node_2 : graph.Nodes()) {
      graph.CreateRelationship(node_1, node_2, "R");
    }
  }

  for (const auto &node_1 : graph.Nodes()) {
    graph.DetachDeleteNode(node_1);
    break;
  }

  for (const auto &rel_1 : graph.Relationships()) {
    graph.DeleteRelationship(rel_1);
    break;
  }
}

void SimpleFunc(mgp_list *args, mgp_func_context *ctx, mgp_func_result *res, mgp_memory *memory) {
  mage::memory = memory;

  std::vector<mage::Value> arguments;
  for (size_t i = 0; i < mgp::list_size(args); i++) {
    auto arg = mage::Value(mgp::list_at(args, i));
    arguments.push_back(arg);
  }

  auto result = mage::Result(res);

  auto first = arguments[0].ValueInt();
  auto second = arguments[1].ValueInt();

  result.SetValue(first * second);
}

extern "C" int mgp_init_module(struct mgp_module *module, struct mgp_memory *memory) {
  try {
    mage::memory = memory;

    const auto default_list = mage::List((size_t)0);
    const auto default_map = mage::Map();
    const auto default_date = mage::Date("2022-09-04");
    auto default_local_time = mage::LocalTime("09:15:00");
    auto default_local_date_time = mage::LocalDateTime("2021-10-05T14:15:00");
    auto default_duration = mage::Duration("PT2M2.33S");

    AddProcedure(
        TestProcWrapper, "run_opt", mage::ProdecureType::Read,
        {mage::Parameter("node", mage::Type::Node), mage::Parameter("relationship", mage::Type::Relationship),
         mage::Parameter("path", mage::Type::Path), mage::Parameter("bool", mage::Type::Bool, false),
         mage::Parameter("int", mage::Type::Int, (int64_t)2), mage::Parameter("double", mage::Type::Double, 2.3),
         mage::Parameter("string", mage::Type::String, "abc"),
         mage::Parameter("list", {mage::Type::List, mage::Type::String}, mage::Value(default_list)),
         mage::Parameter("map", mage::Type::Map, mage::Value(default_map)),
         mage::Parameter("date", mage::Type::Date, mage::Value(default_date)),
         mage::Parameter("local_time", mage::Type::LocalTime, mage::Value(default_local_time)),
         mage::Parameter("local_date_time", mage::Type::LocalDateTime, mage::Value(default_local_date_time)),
         mage::Parameter("duration", mage::Type::Duration, mage::Value(default_duration))},
        {mage::Return("out_0", mage::Type::Node), mage::Return("out_1", mage::Type::Relationship),
         mage::Return("out_2", mage::Type::Path), mage::Return("out_3", mage::Type::Bool),
         mage::Return("out_4", mage::Type::Int), mage::Return("out_5", mage::Type::Double),
         mage::Return("out_6", mage::Type::String), mage::Return("out_7", {mage::Type::List, mage::Type::String}),
         mage::Return("out_8", mage::Type::Map), mage::Return("out_9", mage::Type::Date),
         mage::Return("out_10", mage::Type::LocalTime), mage::Return("out_11", mage::Type::LocalDateTime),
         mage::Return("out_12", mage::Type::Duration)},
        module, memory);
  } catch (const std::exception &e) {
    return 1;
  }

  try {
    mage::memory = memory;

    AddProcedure(
        TestProcWrapper, "run", mage::ProdecureType::Read,
        {mage::Parameter("node", mage::Type::Node), mage::Parameter("relationship", mage::Type::Relationship),
         mage::Parameter("path", mage::Type::Path), mage::Parameter("bool", mage::Type::Bool),
         mage::Parameter("int", mage::Type::Int), mage::Parameter("double", mage::Type::Double),
         mage::Parameter("string", mage::Type::String), mage::Parameter("list", {mage::Type::List, mage::Type::String}),
         mage::Parameter("map", mage::Type::Map), mage::Parameter("date", mage::Type::Date),
         mage::Parameter("local_time", mage::Type::LocalTime),
         mage::Parameter("local_date_time", mage::Type::LocalDateTime),
         mage::Parameter("duration", mage::Type::Duration)},
        {mage::Return("out_0", mage::Type::Node), mage::Return("out_1", mage::Type::Relationship),
         mage::Return("out_2", mage::Type::Path), mage::Return("out_3", mage::Type::Bool),
         mage::Return("out_4", mage::Type::Int), mage::Return("out_5", mage::Type::Double),
         mage::Return("out_6", mage::Type::String), mage::Return("out_7", {mage::Type::List, mage::Type::String}),
         mage::Return("out_8", mage::Type::Map), mage::Return("out_9", mage::Type::Date),
         mage::Return("out_10", mage::Type::LocalTime), mage::Return("out_11", mage::Type::LocalDateTime),
         mage::Return("out_12", mage::Type::Duration)},
        module, memory);

  } catch (const std::exception &e) {
    return 1;
  }

  try {
    mage::memory = memory;

    mage::AddProcedure(WriteProc, "write_proc", mage::ProdecureType::Write, {}, {}, module, memory);

  } catch (const std::exception &e) {
    return 1;
  }

  try {
    mage::memory = memory;

    mage::AddFunction(SimpleFunc, "multiply",
                      {mage::Parameter("int", mage::Type::Int), mage::Parameter("int", mage::Type::Int, (int64_t)3)},
                      module, memory);

  } catch (const std::exception &e) {
    return 1;
  }

  return 0;
}

extern "C" int mgp_shutdown_module() { return 0; }