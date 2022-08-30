#include <queue>
#include <unordered_map>

#include <mage.hpp>
#include <mg_exceptions.hpp>
#include <mg_utils.hpp>

namespace {

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

void TestProc(mgp_list *args, mgp_graph *memgraph_graph, mgp_result *result, mgp_memory *memory) {
  mage::memory = memory;

  try {
    if (true) {
      // Test Id:
      {
        int64_t int_1 = 8;
        uint64_t int_2 = 8;

        auto id_1 = mage::Id::FromInt(int_1);
        auto id_2 = mage::Id::FromUint(int_2);

        auto id_0 = mage::Id::FromInt(mgp::value_get_int(mgp::list_at(args, 0)));

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

      // Test List:
      if (false) {
        //   auto list_1 = mage::List(mgp::value_get_list(mgp::list_at(args, 3)));
        //   for (const auto e : list_1) {
        //     std::cout << e.ValueString() << " ";
        //   }
        //   std::cout << "\n";

        auto val_0 = mgp::value_make_int(2, memory);
        auto val_1 = mgp::value_make_int(3, memory);

        // std::vector<mgp_value *> values = {val_0, val_1};
        // std::vector<mgp_value *> values_2 = {mgp::value_copy(val_0, memory), mgp::value_copy(val_1, memory)};

        // std::cout << "examine [";
        // for (auto value : values) {
        //   std::cout << mgp::value_get_int(value) << " ";
        // }
        // std::cout << "]\n";

        // std::cout << "examine [";
        // for (auto value : values_2) {
        //   std::cout << mgp::value_get_int(value) << " ";
        // }
        // std::cout << "]\n";

        // mage::List values_list = mage::List({mage::Value(val_0), mage::Value(val_1)});
        // std::cout << "examine [";
        // for (auto value : values_list) {
        //   std::cout << value.ValueInt() << " ";
        // }
        // std::cout << "]\n";

        auto mgval_0 = mage::Value(val_0);
        auto mgval_1 = mage::Value(val_1);
        std::vector<mage::Value> values_vec;
        std::cout << "add\n";
        values_vec.push_back(mgval_0);
        std::cout << "add\n";
        values_vec.push_back(mgval_1);
        std::cout << "add\n";
        std::cout << "examine [";
        // for (auto value : values_vec) {
        //   std::cout << value.ValueInt() << " ";
        // }
        std::cout << "]\n";

        // mage::List values_list_2 = mage::List(values_vec);
        // std::cout << "examine [";
        // for (auto value : values_list_2) {
        //   std::cout << value.ValueInt() << " ";
        // }
        // std::cout << "]\n";

        // std::cout << "still in\n";
      }

      // Test Map:
      if (false) {
        auto map_1 = mage::Map(mgp::value_get_map(mgp::list_at(args, 4)));
        for (const auto e : map_1) {
          std::cout << e.key << " : " << e.value.ValueString() << "\n";
        }
      }

      // Test Node:
      if (true) {
        auto vertex_1 = mage::Node(mgp::value_get_vertex(mgp::list_at(args, 5)));

        // std::vector<mage::Node> x;
        // x.push_back(vertex_1);
        // std::set<mage::Node> y;
        // y.insert(vertex_1);
        // std::unordered_set<mage::Node> z;
        // z.insert(vertex_1);

        // auto list_1 = mage::List(1);
        // list_1.Append(vertex_1);

        // auto vertex_1 = mgp::vertex_copy(mgp::value_get_vertex(mgp::list_at(args, 5)), memory);
        // std::vector<mgp_vertex *> x;
        // x.push_back(vertex_1);
        std::cout << "be\n";
      }

      std::cout << "beh\n";
      // Test Relationship:
      if (false) {
        auto edge_1 = mage::Relationship(mgp::value_get_edge(mgp::list_at(args, 6)));

        std::vector<mage::Relationship> x;
        x.push_back(edge_1);
        std::set<mage::Relationship> y;
        y.insert(edge_1);
        std::unordered_set<mage::Relationship> z;
        z.insert(edge_1);
      }

      // Test Path:
      if (false) {
        auto path_1 = mage::Path(mgp::value_get_path(mgp::list_at(args, 7)));

        std::vector<mage::Path> x;
        x.push_back(path_1);
      }

      // Test Date:
      if (false) {
        auto date_1 = mage::Date(mgp::value_get_date(mgp::list_at(args, 8)));

        std::vector<mage::Date> x;
        x.push_back(date_1);
        std::set<mage::Date> y;
        y.insert(date_1);
        std::unordered_set<mage::Date> z;
        z.insert(date_1);
      }

      // Test LocalTime:
      if (false) {
        auto local_time_1 = mage::LocalTime(mgp::value_get_local_time(mgp::list_at(args, 9)));

        std::vector<mage::LocalTime> x;
        x.push_back(local_time_1);
        std::set<mage::LocalTime> y;
        y.insert(local_time_1);
        std::unordered_set<mage::LocalTime> z;
        z.insert(local_time_1);
      }

      // Test LocalDateTime:
      if (false) {
        auto local_date_time_1 = mage::LocalDateTime(mgp::value_get_local_date_time(mgp::list_at(args, 10)));

        std::vector<mage::LocalDateTime> x;
        x.push_back(local_date_time_1);
        std::set<mage::LocalDateTime> y;
        y.insert(local_date_time_1);
        std::unordered_set<mage::LocalDateTime> z;
        z.insert(local_date_time_1);
      }

      // Test Duration:
      if (false) {
        auto duration_1 = mage::Duration(mgp::value_get_duration(mgp::list_at(args, 11)));

        std::vector<mage::Duration> x;
        x.push_back(duration_1);
        std::set<mage::Duration> y;
        y.insert(duration_1);
        std::unordered_set<mage::Duration> z;
        z.insert(duration_1);
      }
    }

    // auto graph = mage::Graph(memgraph_graph);

    // std::unordered_map<std::int64_t, std::int64_t> vertex_component;
    // std::int64_t curr_component = 0;

    // std::cout << "in\n";
    // for (auto vertex : graph.vertices()) {
    //   if (vertex_component.find(vertex.id().AsInt()) != vertex_component.end()) continue;

    //   std::cout << "in2\n";
    //   // Run BFS from current vertex.
    //   std::queue<std::int64_t> q;

    //   q.push(vertex.id().AsInt());
    //   vertex_component[vertex.id().AsInt()] = curr_component;
    //   while (!q.empty()) {
    //     std::cout << "in3\n";
    //     auto v_id = q.front();
    //     q.pop();

    //     // Iterate over inbound edges.
    //     std::vector<std::int64_t> neighbor_ids;
    //     for (auto out_edge : graph.GetNodeById(mage::Id::FromInt(v_id)).out_edges()) {
    //       std::cout << "in4a\n";
    //       auto destination = out_edge.to();
    //       std::cout << "in4a1\n";
    //       neighbor_ids.push_back(destination.id().AsInt());
    //       std::cout << "in4a2\n";
    //     }

    //     std::cout << "in4b_pre\n";
    //     for (auto neighbor_id : neighbor_ids) {
    //       std::cout << "in4b\n";
    //       if (vertex_component.find(neighbor_id) != vertex_component.end()) {
    //         continue;
    //       }
    //       vertex_component[neighbor_id] = curr_component;
    //       q.push(neighbor_id);
    //     }
    //     std::cout << "in4b_aft\n";
    //   }
    //   ++curr_component;
    // }
    // std::cout << "ne\n";

    // auto record_factory = mage::RecordFactory(result);

    // for (const auto [vertex_id, component_id] : vertex_component) {
    //   // Insert each weakly component record
    //   auto record = record_factory.NewRecord();
    //   record.Insert(kFieldNode, graph.GetNodeById(mage::Id::FromInt(vertex_id)));
    //   record.Insert(kFieldComponentId, component_id);
    // }
  } catch (const std::exception &e) {
    // We must not let any exceptions out of our module.
    mgp::result_set_error_msg(result, e.what());
    return;
  }
}

void TestProc2(mgp_list *args, mgp_graph *memgraph_graph, mgp_result *result, mgp_memory *memory) {
  mage::memory = memory;

  try {
    std::cout << "w\n";
    auto path_1 = mgp::value_get_path(mgp::list_at(args, 0));
    std::cout << "w\n";
    // auto path_2 = mgp::path_copy(path_1, memory);
    auto x = mage::Path(path_1);
    std::cout << "w\n";
  } catch (const std::exception &e) {
    // We must not let any exceptions out of our module.
    mgp::result_set_error_msg(result, e.what());
    return;
  }
}

extern "C" int mgp_init_module(struct mgp_module *module, struct mgp_memory *memory) {
  try {
    auto *test_proc = mgp::module_add_read_procedure(module, kProcedureRun, TestProc);

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
    auto wrapper = mage::ProcedureWrapper();

    auto *test_proc = mgp::module_add_read_procedure(module, kProcedurePathCheck, wrapper.MGPProc);

    mgp::proc_add_arg(test_proc, kArgumentPath, mgp::type_path());

    mgp::proc_add_result(test_proc, kFieldOut, mgp::type_int());
  } catch (const std::exception &e) {
    return 1;
  }

  return 0;
}

extern "C" int mgp_shutdown_module() { return 0; }