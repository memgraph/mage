#include <chrono>
#include <fstream>

#include <mg_utils.hpp>

namespace {
constexpr char const *kProcedureRun = "run";
constexpr char const *kFieldInteger = "integer";

void InsertRecord(mgp_graph *graph, mgp_result *result, mgp_memory *memory) {
  auto *record = mgp::result_new_record(result);
  mg_utility::InsertIntValueResult(record, kFieldInteger, 1, memory);
}

void ProcedureRun(mgp_list *args, mgp_graph *memgraph_graph, mgp_result *result, mgp_memory *memory) {
  try {
    using std::chrono::duration;
    using std::chrono::high_resolution_clock;
    using std::chrono::milliseconds;

    auto t1 = high_resolution_clock::now();
    std::ofstream bench_file("benchmark.txt");
    auto t2 = high_resolution_clock::now();
    duration<double, std::milli> ms_double = t2 - t1;
    std::cout << "Test"
              << "ms\n";
    std::cerr << ms_double.count() << "ms\n";
    std::cout << ms_double.count() << "ms\n";

    bench_file << "Testing time: "
               << "100"
               << "ms\n";
    bench_file << "CreateCugraphFromMemgraph: " << ms_double.count() << "ms\n";
    bench_file.flush();
    bench_file << "test";
    bench_file.flush();
    bench_file.close();
    InsertRecord(memgraph_graph, result, memory);
  } catch (const std::exception &e) {
    // We must not let any exceptions out of our module.
    mgp::result_set_error_msg(result, e.what());
    return;
  }
}
}  // namespace

extern "C" int mgp_init_module(struct mgp_module *module, struct mgp_memory *memory) {
  try {
    auto *pagerank_proc = mgp::module_add_read_procedure(module, kProcedureRun, ProcedureRun);
    mgp::proc_add_result(pagerank_proc, kFieldInteger, mgp::type_int());
  } catch (const std::exception &e) {
    return 1;
  }

  return 0;
}

extern "C" int mgp_shutdown_module() { return 0; }
