#include <chrono>
#include <fstream>
#include <locale>
#include <clocale>

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
    auto t2 = high_resolution_clock::now();
    duration<double, std::milli> ms_double = t2 - t1;
    std::cout << "Testing iostreams" << std::endl;
    std::cerr << "cerr: " << ms_double.count() << "ms" << std::endl;
    std::cout << "cout: " << ms_double.count() << "ms" << std::endl;
    InsertRecord(memgraph_graph, result, memory);
  } catch (...) {
    mgp::result_set_error_msg(result, "Error during ProcedureRun occured");
    return;
  }
}
}  // namespace

extern "C" int mgp_init_module(struct mgp_module *module, struct mgp_memory *memory) {
  // Check Notes https://en.cppreference.com/w/cpp/locale/setlocale
  std::setlocale(LC_ALL, "en_US.UTF-8");
  auto locale = std::locale{std::locale{"C"}, new std::num_put<char>{}};
  // Check https://en.cppreference.com/w/cpp/locale/locale/global
  // Does 0 related to six standard I/O C++ streams.
  // Check https://stackoverflow.com/a/25696480/4888809
  std::locale::global(locale);

  std::cout.imbue(locale);
  std::cerr.imbue(locale);
  std::clog.imbue(locale);
  std::wcout.imbue(locale);
  std::wcerr.imbue(locale);
  std::wclog.imbue(locale);

  try {
    auto *pagerank_proc = mgp::module_add_read_procedure(module, kProcedureRun, ProcedureRun);
    mgp::proc_add_result(pagerank_proc, kFieldInteger, mgp::type_int());
  } catch (const std::exception &e) {
    return 1;
  }

  return 0;
}

extern "C" int mgp_shutdown_module() { return 0; }
