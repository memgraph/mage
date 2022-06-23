#include <thread>

#include <mg_utils.hpp>

namespace {

constexpr char const *kProcedureGet = "get";

constexpr char const *kFieldAddedValue = "added_value";

constexpr char const *kArgumentValue = "value";

void InsertResultRecord(mgp_result *result, mgp_memory *memory, const int value) {
  auto *record = mgp::result_new_record(result);

  mg_utility::InsertIntValueResult(record, kFieldAddedValue, value, memory);
}

void GetSum(mgp_list *args, mgp_graph *memgraph_graph, mgp_result *result, mgp_memory *memory) {
  try {

    auto value = mgp::value_get_int(mgp::list_at(args, 0));

    auto addedValue = mgp::add_10(value);

    InsertResultRecord(result, memory, addedValue);
    char array[] = "dobar dan";
    const char* pero = array;
    enum mgp_log_level level = mgp_log_level::Info;
    mg_utility::log(level, "Broj koji ste upisali {} {} {} {}", "je", value, "a broji koji ste dobili je",addedValue);
    mg_utility::log(level, pero);
    //mgp::log(level, pero);
    //mgp::log(level, "Broj koji ste upisali {} {} {} {}", "je", value, "a broji koji ste dobili je",addedValue);
      
  } catch (const std::exception &e) {
    // We must not let any exceptions out of our module.
    mgp::result_set_error_msg(result, e.what());
    return;
  }
}
}  // namespace

// Each module needs to define mgp_init_module function.
// Here you can register multiple procedures your module supports.
extern "C" int mgp_init_module(mgp_module *module, mgp_memory *memory) {
  mgp_value *int_value_number;

  try {
    auto *proc = mgp::module_add_read_procedure(module, kProcedureGet, GetSum);

    // Query module arguments
    int_value_number = mgp::value_make_int(1, memory);

    mgp::proc_add_opt_arg(proc, kArgumentValue, mgp::type_int(), int_value_number);

    // Query module output record
    mgp::proc_add_result(proc, kFieldAddedValue, mgp::type_int());

    


  } catch (const std::exception &e) {
    // Destroy the values if exception occurs
    mgp_value_destroy(int_value_number);
    return 1;
  }

  mgp_value_destroy(int_value_number);

  return 0;
}

// This is an optional function if you need to release any resources before the
// module is unloaded. You will probably need this if you acquired some
// resources in mgp_init_module.
extern "C" int mgp_shutdown_module() {
  // Return 0 to indicate success.
  return 0;
}
