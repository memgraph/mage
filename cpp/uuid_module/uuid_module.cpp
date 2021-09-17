// #include <uuid/uuid.h>

// #include <mg_procedure.h>
// #include <mg_exceptions.hpp>
// #include <mg_utils.hpp>

// namespace {

// constexpr char const *kProcedureGet = "get";

// constexpr char const *kFieldUuid = "uuid";

// void Generate(const mgp_list *args, const mgp_graph *memgraph_graph, mgp_result *result, mgp_memory *memory) {
//   try {
//     uuid_t id;
//     uuid_generate(id);
//     char *string = new char[100];
//     uuid_unparse(id, string);

//     mgp_result_record *record = mgp_result_new_record(result);
//     if (record == nullptr) {
//       throw mg_exception::NotEnoughMemoryException();
//     }

//     mg_utility::InsertStringValueResult(record, kFieldUuid, string, memory);
//   } catch (const std::exception &e) {
//     // We must not let any exceptions out of our module.
//     mgp_result_set_error_msg(result, e.what());
//     return;
//   }
// }
// }  // namespace

// extern "C" int mgp_init_module(struct mgp_module *module, struct mgp_memory *memory) {
//   struct mgp_proc *uuid_proc = mgp_module_add_read_procedure(module, kProcedureGet, Generate);
//   if (!uuid_proc) return 1;
//   if (!mgp_proc_add_result(uuid_proc, kFieldUuid, mgp_type_string())) return 1;

//   return 0;
// }

// extern "C" int mgp_shutdown_module() { return 0; }
