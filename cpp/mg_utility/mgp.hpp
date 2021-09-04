#include "mg_exceptions.hpp"
#include "mg_procedure.h"

namespace mgp {

namespace {
void MgExceptionHandle(mgp_error_code result_code) {
  switch (result_code) {
    case MGP_ERROR_UNKNOWN_ERROR:
      throw mg_exception::UnknownException();
    case MGP_ERROR_UNABLE_TO_ALLOCATE:
      throw mg_exception::AllocationException();
    case MGP_ERROR_INSUFFICIENT_BUFFER:
      throw mg_exception::InsufficientBufferException();
    case MGP_ERROR_OUT_OF_RANGE:
      throw mg_exception::OutOfRangeException();
    case MGP_ERROR_LOGIC_ERROR:
      throw mg_exception::LogicException();
    case MGP_ERROR_NON_EXISTENT_OBJECT:
      throw mg_exception::NonExistendObjectException();
    case MGP_ERROR_INVALID_ARGUMENT:
      throw mg_exception::InvalidArgumentException();
    default:
      return;
  }
}
template <typename TResult, typename TFunc, typename... TArgs>
TResult MgInvoke(TFunc func, TArgs... args) {
  TResult result{};

  auto result_code = func(args..., &result);
  MgExceptionHandle(result_code);

  return result;
}

template <typename TFunc, typename... TArgs>
void MgInvokeVoid(TFunc func, TArgs... args) {
  auto result_code = func(args...);
  MgExceptionHandle(result_code);
}
}  // namespace

mgp_vertex_id vertex_get_id(const mgp_vertex *v) { return MgInvoke<mgp_vertex_id>(mgp_vertex_get_id, v); }

bool value_get_bool(const mgp_value *val) { return MgInvoke<int>(mgp_value_get_bool, val); }

mgp_vertex *value_get_vertex(const mgp_value *val) { return MgInvoke<mgp_vertex *>(mgp_value_get_vertex, val); }

mgp_value *list_at(mgp_list *list, size_t index) { return MgInvoke<mgp_value *>(mgp_list_at, list, index); }

mgp_proc *module_add_read_procedure(mgp_module *module, const char *name, mgp_proc_cb cb) {
  return MgInvoke<mgp_proc *>(mgp_module_add_read_procedure, module, name, cb);
}

void proc_add_result(mgp_proc *proc, const char *name, const struct mgp_type *type) {
  MgInvokeVoid(mgp_proc_add_result, proc, name, type);
}

void proc_add_arg(mgp_proc *proc, const char *name, const struct mgp_type *type) {
  MgInvokeVoid(mgp_proc_add_arg, proc, name, type);
}

void result_set_error_msg(struct mgp_result *res, const char *error_msg) {
  MgInvokeVoid(mgp_result_set_error_msg, res, error_msg);
}

mgp_result_record *result_new_record(struct mgp_result *res) {
  return MgInvoke<mgp_result_record *>(mgp_result_new_record, res);
}

const struct mgp_type *type_node() { return MgInvoke<const struct mgp_type *>(mgp_type_node); }
const struct mgp_type *type_int() { return MgInvoke<const struct mgp_type *>(mgp_type_int); }

struct mgp_value *value_make_int(int64_t val, struct mgp_memory *memory) {
  return MgInvoke<mgp_value *>(mgp_value_make_int, val, memory);
}

void result_record_insert(mgp_result_record *record, const char *field_name, const mgp_value *val) {
  MgInvokeVoid(mgp_result_record_insert, record, field_name, val);
}

void InsertIntValueResult(mgp_result_record *record, const char *field_name, const int int_value, mgp_memory *memory) {
  auto value = mgp::value_make_int(int_value, memory);
  mgp::result_record_insert(record, field_name, value);
  mgp_value_destroy(value);
}

}  // namespace mgp