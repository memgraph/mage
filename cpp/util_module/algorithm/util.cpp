#include "util.hpp"
#include "md5.hpp"

void Util::Md5Procedure(mgp_list *args, mgp_graph *memgraph_graph, mgp_result *result, mgp_memory *memory) {
  mgp::MemoryDispatcherGuard guard{memory};
  const auto arguments = mgp::List(args);
  const auto record_factory = mgp::RecordFactory(result);
  try {
    if (arguments[0].IsList()) {
      const mgp::List arg_list = arguments[0].ValueList();
      std::string return_string{""};
      for (auto value : arg_list) {
        return_string += value.ToString();
      }
      auto record = record_factory.NewRecord();
      record.Insert(std::string(kArgumentResultMd5).c_str(), md5(return_string));
    } else {
      auto record = record_factory.NewRecord();
      auto string_value = std::string(arguments[0].ToString());
      record.Insert(std::string(kArgumentResultMd5).c_str(), md5(string_value));
    }
  } catch (const std::exception &e) {
    record_factory.SetErrorMessage(e.what());
    return;
  }
}

void Util::Md5Function(mgp_list *args, mgp_func_context *func_context, mgp_func_result *res, mgp_memory *memory) {
  mgp::MemoryDispatcherGuard guard{memory};
  const auto arguments = mgp::List(args);
  auto result = mgp::Result(res);
  try {
    if (arguments[0].IsList()) {
      const mgp::List arg_list = arguments[0].ValueList();
      std::string return_string{""};
      for (auto value : arg_list) {
        return_string += value.ToString();
      }
      result.SetValue(md5(return_string));
    } else {
      auto string_to_hash = std::string(arguments[0].ToString());
      result.SetValue(md5(string_to_hash));
    }
  } catch (const std::exception &e) {
    result.SetErrorMessage(e.what());
    return;
  }
}
