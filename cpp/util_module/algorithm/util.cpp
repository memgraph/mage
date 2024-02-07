#include "util.hpp"
#include "md5.hpp"

void Util::Md5(mgp_list *args, mgp_graph *memgraph_graph, mgp_result *result, mgp_memory *memory) {
  mgp::MemoryDispatcherGuard guard{memory};
  const auto arguments = mgp::List(args);
  const auto record_factory = mgp::RecordFactory(result);
  try {
    const mgp::List arg_list = arguments[0].ValueList();
    std::string return_string{""};
    for (auto value : arg_list) {
      return_string += value.ToString();
    }
    auto record = record_factory.NewRecord();
    record.Insert(std::string(kArgumentResultMd5).c_str(), md5(return_string));
  } catch (const std::exception &e) {
    record_factory.SetErrorMessage(e.what());
    return;
  }
}

void Util::Md5List(mgp_list *args, mgp_graph *memgraph_graph, mgp_result *result, mgp_memory *memory) {
  mgp::MemoryDispatcherGuard guard{memory};
  const auto arguments = mgp::List(args);
  const auto record_factory = mgp::RecordFactory(result);
  try {
    const mgp::List arg_list = arguments[0].ValueList();
    auto return_list = mgp::List();
    std::string return_string{""};
    for (auto value : arg_list) {
      return_list.Append(mgp::Value(md5(value.ToString())));
    }

    auto record = record_factory.NewRecord();
    record.Insert(std::string(kArgumentResultMd5).c_str(), return_list);
  } catch (const std::exception &e) {
    record_factory.SetErrorMessage(e.what());
    return;
  }
}

void Util::Md5Func(mgp_list *args, mgp_func_context *func_context, mgp_func_result *res, mgp_memory *memory) {
  mgp::MemoryDispatcherGuard guard{memory};
  const auto arguments = mgp::List(args);
  auto result = mgp::Result(res);
  try {
    auto string_to_hash = std::string(arguments[0].ValueString());
    result.SetValue(md5(string_to_hash));
  } catch (const std::exception &e) {
    result.SetErrorMessage(e.what());
    return;
  }
}

void Util::Md5ListFunc(mgp_list *args, mgp_func_context *func_context, mgp_func_result *res, mgp_memory *memory) {
  mgp::MemoryDispatcherGuard guard{memory};
  const auto arguments = mgp::List(args);
  auto result = mgp::Result(res);
  try {
    const mgp::List arg_list = arguments[0].ValueList();
    auto return_list = mgp::List(arg_list.Size());
    std::string return_string{""};
    for (auto value : arg_list) {
      return_list.Append(mgp::Value(md5(value.ToString())));
    }

    result.SetValue(return_list);
  } catch (const std::exception &e) {
    result.SetErrorMessage(e.what());
    return;
  }
}
