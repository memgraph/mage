#include <mgp.hpp>
#include "collections.hpp"
#include <unordered_set>

void Collections::Contains(mgp_list *args, mgp_graph *memgraph_graph, mgp_result *result, mgp_memory *memory) {
  mgp::memory = memory;
  const auto arguments = mgp::List(args);
  const auto record_factory = mgp::RecordFactory(result);
  try {
    const mgp::List &list = arguments[0].ValueList();
    const mgp::Value &value = arguments[1];

    bool contains_value{false};
    auto record = record_factory.NewRecord();

    if (list.Empty()) {
      record.Insert(std::string(kReturnValueContains).c_str(), contains_value);
      return;
    }
    for (size_t i = 0; i < list.Size(); i++) {
      if (list[i] == value) {
        contains_value = true;
        break;
      }
    }
    record.Insert(std::string(kReturnValueContains).c_str(), contains_value);

  } catch (const std::exception &e) {
    record_factory.SetErrorMessage(e.what());
    return;
  }
}


void Collections::UnionAll(mgp_list *args, mgp_graph *memgraph_graph, mgp_result *result, mgp_memory *memory) {
  mgp::memory = memory;
  auto arguments = mgp::List(args);
  const auto record_factory = mgp::RecordFactory(result);
  try {
    mgp::List list1 = arguments[0].ValueList();
    mgp::List list2 = arguments[1].ValueList();

    for (size_t i = 0; i < list2.Size(); i++) {
      list1.AppendExtend(std::move(list2[i]));
    }
    auto record = record_factory.NewRecord();
    record.Insert(std::string(kReturnValueUnionAll).c_str(), list1);

  } catch (const std::exception &e) {
    record_factory.SetErrorMessage(e.what());
    return;
  }
}

void Collections::Min(mgp_list *args, mgp_graph *memgraph_graph, mgp_result *result, mgp_memory *memory) {
  mgp::memory = memory;
  const auto arguments = mgp::List(args);
  const auto record_factory = mgp::RecordFactory(result);
  try {
    const mgp::List &list = arguments[0].ValueList();
    if (list.Empty()) {
      throw mgp::ValueException("Empty input list");
    }
    const mgp::Type &type = list[0].Type();
    auto record = record_factory.NewRecord();

    if (type == mgp::Type::Map || type == mgp::Type::Path || type == mgp::Type::List) {
      std::ostringstream oss;
      oss << type;
      std::string s = oss.str();
      throw mgp::ValueException("Unsuppported type for this operation, receieved type: " + s);
    }

    bool isListNumeric = list[0].IsNumeric();
    mgp::Value min{std::move(list[0])};
    for (size_t i = 0; i < list.Size(); i++) {
      if (list[i].Type() != type && !(isListNumeric && list[i].IsNumeric())) {
        throw mgp::ValueException("All elements must be of the same type!");
      }

      if (list[i] < min) {
        min = std::move(list[i]);
      }
    }

    record.Insert(std::string(kReturnValueMin).c_str(), min);

  } catch (const std::exception &e) {
    record_factory.SetErrorMessage(e.what());
    return;
  }
}

void Collections::ToSet(mgp_list *args, mgp_graph *memgraph_graph, mgp_result *result, mgp_memory *memory) {
  mgp::memory = memory;
  auto arguments = mgp::List(args);
  const auto record_factory = mgp::RecordFactory(result);
  try {
    mgp::List list = arguments[0].ValueList();
    std::unordered_set<mgp::Value> set(list.begin(), list.end());
    mgp::List return_list;
    for (auto elem : set) {
      return_list.AppendExtend(std::move(elem));
    }
    auto record = record_factory.NewRecord();
    record.Insert(std::string(kReturnToSet).c_str(), std::move(return_list));

  } catch (const std::exception &e) {
    record_factory.SetErrorMessage(e.what());
    return;
  }
}

void Collections::Partition(mgp_list *args, mgp_graph *memgraph_graph, mgp_result *result, mgp_memory *memory) {
  mgp::memory = memory;
  auto arguments = mgp::List(args);
  const auto record_factory = mgp::RecordFactory(result);
  try {
    mgp::List input_list = arguments[0].ValueList();
    const int64_t partition_size = arguments[1].ValueInt();

    int64_t current_size = 0;
    mgp::List temp;
    mgp::List result;
    for (mgp::Value list_value : input_list) {
      if (current_size == 0) {
        temp = mgp::List();
      }
      temp.AppendExtend(std::move(list_value));
      current_size++;

      if (current_size == partition_size) {
        auto record = record_factory.NewRecord();
        record.Insert(std::string(kReturnValuePartition).c_str(), std::move(temp));
        current_size = 0;
      }
    }

    if (current_size != partition_size && current_size != 0) {
      auto record = record_factory.NewRecord();
      record.Insert(std::string(kReturnValuePartition).c_str(), std::move(temp));
    }

  } catch (const std::exception &e) {
    record_factory.SetErrorMessage(e.what());
    return;
  }
}
