#include "csv_utils.hpp"

namespace CsvUtils {

void CreateCsvFile(mgp_list *args, mgp_graph *memgraph_graph, mgp_result *result, mgp_memory *memory) {
  mgp::MemoryDispatcherGuard guard(memory);
  const auto arguments = mgp::List(args);
  const auto record_factory = mgp::RecordFactory(result);

  try {
    const std::string_view filepath = arguments[0].ValueString();
    const auto content = arguments[1].ValueString();
    const auto isAppend = arguments[2].ValueBool();

    std::ofstream fout;
    fout.open(std::string(filepath), isAppend ? std::ofstream::app : std::ofstream::out);
    fout << content << std::endl;
    fout.close();

    auto record = record_factory.NewRecord();
    record.Insert(std::string(filepath).c_str(), filepath);

    return;

  } catch (const std::exception &e) {
    record_factory.SetErrorMessage(e.what());
    return;
  }
  return;
}

void DeleteCsvFile(mgp_list *args, mgp_graph *memgraph_graph, mgp_result *result, mgp_memory *memory) {
  mgp::MemoryDispatcherGuard guard(memory);
  const auto arguments = mgp::List(args);
  const auto record_factory = mgp::RecordFactory(result);

  try {
    const std::string_view filepath = arguments[0].ValueString();

    int result = std::remove(std::string(filepath).c_str());

    return;

  } catch (const std::exception &e) {
    record_factory.SetErrorMessage(e.what());
    return;
  }
  return;
}

}  // namespace CsvUtils
