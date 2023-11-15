#include "csv_utils.hpp"

namespace CsvUtils {

void CreateCsvFile(mgp_list *args, mgp_graph *memgraph_graph, mgp_result *result, mgp_memory *memory) {
  mgp::MemoryDispatcherGuard guard(memory);
  const auto arguments = mgp::List(args);
  const auto record_factory = mgp::RecordFactory(result);

  try {
    const auto filepath = arguments[0].ValueString();
    const auto content = arguments[1].ValueString();
    const auto is_append = arguments[2].ValueBool();

    std::ofstream fout;
    fout.open(std::string(filepath), is_append ? std::ofstream::app : std::ofstream::out);
    fout << content << std::endl;
    fout.close();

    auto record = record_factory.NewRecord();
    record.Insert(std::string(kArgumentCreateCsvFile1).c_str(), filepath);

  } catch (const std::exception &e) {
    record_factory.SetErrorMessage(e.what());
  }
  return;
}

void DeleteCsvFile(mgp_list *args, mgp_graph *memgraph_graph, mgp_result *result, mgp_memory *memory) {
  mgp::MemoryDispatcherGuard guard(memory);
  const auto arguments = mgp::List(args);
  const auto record_factory = mgp::RecordFactory(result);

  try {
    const std::string_view filepath = arguments[0].ValueString();
    std::remove(std::string(filepath).c_str());

  } catch (const std::exception &e) {
    record_factory.SetErrorMessage(e.what());
  }
  return;
}

}  // namespace CsvUtils
