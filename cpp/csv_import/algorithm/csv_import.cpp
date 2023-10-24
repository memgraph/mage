#include "csv_import.hpp"

namespace CsvImport {

void CreateCsvFile(mgp_list *args, mgp_graph *memgraph_graph, mgp_result *result, mgp_memory *memory) {
  // mgp::memory = memory;
  mgp::MemoryDispatcherGuard guard(memory);
  const auto arguments = mgp::List(args);
  const auto record_factory = mgp::RecordFactory(result);

  try {
    const std::string_view filepath = arguments[0].ValueString();
    const auto content = arguments[1].ValueString();

    std::ofstream fout;
    fout.open(std::string(filepath));
    fout << content << std::endl;
    fout.close();

    mgp::List return_list;
    auto record = record_factory.NewRecord();
    record.Insert(std::string(kFilepath).c_str(), filepath);

    return;

  } catch (const std::exception &e) {
    record_factory.SetErrorMessage(e.what());
    return;
  }
  return;
}

void DeleteFile(mgp_list *args, mgp_graph *memgraph_graph, mgp_result *result, mgp_memory *memory) {
  // mgp::memory = memory;
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

}  // namespace CsvImport