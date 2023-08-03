#include "util.hpp"
#include "md5.hpp"

void Util::Md5(mgp_list *args, mgp_graph *memgraph_graph, mgp_result *result, mgp_memory *memory) {
  mgp::memory = memory;
  const auto arguments = mgp::List(args);
  const auto record_factory = mgp::RecordFactory(result);
  try {
    const mgp::List arg_list = arguments[0].ValueList();
    std::string return_string = "";
    for(auto value : arg_list){
      return_string += value.ToString();
    }
    const std::string md5_hash = md5(return_string);
    auto record = record_factory.NewRecord();
    record.Insert(std::string(kArgumentResultMd5).c_str(), md5_hash);
  } catch (const std::exception &e) {
    record_factory.SetErrorMessage(e.what());
    return;
  }
}