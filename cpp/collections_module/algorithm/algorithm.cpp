#include "algorithm.hpp"
#include <unordered_set>

void Collections::toSet(mgp_list *args, mgp_graph *memgraph_graph, mgp_result *result, mgp_memory *memory){
    mgp::memory = memory;
    auto arguments = mgp::List(args);
    const auto record_factory = mgp::RecordFactory(result);
    try {
        mgp::List list = arguments[0].ValueList();
        std::unordered_set<mgp::Value, std::hash<mgp::Value>> set;
        mgp::List return_list;
        for(auto list_elem: list){
            if(set.find(list_elem) == set.end()){
                set.insert(list_elem);
                return_list.AppendExtend(std::move(list_elem));
            }
        }
        auto record = record_factory.NewRecord();
        record.Insert(std::string(kReturnToSet).c_str(), std::move(return_list));

    }catch (const std::exception &e) {
        record_factory.SetErrorMessage(e.what());
        return;
  }
}