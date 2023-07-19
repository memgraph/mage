#include <mgp.hpp>


void unionAll(mgp_list *args, mgp_graph *memgraph_graph, mgp_result *result, mgp_memory *memory){
  mgp::memory=memory;
  const auto arguments=mgp::List(args);
  const auto record_factory = mgp::RecordFactory(result);
  try{
    mgp::List list1=arguments[0].ValueList();
    mgp::List list2=arguments[1].ValueList();
    
    for (size_t i=0; i< list2.Size(); i++){
        list1.AppendExtend(list2[i]);
    }
    auto record=record_factory.NewRecord();
    record.Insert("return_list",list1);

  }catch (const std::exception &e) {
    record_factory.SetErrorMessage(e.what());
    return;
  }
}

