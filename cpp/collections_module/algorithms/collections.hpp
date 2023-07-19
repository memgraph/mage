#include <mgp.hpp>


void contains(mgp_list *args, mgp_graph *memgraph_graph, mgp_result *result, mgp_memory *memory){
  mgp::memory=memory;
  const auto arguments=mgp::List(args);
  const auto record_factory = mgp::RecordFactory(result);
  try{
    mgp::List list=arguments[0].ValueList();
    mgp::Value value=arguments[1];
    bool contains_value=false;
    if(list.Size()==0 || list.Empty()){
      contains_value=false;
    }else{
      for(size_t i=0;i<list.Size();i++){
        if(list[i]==value){
          contains_value=true;
          break;
        }
      }
    }
    auto record=record_factory.NewRecord();
    record.Insert("output",contains_value);
    
  }catch (const std::exception &e) {
    record_factory.SetErrorMessage(e.what());
    return;
  }

}