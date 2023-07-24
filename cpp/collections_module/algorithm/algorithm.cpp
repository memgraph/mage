#include "algorithm.hpp"


void Collections::Partition(mgp_list *args, mgp_graph *memgraph_graph, mgp_result *result, mgp_memory *memory) {
  mgp::memory = memory;
  auto arguments = mgp::List(args);
  const auto record_factory = mgp::RecordFactory(result);
  try {
    auto record = record_factory.NewRecord();
    mgp::List input_list = arguments[0].ValueList();
    const int64_t &partition_size = arguments[1].ValueInt();
    
    int64_t current_size = 0;
    mgp::List temp;
    mgp::List result;
    for(mgp::Value list_value: input_list){
        if(current_size == 0){
            temp = mgp::List();
        }
        temp.AppendExtend(std::move(list_value));
        current_size++;

        if(current_size == partition_size){
            mgp::Value temp_result = mgp::Value(std::move(temp));
            result.AppendExtend(std::move(temp_result));
            current_size = 0;
        }


    }

    if(current_size != partition_size && current_size !=0 ){
        mgp::Value temp_result = mgp::Value(std::move(temp));
        result.AppendExtend(std::move(temp_result));
        
    }
    record.Insert(std::string(kReturnValuePartition).c_str(), std::move(result));

    

  } catch (const std::exception &e) {
    record_factory.SetErrorMessage(e.what());
    return;
  }
}