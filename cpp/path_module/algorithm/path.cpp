#include "path.hpp"


   
void Path_DFS(mgp::Path path, std::vector<mgp::Path> &path_list, size_t path_size, int64_t min_hops, int64_t max_hops, const std::unordered_set<std::string> &termination_list, const std::unordered_set<std::string> &blacklist){
    const mgp::Node &node = path.GetNodeAt(path_size);
    for(auto rel: node.OutRelationships()){
        //filter label part
        bool terminated= false;
        bool blacklisted = false;
        for(auto label : rel.To().Labels()){
            std::cout<< std::string(label) <<std::endl;
            //termination
            if(termination_list.find(std::string(label)) != termination_list.end()){
                terminated = true;
                break;
            } 
            //blacklist
            if(blacklist.find(std::string(label)) != blacklist.end()){
                blacklisted = true;
                break;
            } 

        }
        //end filter label part
        mgp::Path cpy = mgp::Path(path);
        cpy.Expand(rel);
        
        if((path_size + 1 <= max_hops) && (path_size + 1 >= min_hops) && !blacklisted){
            path_list.push_back(cpy);
        }
        
        if(path_size + 1 < max_hops && !terminated){
            Path_DFS(cpy, path_list, path_size+1, min_hops, max_hops,termination_list, blacklist);
        }
        
    }
}

void Path::Expand(mgp_list *args, mgp_graph *memgraph_graph, mgp_result *result, mgp_memory *memory) {
  mgp::memory = memory;
  const auto arguments = mgp::List(args);
  const auto record_factory = mgp::RecordFactory(result);
  try {
    const mgp::Node start_node = arguments[0].ValueNode();
    mgp::List relationships = arguments[1].ValueList();
    mgp::List labels = arguments[2].ValueList();
    int64_t min_hops = arguments[3].ValueInt();
    int64_t max_hops = arguments[4].ValueInt();

    mgp::Path path = mgp::Path(start_node);
    std::vector<mgp::Path> path_list;
    std::unordered_set<std::string> termination_list;
    std::unordered_set<std::string> blacklist;
    termination_list.insert("Terminate");
    blacklist.insert("Blacklist");
    Path_DFS(path, path_list, 0, min_hops, max_hops, termination_list, blacklist);
    for(auto el: path_list){
        auto record = record_factory.NewRecord();
        record.Insert(std::string("result").c_str(), std::move(el));
    }

    
  }catch (const std::exception &e) {
    record_factory.SetErrorMessage(e.what());
    return;
  }
}