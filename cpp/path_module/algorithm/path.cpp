#include "path.hpp"

void FilterLabel(const std::string_view label, const std::unordered_set<std::string> &termination_list, const std::unordered_set<std::string> &blacklist,
const std::unordered_set<std::string> &whitelist, const std::unordered_set<std::string> end_list,
bool &blacklisted, bool &terminated, bool &end_node, bool &whitelisted){

    const std::string label_string = std::string(label);
    if(blacklist.find(label_string) != blacklist.end()){
        blacklisted = true;
    }

    if(termination_list.find(label_string) != termination_list.end()){
        terminated = true;
    }

    if(end_list.find(label_string) != end_list.end()){
        end_node = true;
    }

    if(whitelist.find(label_string) != whitelist.end()){
        whitelisted = true;
    }
}

void ParseLabels(const mgp::List &list_of_labels,std::unordered_set<std::string> &termination_list, std::unordered_set<std::string> &blacklist,
std::unordered_set<std::string> &whitelist, std::unordered_set<std::string> &end_list){
    for(auto label: list_of_labels){
        std::string label_string = std::string(label.ValueString());
        const char first_elem = label_string[0];
        switch(first_elem){
            case '-':
                blacklist.insert(label_string.erase(0,1));
                break;
            case '>':
                end_list.insert(label_string.erase(0,1));
                break;
            case '+':
                whitelist.insert(label_string.erase(0,1));
                break;
            case '/':
                termination_list.insert(label_string.erase(0,1));
                break;
            default:
                whitelist.insert(label_string);
                break;
        }
        

    }
    

}

void Parse_Relationships(const mgp::List &list_of_relationships, std::unordered_set<std::string> &outgoing_rel, std::unordered_set<std::string> &incoming_rel
,std::unordered_set<std::string> &any_rel, bool &any_outgoing, bool &any_incoming){
    if(list_of_relationships.Size() == 0){
        any_outgoing = true;
        any_incoming = true;
        return;
    }
    for(auto rel: list_of_relationships){
        std::string rel_type = std::string(rel.ValueString());
        const size_t size = rel_type.size();
        const char first_elem = rel_type[0];
        const char last_elem = rel_type[size-1];
        if(first_elem == '<' && last_elem == '>'){
            throw mgp::ValueException("Wrong relationship format => <relationship> is not allowed!");
        }else if(first_elem == '<' && size == 1){
            any_incoming = true;
        }else if(first_elem == '<' && size != 1){
            incoming_rel.insert(rel_type.erase(0,1));
        }else if(last_elem == '>' && size == 1){
            any_outgoing = true;
            
        }else if(last_elem == '>' && size != 1){
            rel_type.pop_back();
            outgoing_rel.insert(rel_type);
        }else{
            any_rel.insert(rel_type);
        }
    }
}
   
void Path_DFS(mgp::Path path, std::vector<mgp::Path> &path_list, size_t path_size, int64_t min_hops, int64_t max_hops,
const std::unordered_set<std::string> &termination_list, const std::unordered_set<std::string> &blacklist,
const std::unordered_set<std::string> &whitelist, const std::unordered_set<std::string> &end_list, 
const bool &end_node_activated, const bool &whitelist_empty, const bool &termination_activated,
std::unordered_set<std::string> &outgoing_rel, std::unordered_set<std::string> &incoming_rel
,std::unordered_set<std::string> &any_rel, bool &any_outgoing, bool &any_incoming){

    const mgp::Node &node = path.GetNodeAt(path_size);
    bool allowed = true;
    //outgoing rels
    for(auto rel: node.OutRelationships()){
        if(path_size > 0 && rel == path.GetRelationshipAt(path_size - 1)){ //to prevent backtracking the same way path came from, and starting a loop of back and forth
            continue;
        }
        mgp::Path cpy = mgp::Path(path);
        std::string rel_type = std::string(rel.Type());
        if(!any_outgoing && (outgoing_rel.find(rel_type) == outgoing_rel.end()) && (any_rel.find(rel_type) == any_rel.end())){
            continue;
        }
        cpy.Expand(rel);
        
        bool blacklisted = false;
        bool whitelisted = false;
        bool end_node = false;
        bool terminated = false;
        for(auto label: rel.To().Labels()){
            FilterLabel(label, termination_list , blacklist, whitelist, end_list, blacklisted, terminated, end_node, whitelisted);
        }

        if((path_size + 1 <= max_hops) && (path_size + 1 >= min_hops) && !blacklisted && ((end_node && end_node_activated) || terminated || (!termination_activated && !end_node_activated && (whitelisted || whitelist_empty)))){
            path_list.push_back(cpy);
        }
        if(path_size + 1 < max_hops && !blacklisted && !terminated && (end_node || (whitelisted || whitelist_empty))){
            Path_DFS(cpy, path_list, path_size+1, min_hops, max_hops,termination_list, blacklist, whitelist, end_list, end_node_activated, whitelist_empty, termination_activated, outgoing_rel, incoming_rel, any_rel, any_outgoing, any_incoming);
        }
    }
    //incoming rels
    for(auto rel: node.InRelationships()){
        if(path_size > 0 && rel == path.GetRelationshipAt(path_size - 1)){ //to prevent backtracking the same way path came from, and starting a loop of back and forth
            continue;
        }
        mgp::Path cpy = mgp::Path(path);
        std::string rel_type = std::string(rel.Type());
        if(!any_incoming && (incoming_rel.find(rel_type) == incoming_rel.end()) && (any_rel.find(rel_type) == any_rel.end())){
            continue;
        }
        cpy.Expand(rel);
        
        bool blacklisted = false;
        bool whitelisted = false;
        bool end_node = false;
        bool terminated = false;
        for(auto label: rel.From().Labels()){
            FilterLabel(label, termination_list , blacklist, whitelist, end_list, blacklisted, terminated, end_node, whitelisted);
        }

        if((path_size + 1 <= max_hops) && (path_size + 1 >= min_hops) && !blacklisted && ((end_node && end_node_activated) || terminated || (!termination_activated && !end_node_activated && (whitelisted || whitelist_empty)))){
            path_list.push_back(cpy);
        }
        if(path_size + 1 < max_hops && !blacklisted && !terminated && (end_node || (whitelisted || whitelist_empty))){
            Path_DFS(cpy, path_list, path_size+1, min_hops, max_hops, termination_list, blacklist, whitelist, end_list, end_node_activated, whitelist_empty, termination_activated, outgoing_rel, incoming_rel, any_rel, any_outgoing, any_incoming);
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
    const mgp::List labels = arguments[2].ValueList();
    int64_t min_hops = arguments[3].ValueInt();
    int64_t max_hops = arguments[4].ValueInt();

    /*filter label part*/
    std::unordered_set<std::string> termination_list;
    std::unordered_set<std::string> blacklist;
    std::unordered_set<std::string> end_list;
    std::unordered_set<std::string> whitelist;
    ParseLabels(labels,termination_list,blacklist,whitelist,end_list);
    bool end_node_activated = false;
    if(end_list.size() != 0){
        end_node_activated = true;
    }
    bool whitelist_empty = false;
    if(whitelist.size() == 0){
        whitelist_empty = true;
    }
    bool termination_activated = false;
    if(termination_list.size() != 0){
        termination_activated = true;
    }
    /*end filter label part*/

    /*filter relationships part*/
    std::unordered_set<std::string> outgoing_rel;
    std::unordered_set<std::string> incoming_rel;
    std::unordered_set<std::string> any_rel;
    bool any_outgoing = false;
    bool any_incoming = false;
    Parse_Relationships(relationships,outgoing_rel,incoming_rel, any_rel, any_outgoing, any_incoming);
    /*end filter relationships part*/

    mgp::Path path = mgp::Path(start_node);
    std::vector<mgp::Path> path_list;
    Path_DFS(path, path_list, 0, min_hops, max_hops, termination_list, blacklist, whitelist, end_list, end_node_activated, whitelist_empty, termination_activated, outgoing_rel, incoming_rel, any_rel, any_outgoing, any_incoming);
  
    for(auto el: path_list){
        auto record = record_factory.NewRecord();
        record.Insert(std::string("result").c_str(), std::move(el));
    }

    
  }catch (const std::exception &e) {
    record_factory.SetErrorMessage(e.what());
    return;
  }
}