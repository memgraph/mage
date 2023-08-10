#include "path.hpp"

/*function to set appropriate parameters for filtering*/
void Path::FilterLabel(const std::string_view label, const std::unordered_set<std::string> &termination_list, const std::unordered_set<std::string> &blacklist,
const std::unordered_set<std::string> &whitelist, const std::unordered_set<std::string> &end_list,
bool &blacklisted, bool &terminated, bool &end_node, bool &whitelisted){

    const std::string label_string = std::string(label);
    if(blacklist.find(label_string) != blacklist.end()){ //if label is blacklisted
        blacklisted = true;
    }

    if(termination_list.find(label_string) != termination_list.end()){//if label is termination label
        terminated = true;
    }

    if(end_list.find(label_string) != end_list.end()){//if label is end label 
        end_node = true;
    }

    if(whitelist.find(label_string) != whitelist.end()){//if label is whitelisted
        whitelisted = true;
    }
}

/*function that takes input list of labels, and sorts them into appropriate category
sets were used so when filtering is done, its done in O(1)*/
void Path::ParseLabels(const mgp::List &list_of_labels,std::unordered_set<std::string> &termination_list, std::unordered_set<std::string> &blacklist,
std::unordered_set<std::string> &whitelist, std::unordered_set<std::string> &end_list){
    for(const auto label: list_of_labels){
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
            default:  //default is that everything goes to whitelist, unless specified as above
                whitelist.insert(label_string);
                break;
        }
        

    }
    

}

/*function that takes input list of relationships, and sorts them into appropriate categories
sets were also used to reduce complexity*/
void Path::Parse_Relationships(const mgp::List &list_of_relationships, std::unordered_set<std::string> &outgoing_rel, std::unordered_set<std::string> &incoming_rel
,std::unordered_set<std::string> &any_rel, bool &any_outgoing, bool &any_incoming){

    if(list_of_relationships.Size() == 0){ //if no relationships were passed as arguments, all relationships are allowed
        any_outgoing = true;
        any_incoming = true;
        return;
    }
    for(const auto rel: list_of_relationships){
        std::string rel_type = std::string(rel.ValueString());
        const size_t size = rel_type.size();
        const char first_elem = rel_type[0];
        const char last_elem = rel_type[size-1];
        if(first_elem == '<' && last_elem == '>'){
            throw mgp::ValueException("Wrong relationship format => <relationship> is not allowed!");
        }else if(first_elem == '<' && size == 1){  
            any_incoming = true;//all incoming relatiomships are allowed
        }else if(first_elem == '<' && size != 1){ 
            incoming_rel.insert(rel_type.erase(0,1));//only specified incoming relationships are allowed
        }else if(last_elem == '>' && size == 1){ 
            any_outgoing = true;//all outgoing relationships are allowed
            
        }else if(last_elem == '>' && size != 1){ 
            rel_type.pop_back();
            outgoing_rel.insert(rel_type);//only specifed outgoing relationships are allowed
        }else{  //if not specified, a relationship goes both ways
            any_rel.insert(rel_type);
        }
    }
}

/*function used for traversal and filtering*/
void Path::Path_DFS(mgp::Path path, std::unordered_set<mgp::Relationship> relationships_set, std::vector<mgp::Path> &path_list, 
size_t path_size, const int64_t min_hops, const int64_t max_hops,
const std::unordered_set<std::string> &termination_list, const std::unordered_set<std::string> &blacklist,
const std::unordered_set<std::string> &whitelist, const std::unordered_set<std::string> &end_list, 
const bool &end_node_activated, const bool &whitelist_empty, const bool &termination_activated,
std::unordered_set<std::string> &outgoing_rel, std::unordered_set<std::string> &incoming_rel
,std::unordered_set<std::string> &any_rel, bool &any_outgoing, bool &any_incoming){

    const mgp::Node &node = path.GetNodeAt(path_size); //get latest node in path, and expand on it

    /*the function is split inot outgoing and incomin rels*/

    /*QUESTION FOR REVIEWER- IS IT BETTER FOR SPEED TO INCLUDE AN IF CLAUSE HERE, FOR EXAMPLE
    if(outgoing_rel.size == 0 && !any_outgoing), WHICH MEANS NO OUTGOING RELATIONSHIPS ARE ALLOWED
    TO SKIP THIS FOR LOOP FOR OUTGOING RELS*/

    //outgoing rels
    for(auto rel: node.OutRelationships()){
        //go through every relationship of the node and expand to the other node of the relationship
        if(path_size > 0 && rel == path.GetRelationshipAt(path_size - 1)){ //to prevent backtracking the same way path came from, and starting a loop of back and forth
            continue;
        }
        if(relationships_set.find(rel) != relationships_set.end()){ //relationships_set contains all relationships already visited in this path, and the usage of this if loop is to evade cycles
            continue;
        }
        mgp::Path cpy = mgp::Path(path); // I make a copy here because I need to store every path and sub-path as a separate result, feel free to give me a better idea :)
        std::string rel_type = std::string(rel.Type());
        if(!any_outgoing && (outgoing_rel.find(rel_type) == outgoing_rel.end()) && (any_rel.find(rel_type) == any_rel.end())){ //check if relationship is allowed or all relationships are allowed
            continue;
        }
        cpy.Expand(rel); //expand the path with this relationships
        std::unordered_set<mgp::Relationship> relationships_set_cpy = relationships_set; /* NOTE TO REWIEVER- HERE, I ALSO MAKE A COPY
        BECAUSE EVERY PATH AND SUBPATH HAVE THEIR OWN RELATIONSHIPS VISITED, BUT I FEEL THIS CAN BE DONE BETTER, BUT WHEN I DO IT WITHOUT THE COPY
        IT DOESNT WORK AS INTENDED, THOUGHTS?*/
        relationships_set_cpy.insert(rel); //insert the relationship into visited relationships
        
        /*this part is for label filtering*/
        bool blacklisted = false;
        bool whitelisted = false;
        bool end_node = false;
        bool terminated = false;
        for(auto label: rel.To().Labels()){  //set booleans to their value for the label of the finish node
            FilterLabel(label, termination_list , blacklist, whitelist, end_list, blacklisted, terminated, end_node, whitelisted);
        }

        /*if path is inside specified size, and the node to which we are expanding is not blacklisted
        or it is an end node(end_node_activated is se to true at the beggining of the procedure if there is end nodes, not sure if it is needed)

        or the node is a termination node

        or if there are no termination nodes(termination_activated == false => this is neccesary because if there are specified termination labels, only paths ending in them can be returned)
        and the node is either whitelisted or the whitelist is empty(which means everything is whitelisted)

        save the path as result

        NOTE TO REVIEWER: this if loop is a crime against programming, so Im open to advice to make it more readable
        */
        if((path_size + 1 <= max_hops) && (path_size + 1 >= min_hops) && !blacklisted && ((end_node && end_node_activated) || terminated || (!termination_activated && !end_node_activated && (whitelisted || whitelist_empty)))){
            path_list.push_back(cpy); //save the path in result vector
        }

        /*if node is not blacklisted, or termination node and if it is an end node or in the whitelist, it can be expandend*/
        if(path_size + 1 < max_hops && !blacklisted && !terminated && (end_node || (whitelisted || whitelist_empty))){
            Path_DFS(cpy, relationships_set_cpy, path_list, path_size+1, min_hops, max_hops,termination_list, blacklist, whitelist, end_list, end_node_activated, whitelist_empty, termination_activated, outgoing_rel, incoming_rel, any_rel, any_outgoing, any_incoming);
        }
    }
    //incoming rels
    //the code here is similar as above, only few things are changed to accomdodate for INRelationship type, so the comments and notes are the saem
    for(auto rel: node.InRelationships()){
        if(path_size > 0 && rel == path.GetRelationshipAt(path_size - 1)){ //to prevent backtracking the same way path came from, and starting a loop of back and forth
            continue;
        }
        if(relationships_set.find(rel) != relationships_set.end()){
            continue;//prevent cycles
        }
        mgp::Path cpy = mgp::Path(path); //copy for distinct path result
        std::string rel_type = std::string(rel.Type());
        if(!any_incoming && (incoming_rel.find(rel_type) == incoming_rel.end()) && (any_rel.find(rel_type) == any_rel.end())){ //check if rel allowed
            continue;
        }
        cpy.Expand(rel);
        std::unordered_set<mgp::Relationship> relationships_set_cpy = relationships_set;
        relationships_set_cpy.insert(rel);
        
        bool blacklisted = false;
        bool whitelisted = false;
        bool end_node = false;
        bool terminated = false;
        for(auto label: rel.From().Labels()){//label filtering
            FilterLabel(label, termination_list , blacklist, whitelist, end_list, blacklisted, terminated, end_node, whitelisted);
        }


        //same if loops as in out relationships, crime against programming number #2
        if((path_size + 1 <= max_hops) && (path_size + 1 >= min_hops) && !blacklisted && ((end_node && end_node_activated) || terminated || (!termination_activated && !end_node_activated && (whitelisted || whitelist_empty)))){
            path_list.push_back(cpy);
        }
        if(path_size + 1 < max_hops && !blacklisted && !terminated && (end_node || (whitelisted || whitelist_empty))){
            Path_DFS(cpy, relationships_set_cpy, path_list, path_size+1, min_hops, max_hops, termination_list, blacklist, whitelist, end_list, end_node_activated, whitelist_empty, termination_activated, outgoing_rel, incoming_rel, any_rel, any_outgoing, any_incoming);
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
    if(end_list.size() != 0){ //end node is activated, which means only paths ending with it can be saved as result, but they can be expanded further
        end_node_activated = true;
    }
    bool whitelist_empty = false;
    if(whitelist.size() == 0){ //whitelist is emptym which means, everything is a whitelisted node
        whitelist_empty = true;
    }
    bool termination_activated = false;
    if(termination_list.size() != 0){
        termination_activated = true;  //there is a termination node, so only paths ending with it are allowed
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
    std::unordered_set<mgp::Relationship> relationships_set;
    Path_DFS(path, relationships_set, path_list, 0, min_hops, max_hops, termination_list, blacklist, whitelist, end_list, end_node_activated, whitelist_empty, termination_activated, outgoing_rel, incoming_rel, any_rel, any_outgoing, any_incoming);
  
    for(auto el: path_list){
        auto record = record_factory.NewRecord();
        record.Insert(std::string(std::string(Path::kResultExpand).c_str()).c_str(), std::move(el));
    }

    
  }catch (const std::exception &e) {
    record_factory.SetErrorMessage(e.what());
    return;
  }
}