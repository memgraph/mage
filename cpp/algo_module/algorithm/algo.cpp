#include "algo.hpp"

double Algo::toRadians(double degrees) {
    return degrees * M_PI / 180.0;
}

double Algo::haversineDistance(double lat1, double lon1, double lat2, double lon2) {

    const double earthRadius = 6371.0;
    
    lat1 = toRadians(lat1);
    lon1 = toRadians(lon1);
    lat2 = toRadians(lat2);
    lon2 = toRadians(lon2);

    double dLat = lat2 - lat1;
    double dLon = lon2 - lon1;
    double a = sin(dLat / 2) * sin(dLat / 2) + cos(lat1) * cos(lat2) * sin(dLon / 2) * sin(dLon / 2);
    double c = 2 * atan2(sqrt(a), sqrt(1 - a));
    double distance = earthRadius * c;

    return distance;  //returns distance in km, could be further configurated
}

/*calculates the heuristic based on haversine, or returns the value if the heuristic is custom*/
double Algo::CalculateHeuristic(const Config &config, const mgp::Node &node, const GoalNodes &nodes){

    if(config.heuristic_name != ""){
        auto heuristic = node.GetProperty(config.heuristic_name);
        if(heuristic.IsNumeric()){
            return heuristic.ValueNumeric();
        }
        throw mgp::ValueException("Custom heuristic property must be of a numeric data type!");
    }

    auto latitude_source = node.GetProperty(config.latitude_name);
    auto longitude_source = node.GetProperty(config.latitude_name);
    if(latitude_source.IsNull() || longitude_source.IsNull()){
        throw mgp::ValueException("Latitude and longitude properties, or a custom heuristic value, must be specified in every node!");
    }
    if(latitude_source.IsNumeric() && longitude_source.IsNumeric()){
        return haversineDistance(latitude_source.ValueNumeric(), longitude_source.ValueNumeric(), nodes.latLon.first, nodes.latLon.second);
    }
    throw mgp::ValueException("Latitude and longitude must be numeric data types!");

}

std::pair<double, double> Algo::TargetLatLon(const mgp::Node &target, const Config &config){

    if(config.heuristic_name != ""){  //if custom heuristic, dont return latitude and longitude
        return std::make_pair<double,double>(0,0);
    }

    auto latitude = target.GetProperty(config.latitude_name);
    auto longitude= target.GetProperty(config.latitude_name);
    if(latitude.IsNull() || longitude.IsNull()){
        throw mgp::ValueException("Latitude and longitude properties of the target node must be specified!");
    }
    if(latitude.IsNumeric() && longitude.IsNumeric()){
        return std::make_pair<double, double>(latitude.ValueNumeric(), longitude.ValueNumeric());
    }
    throw mgp::ValueException("Latitude and longitude must be numeric data types!");

}

double Algo::CalculateDistance(const Config &config, const mgp::Relationship &rel){
    if(config.unweighted){
        return 10;
    }
    auto distance = rel.GetProperty(config.distance_prop);
    if(distance.IsNull()){
        throw mgp::ValueException("If the graph is weighted, distance property of the relationship must be specified!");
    }
    if(distance.IsNumeric()){
        return distance.ValueNumeric();
    }
    throw mgp::ValueException("Distance property must be a numeric datatype!");

}

bool RelOk(){
    return true;
}

bool LabelOk(){
    return true;
}
void Algo::ParseRelationships(const mgp::Relationships &rels,  bool in, const GoalNodes &nodes, NodeObject* prev, Lists &lists, const Config &config){
    for(const auto rel: rels){
        if(!RelOk()){
            continue;
        }
        const auto node = in ? rel.From() : rel.To();
        if(!LabelOk){
            continue;
        }
        auto heuristic = CalculateHeuristic(config, node, nodes) * config.epsilon; //epsilon 0 == UCS
        auto distance = CalculateDistance(config, rel);
        std::shared_ptr<NodeObject> nb = std::make_shared<NodeObject>(heuristic, distance + prev->total_distance, node);
        if(!lists.closed.FindAndCompare(*nb)){
            continue;
        }
        if(!lists.open.Insert(nb)){  //this check is absolutely necessary, it may seem trivial, but, to any future refactorers, DO NOT REMOVE IT
            continue;   //you need thic check, so you dont add worse relationships than one that exists in the open list
        }
        

        //rel obj part

        RelObject relobj = RelObject(node.Id().AsInt(), prev->node.Id().AsInt(), rel);

        auto it = lists.visited_rel.find(relobj);
        if(it == lists.visited_rel.end()){
            lists.visited_rel.insert(relobj);
        }else{
            lists.visited_rel.erase(it);  //if we already visited with this relationship, we need to remove it and add better, because we passed the closed  and open test
            lists.visited_rel.insert(relobj);
        }
    }

}

mgp::Path Algo::HelperAstar(const GoalNodes &nodes, const Config &config){

    Lists lists = Lists();

    std::shared_ptr<NodeObject> start_nb = std::make_shared<NodeObject>(0,0,nodes.start);
    lists.open.Insert(start_nb);

    while(!lists.open.Empty()){

        auto nb = lists.open.Top();
        lists.open.Pop();
        std::cout << nb.ToString() << std::endl;
        if(nb.node == nodes.target){
            return BuildResult(lists.visited_rel, nodes.start, nb.node.Id().AsInt());
        }
        lists.closed.Insert(nb);

        ParseRelationships(nb.node.OutRelationships(), false, nodes, &nb, lists, config);
        ParseRelationships(nb.node.InRelationships(), true, nodes, &nb, lists, config);

    }
    return mgp::Path(nodes.start);
}
 

mgp::Path Algo::BuildResult(const std::unordered_set<RelObject, RelObject::Hash> &visited_rel, const mgp::Node &startNode, int id){

    mgp::Path path = mgp::Path(startNode);

    if(visited_rel.size() == 0){
        return path;
    }

    auto dummy_rel = (*visited_rel.begin()).rel;
    auto relobj = RelObject(id, 0, dummy_rel);
    int start_id = startNode.Id().AsInt();
    std::vector<mgp::Relationship> final_rels;

    while(relobj.id != start_id){
        auto other = visited_rel.find(relobj);
        final_rels.push_back(other->rel);
        relobj.id = other->id_prev;
    }

    for(auto it = final_rels.rbegin(); it != final_rels.rend(); ++it){
        path.Expand(*it);
    }

    return path;


}

void Algo::AStar(mgp_list *args, mgp_graph *memgraph_graph, mgp_result *result, mgp_memory *memory) {
  mgp::MemoryDispatcherGuard guard{memory};
  const auto arguments = mgp::List(args);
  const auto record_factory = mgp::RecordFactory(result);

  try {
    auto start = arguments[0].ValueNode();
    auto target = arguments[1].ValueNode();
    auto config = Config(arguments[2].ValueMap());
    std::pair<double, double> latLon = TargetLatLon(target, config);
    auto nodes = GoalNodes(start, target, latLon);
    std::cout << config.ToString() << std::endl;
    mgp::Path path = HelperAstar(nodes, config);
    
    auto record = record_factory.NewRecord();
    record.Insert("result", path);

  } catch (const std::exception &e) {
    record_factory.SetErrorMessage(e.what());
    return;
  }
}

