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

    return distance * 1000;
}

mgp::Path Algo::BuildResult(const std::unordered_set<RelObject, RelObject::Hash> &vis_rel, const mgp::Node &startNode, int id){

    auto dummy_rel = (*vis_rel.begin()).rel;
    auto relobj = RelObject(id, 0, dummy_rel);
    int start_id = startNode.Id().AsInt();
    std::vector<mgp::Relationship> final_rels;
    while(relobj.id != start_id){
        auto other = vis_rel.find(relobj);
        final_rels.push_back(other->rel);
        relobj.id = other->id_prev;
    }

    mgp::Path path = mgp::Path(startNode);
    for(auto it = final_rels.rbegin(); it != final_rels.rend(); ++it){
        path.Expand(*it);
    }

    return path;


}


bool RelOk(){
    return true;
}

bool LabelOk(){
    return true;
}
void Algo::ParseRelationships(const mgp::Relationships &rels, Open &open, bool in, const mgp::Node &target, NodeObject* prev, Closed &closed, std::unordered_set<RelObject, RelObject::Hash> &vis_rel){
    for(const auto rel: rels){
        if(!RelOk()){
            continue;
        }
        const auto node = in ? rel.From() : rel.To();
        if(!LabelOk){
            continue;
        }
        NodeObject nb = NodeObject(node.GetProperty("heur").ValueNumeric(), rel.GetProperty("distance").ValueNumeric() + prev->total_distance, node);
        if(!closed.FindAndCompare(nb)){
            continue;
        }
        open.Insert(nb);
        
        int node_id = node.Id().AsInt();
        int prev_id = prev->node.Id().AsInt();
        RelObject relobj = RelObject(node_id, prev_id, rel);
        auto it = vis_rel.find(relobj);
        if(it == vis_rel.end()){ //not sure if this if loop is needed
            vis_rel.insert(relobj);
        }else{
            vis_rel.erase(it);
            vis_rel.insert(relobj);
        }
    }

}

Algo::NodeObject Algo::InitializeStart(mgp::Node &startNode){
    if(startNode.InDegree() == 0 && startNode.OutDegree() == 0){
        throw mgp::ValueException("Start node doesnt have any ingoing or outgoing relationships");
    }

    if(startNode.InDegree() != 0){
        NodeObject nb = NodeObject( 0.0, 0.0, startNode);
        return nb;
    }
    NodeObject nb = NodeObject(0.0, 0.0, startNode);
    return nb;
}

mgp::Path Algo::HelperAstar(mgp::Node &start, const mgp::Node &target){
    Open open = Open();
    Closed closed = Closed();
    std::unordered_set<RelObject, RelObject::Hash> vis_rel;
    auto start_nb = InitializeStart(start);
    open.Insert(start_nb);
    while(!open.Empty()){
        auto nb = open.Top();
        open.Pop();
        std::cout << nb.ToString() << std::endl;
        if(nb.node == target){
            return BuildResult(vis_rel, start, nb.node.Id().AsInt());
        }
        closed.Insert(nb);
        ParseRelationships(nb.node.OutRelationships(), open, false, target, &nb, closed, vis_rel);

    }
    return mgp::Path(start);
}
 

void Algo::AStar(mgp_list *args, mgp_graph *memgraph_graph, mgp_result *result, mgp_memory *memory) {
  mgp::MemoryDispatcherGuard guard{memory};
  const auto arguments = mgp::List(args);
  const auto record_factory = mgp::RecordFactory(result);

  try {
    auto start = arguments[0].ValueNode();
    auto target = arguments[1].ValueNode();
    std::cout << HelperAstar(start,target).ToString() << std::endl;

  } catch (const std::exception &e) {
    record_factory.SetErrorMessage(e.what());
    return;
  }
}

