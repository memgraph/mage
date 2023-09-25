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

mgp::Path Algo::BuildResult(const std::vector<mgp::Relationship> &rels, const mgp::Node &startNode){

    auto resultPath = mgp::Path(startNode);
    for (auto it = rels.rbegin(); it != rels.rend(); ++it){
        resultPath.Expand((*it));
    }
    return resultPath;
}

void Algo::FindPath(NodeObject &final, std::vector<mgp::Relationship> &rels){
    bool proceed = true;
    while(proceed){
        rels.push_back(final.rel);
        if(final.prev){
            final = (*final.prev);
            continue;
        }
        proceed = false;
        
    }
}

bool RelOk(){
    return true;
}

bool LabelOk(){
    return true;
}
void Algo::ParseRelationships(const mgp::Relationships &rels, Open &open, bool in, const mgp::Node &target, NodeObject* prev, Closed &closed){
    for(const auto rel: rels){
        if(!RelOk()){
            continue;
        }
        const auto node = in ? rel.From() : rel.To();
        if(!LabelOk){
            continue;
        }
        //STEP 1: if successor is target, stop search
        if(node == target){ 
            std::cout << "sol found" << std::endl;
            return;
        }
        NodeObject nb = NodeObject(prev, node.GetProperty("heur").ValueNumeric(), rel.GetProperty("distance").ValueNumeric(), rel, node);
        if(!closed.FindAndCompare(nb)){
            continue;
        }
        open.Insert(nb);


    }

}
void Algo::HelperAstar(mgp::Node &start, const mgp::Node &target){
    Open open = Open();
    Closed closed = Closed();
    ParseRelationships(start.OutRelationships(), open, false, target, nullptr, closed);
    while(!open.Empty()){
        std::cout << open.Top().heuristic_distance << std::endl;
        open.Pop();
    }
    for(auto &[key, value]: closed.closed){
        std::cout<< key.AsInt() << " " << value << std::endl;
    }
    
}
 

void Algo::AStar(mgp_list *args, mgp_graph *memgraph_graph, mgp_result *result, mgp_memory *memory) {
  mgp::MemoryDispatcherGuard guard{memory};
  const auto arguments = mgp::List(args);
  const auto record_factory = mgp::RecordFactory(result);

  try {
    auto start = arguments[0].ValueNode();
    HelperAstar(start,start);

  } catch (const std::exception &e) {
    record_factory.SetErrorMessage(e.what());
    return;
  }
}

