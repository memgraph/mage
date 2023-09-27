#pragma once

#include <mgp.hpp>
#include <cmath>
#include <vector>
#include <unordered_set>
#include <queue>
#include <unordered_map>
#include <set>
#include <memory>

namespace Algo {

    class RelObject{
        public:
            int id;
            int id_prev;
            mgp::Relationship rel;

            RelObject(int id, int id_prev, const mgp::Relationship &rel): id(id), id_prev(id_prev), rel(rel) {}

            bool operator==(const RelObject& other) const {
                return id == other.id;
            }

            struct Hash {
                size_t operator()(const RelObject& rel) const {
                    return std::hash<int>()(rel.id);
                }
            };
    };
    class NodeObject{
        public:
            double heuristic_distance;
            double total_distance;
            mgp::Node node;

            NodeObject(const double heuristic_distance, const double total_distance, const mgp::Node &node)
            :  heuristic_distance(heuristic_distance), total_distance(total_distance), node(node){}
            

            bool operator==(const NodeObject& other) const {
                return node.Id() == other.node.Id();
            }

            bool operator<(const NodeObject& other) const { //flipped because of priority queue
                return (total_distance + heuristic_distance) > (other.total_distance + other.heuristic_distance);

            }


            const std::string ToString() const{
                return std::to_string(node.Id().AsInt()) + " " + std::to_string(total_distance) + " " + std::to_string(heuristic_distance);
            }

            struct Hash {
                size_t operator()(const NodeObject& nodeObj) const {
                    return std::hash<int64_t>()(nodeObj.node.Id().AsInt());
                }
            };

            struct Comp {
                bool operator()(const std::shared_ptr<NodeObject> &nodeObj, const std::shared_ptr<NodeObject> &nodeObj2){
                    return nodeObj->total_distance + nodeObj->heuristic_distance > nodeObj2->total_distance + nodeObj2->heuristic_distance;
                }
            };

    };

    class Open{
        public:
            std::priority_queue<std::shared_ptr<NodeObject>, std::vector<std::shared_ptr<NodeObject>>, NodeObject::Comp> pq;
            std::unordered_map<mgp::Id, double> set;

            bool Empty(){
                return set.empty();
            }

            const NodeObject& Top(){  //should this be just object without reference
                while(set.find(pq.top()->node.Id()) == set.end()){ //this is to make sure duplicates are ignored
                    pq.pop();
                }
                return *pq.top();
            }

            void Pop(){
                set.erase(pq.top()->node.Id());
                pq.pop();
            }

            bool Insert(const std::shared_ptr<NodeObject> &elem){ //returns true if we insert, or if we insert a better option
                auto it = set.find(elem->node.Id());
                if(it != set.end()){
                    if(elem->total_distance < it->second){
                        it->second = elem->total_distance;
                        pq.push(elem);
                        return true;
                    }
                    return false;
                }
                pq.push(elem);
                set.insert({elem->node.Id(), elem->total_distance});
                return true;
            }

            Open() = default;
    };

    
    class Closed{
        public:
            std::unordered_map<mgp::Id, double> closed;
            
            bool Empty(){
                return closed.empty();
            }
            void Erase(const NodeObject &obj){
                closed.erase(obj.node.Id());
            } 

            bool FindAndCompare(const NodeObject &obj){
                auto it = closed.find(obj.node.Id());
                if(it != closed.end()){
                    bool erase = it->second > obj.total_distance;
                    if(erase){
                        closed.erase(it);
                    }
                    return erase;
                }
                return true;
            }

            void Insert(const NodeObject &obj){
                closed.insert({obj.node.Id(), obj.total_distance});
            }

            Closed() = default;

    };

    class Config{
        public:

            bool unweighted = false;
            double epsilon = 1.0;
            std::string distance_prop = "distance";
            std::string heuristic_name = "";
            std::string latitude_name = "lat";
            std::string longitude_name = "lon";

            Config(const mgp::Map &map){
                if(!map.At("unweighted").IsNull() && map.At("unweighted").IsBool()){
                    unweighted = map.At("unweighted").ValueBool();
                }
                if(!map.At("epsilon").IsNull() && map.At("epsilon").IsNumeric()){
                    epsilon = map.At("epsilon").ValueNumeric();
                }
                if(!map.At("distance_prop").IsNull() && map.At("distance_prop").IsString()){
                    distance_prop = map.At("distance_prop").ValueString();
                }
                if(!map.At("heuristic_name").IsNull() && map.At("heuristic_name").IsString()){
                    heuristic_name = map.At("heuristic_name").ValueString();
                }
                if(!map.At("latitude_name").IsNull() && map.At("latitude_name").IsString()){
                    latitude_name = map.At("latitude_name").ValueString();
                }
                if(!map.At("longitude_name").IsNull() && map.At("longitude_name").IsString()){
                    longitude_name = map.At("longitude_name").ValueString();
                }
            }

            std::string ToString(){
                return distance_prop + " " + heuristic_name + " " + latitude_name + " " + longitude_name + " " + std::to_string(epsilon) + " " + std::to_string(unweighted);
            }

    };

    struct GoalNodes{
        const mgp::Node start;
        const mgp::Node target;
        std::pair<double,double> latLon;

        GoalNodes(const mgp::Node &start, const mgp::Node &target, const std::pair<double,double> latLon): start(start), target(target), latLon(latLon){}
    };

    struct Lists{
        Open open;
        Closed closed;
        std::unordered_set<RelObject, RelObject::Hash> visited_rel;

        Lists() = default;

    };

    double haversineDistance(double lat1, double lon1, double lat2, double lon2);
    double toRadians(double degrees);
    void AStar(mgp_list *args, mgp_graph *memgraph_graph, mgp_result *result, mgp_memory *memory); 
    mgp::Path BuildResult(const std::unordered_set<RelObject, RelObject::Hash> &vis_rel, const mgp::Node &startNode, int id);
    mgp::Path HelperAstar(const GoalNodes &nodes, const Config &config);
    void ParseRelationships(const mgp::Relationships &rels,  bool in, const GoalNodes &nodes, NodeObject* prev, Lists &lists, const Config &config);
    double CalculateHeuristic(const Config &config, const mgp::Node &node, const GoalNodes &nodes);
    std::pair<double, double> TargetLatLon(const mgp::Node &target, const Config &config);
    double CalculateDistance(const Config &config, const mgp::Relationship &rel);
    

}  // namespace Algo
