#pragma once

#include <mgp.hpp>
#include <cmath>
#include <vector>
#include <unordered_set>
#include <queue>
#include <unordered_map>
#include <set>

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

    };

    class Open{
        public:
            std::priority_queue<NodeObject> pq;
            std::unordered_map<mgp::Id, double> set;

            bool Empty(){
                return set.empty();
            }

            const NodeObject& Top(){  //should this be just object without reference
                while(set.find(pq.top().node.Id()) == set.end()){ //this is to make sure duplicates are ignored
                    pq.pop();
                }
                return pq.top();
            }

            void Pop(){
                set.erase(pq.top().node.Id());
                pq.pop();
            }

            void Insert(const NodeObject &elem){
                auto it = set.find(elem.node.Id());
                if(it != set.end()){
                    if(elem.total_distance < it->second){
                        it->second = elem.total_distance;
                        pq.push(elem);
                    }
                    return;
                }
                pq.push(elem);
                set.insert({elem.node.Id(), elem.total_distance});
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

    double haversineDistance(double lat1, double lon1, double lat2, double lon2);
    double toRadians(double degrees);
    void AStar(mgp_list *args, mgp_graph *memgraph_graph, mgp_result *result, mgp_memory *memory); 
    mgp::Path BuildResult(const std::unordered_set<RelObject, RelObject::Hash> &vis_rel, const mgp::Node &startNode, int id);
    mgp::Path HelperAstar(mgp::Node &start, const mgp::Node &target);
    void ParseRelationships(const mgp::Relationships &rels, Open &open, bool in, const mgp::Node &target, NodeObject* prev, Closed &closed, std::unordered_set<RelObject, RelObject::Hash> &vis_rel);
    NodeObject InitializeStart(mgp::Node &startNode);

}  // namespace Algo
