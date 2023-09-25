#pragma once

#include <mgp.hpp>
#include <cmath>
#include <vector>
#include <unordered_set>
#include <queue>
#include <unordered_map>
#include <set>

namespace Algo {
    class NodeObject{
        public:
            NodeObject* prev;
            double heuristic_distance;
            double total_distance;
            mgp::Relationship rel;
            mgp::Node node;

            NodeObject(NodeObject* prev, const double heuristic_distance, const double total_distance, const mgp::Relationship &rel, const mgp::Node &node)
            : prev(prev), heuristic_distance(heuristic_distance), total_distance(total_distance), rel(rel), node(node){}
            

            bool operator==(const NodeObject& other) const {
                return node.Id() == other.node.Id();
            }

            bool operator<(const NodeObject& other) const { //flipped because of priority queue
                return (total_distance + heuristic_distance) > (other.total_distance + other.heuristic_distance);

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
            std::unordered_set<mgp::Id> set;

            bool Empty(){
                return set.empty();
            }

            const NodeObject& Top(){
                while(set.find(pq.top().node.Id()) == set.end()){
                    pq.pop();
                }
                return pq.top();
            }

            void Pop(){
                set.erase(pq.top().node.Id());
                pq.pop();
            }

            void Insert(const NodeObject &elem){
                pq.push(elem);
                set.insert(elem.node.Id());
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
                    bool erase = it->second < obj.total_distance + obj.heuristic_distance;
                    if(erase){
                        closed.erase(it);
                    }
                    return erase;
                }
                return true;
            }

            void Insert(const NodeObject &obj){
                closed.insert({obj.node.Id(), obj.heuristic_distance + obj.total_distance});
            }

            Closed() = default;

    };

    double haversineDistance(double lat1, double lon1, double lat2, double lon2);
    double toRadians(double degrees);
    void AStar(mgp_list *args, mgp_graph *memgraph_graph, mgp_result *result, mgp_memory *memory); 
    mgp::Path BuildResult(const std::vector<mgp::Relationship> &rels, const mgp::Node &startNode);
    void HelperAstar(mgp::Node &start, const mgp::Node &target);
    void FindPath(NodeObject &final, std::vector<mgp::Relationship> &rels);
    void ParseRelationships(const mgp::Relationships &rels, Open &open, bool in, const mgp::Node &target, NodeObject* prev, Closed &closed);

}  // namespace Algo
