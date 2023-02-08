#include <mgp.hpp>
#include <math.h>


namespace node_similarity_util {

struct pair_hash {
    template <class T1, class T2>
    std::size_t operator () (std::pair<T1, T2> const &pair) const
    {
        std::size_t h1 = std::hash<T1>()(pair.first);
        std::size_t h2 = std::hash<T2>()(pair.second);

        return h1 ^ h2;
    }
};

enum Similarity { jaccard, overlap, cosine };

// Methods
constexpr char const *jaccardAll = "jaccard";
constexpr char const *jaccardPairwise = "jaccard_pairwise";
constexpr char const *overlapAll = "overlap";
constexpr char const *overlapPairwise = "overlap_pairwise";
constexpr char const *cosineAll = "cosine";
constexpr char const *cosinePairwise = "cosine_pairwise";
// Parameter object names
constexpr char const *src_nodes = "src_nodes";
constexpr char const *dst_nodes = "dst_nodes";
constexpr char const *prop_vector = "property";
// Return object names
char const *node1_name = "node1";
char const *node2_name = "node2";
char const *similarity = "similarity";

}  // namespace node_similarity_util


namespace node_similarity_algs {


// Lambda functions for computing concrete similarities
/*
Calculates Jaccard similarity between two nodes based on their neighbours.
ns1 are neighbours of the first node
ns2 are neighbours of the second node
*/

double JaccardFunc(std::set<uint64_t> ns1, std::set<uint64_t> ns2) {
    std::set<uint64_t> elem_union, elem_intersection;
    std::set_intersection(ns1.begin(), ns1.end(), ns2.begin(), ns2.end(), std::inserter(elem_intersection, elem_intersection.begin()));
    std::set_union(ns1.begin(), ns1.end(), ns2.begin(), ns2.end(), std::inserter(elem_union, elem_union.begin()));
    if (elem_union.size() == 0) {
        return 0.0;
    } else {
        return elem_intersection.size() / (double) elem_union.size();
        // return std::ceil((elem_intersection.size() / (double) elem_union.size()) * 10000.0) / 10000.0;
    }
}

/*
Calculates overlap function between two set of neighbours.
ns1 are neighbours of the first node.
ns2 are neighbours of the second node.
*/
double OverlapFunc(std::set<uint64_t> ns1, std::set<uint64_t> ns2) {
    std::set<uint64_t> elem_intersection;
    std::set_intersection(ns1.begin(), ns1.end(), ns2.begin(), ns2.end(), std::inserter(elem_intersection, elem_intersection.begin()));
    int denonominator = std::min(ns1.size(), ns2.size());
    if (denonominator == 0) {
        return 0.0;
    } else {
        return elem_intersection.size() / (double) denonominator;
    }
}

/*
Calculates cosine similarity function between two nodes for a given property.
*/
double CosineFunc(const mgp::Node &node1, const mgp::Node &node2, const std::string &property) {
    std::cout << "Property: " << property << std::endl;
    auto prop1_it = node1.GetProperty(property);
    auto prop2_it = node2.GetProperty(property);
    for (const auto &[key, val]: node1.Properties()) {
        std::cout << key << " "; 
    }
    std::cout << std::endl;
    std::cout << "After checking whether the property exists" << std::endl;
    const auto &prop1 = prop1_it.ValueList();
    const auto &prop2 = prop2_it.ValueList();
    double similarity = 0.0, node1_sum = 0.0, node2_sum = 0.0;
    int size1 = prop1.Size(), size2 = prop2.Size();
    // std::co
    if (size1 != size2) {
        std::cout << "Sizes are different" << std::endl;
        throw mg_exception::InvalidArgumentException();
    }
    for (int i = 0; i < size1; ++i) {
        double val1 = prop1[i].ValueDouble(), val2 = prop2[i].ValueDouble();
        std::cout << "Val1: " << val1 << " " << val2 << std::endl;
        similarity += val1 * val2;
        node1_sum += val1 * val1;
        node2_sum += val2 * val2;
    }
    std::cout << "Before computation" << std::endl;
    double denominator = sqrt(node1_sum) * sqrt(node2_sum);
    std::cout << "After computation" << std::endl;
    if (denominator < 1e-9) {
        return 0.0;
    } else {
        return similarity / denominator;
    }
}



/*
Extract node neighbours.
*/
std::set<uint64_t> GetNeighbors(std::unordered_map<uint64_t, std::set<uint64_t>> &neighbors, const mgp::Node &node) {
    uint64_t node_id = node.Id().AsUint();
    const auto &result_it = neighbors.find(node_id);
    if (result_it == neighbors.end()) {
        std::set<uint64_t> ns;
        std::for_each(node.OutRelationships().begin(), node.OutRelationships().end(), [&ns](const auto &nn) {
            ns.insert(nn.To().Id().AsUint());
        });
        neighbors[node_id] = ns;
        return ns;
    } else {
        return result_it->second;
    }
}


/*
Calculates similiraty between pairs of nodes given by src_nodes and dst_nodes.
*/
std::vector<std::tuple<mgp::Node, mgp::Node, double>> CalculateSimilarityPairwise(const mgp::List &src_nodes, const mgp::List &dst_nodes, node_similarity_util::Similarity similarity_mode, std::string property = "") {
    if (src_nodes.Size() != dst_nodes.Size()) {
        std::cout << "In runtime error" << std::endl;
        throw std::exception();
        // throw mgp::ValueException("Arguments are of different size.");
    }
    std::cout << "after error" << std::endl;
    int num_nodes = src_nodes.Size();
    std::vector<std::tuple<mgp::Node, mgp::Node, double>> results;
    std::unordered_map<uint64_t, std::set<uint64_t>> neighbors;
    for (int i = 0; i < num_nodes; ++i) {
        const mgp::Node &src_node = src_nodes[i].ValueNode(), &dst_node = dst_nodes[i].ValueNode();
        double similarity = 0.0;
        if (similarity_mode == node_similarity_util::Similarity::cosine) {
            similarity = node_similarity_algs::CosineFunc(src_node, dst_node, property);
        } else {
            const auto &ns1 = GetNeighbors(neighbors, src_node);
            const auto &ns2 = GetNeighbors(neighbors, dst_node);
                switch (similarity_mode) {
                    case node_similarity_util::Similarity::jaccard:
                        similarity = node_similarity_algs::JaccardFunc(ns1, ns2);
                        break;
                    case node_similarity_util::Similarity::overlap:
                        similarity = node_similarity_algs::OverlapFunc(ns1, ns2);
                        break;
                    default:
                        break; 
                }
        }
        results.emplace_back(src_node, dst_node, similarity);   
    }
    return results;
}


/*
Calculates similarity between all pairs of nodes, in a cartesian mode.
*/
std::vector<std::tuple<mgp::Node, mgp::Node, double>> CalculateSimilarityCartesian(const mgp::Graph &graph, node_similarity_util::Similarity similarity_mode, std::string property = "") {
    std::unordered_set<std::pair<uint64_t, uint64_t>, node_similarity_util::pair_hash> visited_node_pairs;
    // Cache neighbors
    std::unordered_map<uint64_t, std::set<uint64_t>> neighbors;
    std::vector<std::tuple<mgp::Node, mgp::Node, double>> results;
    for (const auto &node1: graph.Nodes()) {
        uint64_t node1_id = node1.Id().AsUint();
        const std::set<uint64_t> &ns1 = (similarity_mode != node_similarity_util::Similarity::cosine) ? GetNeighbors(neighbors, node1) : std::set<uint64_t>();
        for (const auto &node2: graph.Nodes()) {
            uint64_t node2_id = node2.Id().AsUint();
            if (node1 == node2 || visited_node_pairs.count(std::make_pair<>(node2_id, node1_id))) {
                continue;
            }
            visited_node_pairs.emplace(node1_id, node2_id);
            double similarity = 0.0;
            if (similarity_mode == node_similarity_util::Similarity::cosine) {
                similarity = node_similarity_algs::CosineFunc(node1, node2, property);
            } else {
                const auto& ns2 = GetNeighbors(neighbors, node2);
                switch (similarity_mode) {
                    case node_similarity_util::Similarity::jaccard:                    
                        similarity = node_similarity_algs::JaccardFunc(ns1, ns2);
                        break;
                    case node_similarity_util::Similarity::overlap:
                        similarity = node_similarity_algs::OverlapFunc(ns1, ns2);
                        break;
                    default:
                        break;
                }
            }
            results.emplace_back(node1, node2, similarity);
        }
    }
    return results;
}

}  // namespace node_similarity_algs