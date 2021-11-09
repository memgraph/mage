#pragma once

#include <random>
#include <unordered_set>

namespace pagerank_online_alg {
namespace {
class PageRankData {
  ///
  ///@brief Context for storing data for dynamic pagerank
  ///
  ///
 public:
  void Init() {
    walks.clear();
    walks_counter.clear();
    walks_table.clear();
  }

  bool IsEmpty() const { return walks.empty(); }

  /// Keeping the information about walks on the graph
  std::vector<std::vector<std::uint64_t>> walks;

  // Keeping the information of walk appearance in algorithm for faster calculation
  std::unordered_map<std::uint64_t, uint64_t> walks_counter;

  /// Table that keeps the node appearance and walk ID
  std::unordered_map<std::uint64_t, std::unordered_set<std::uint64_t>> walks_table;
};

extern PageRankData context;
extern std::uint64_t global_R;
extern double global_epsilon;

///
///@brief Function for vector normalization
///
///@param rank Vector that needs to be normalized
///
void NormalizeRank(std::vector<double> &rank);

///
///@brief Calculates pagerank based on current information stored in global context
///
///@return std::vector<std::pair<std::uint64_t, double>>
///
std::vector<std::pair<std::uint64_t, double>> CalculatePageRank();

///
///@brief Creates a route starting from start_id, stores it in walk and updates the walk_index. Route is created via
/// random walk depending on random number genrator.
///
///@param graph Graph for route creation
///@param start_id Starting node in graph creation
///@param walk Walk vector that stores a route
///@param walk_index Index of a walk for context storing
///@param epsilon Probability of stopping the route creation
///
void CreateRoute(const mg_graph::GraphView<> &graph, std::uint64_t start_id, std::vector<std::uint64_t> &walk,
                 std::uint64_t walk_index, double epsilon);

///
///@brief Updates the context based on adding the new vertex. This means adding it to a context tables and creating
/// walks from it.
///
///@param graph Graph for updating
///@param new_vertex New vertex
///
void UpdateCreate(const mg_graph::GraphView<> &graph, std::uint64_t new_vertex);

///
///@brief Updates the context based on new edge addition. Reverts previous walks made from starting node and updates
/// them.
///
///@param graph Graph for updating
///@param new_edge New edge
///
void UpdateCreate(const mg_graph::GraphView<> &graph, const std::pair<std::uint64_t, std::uint64_t> &new_edge);

///
///@brief Removes the edge from the context and updates walks. This method works by updating walks that contain starting
/// node because they no longer exist.
///
///@param graph Graph for updating
///@param removed_edge Deleted edge
///
void UpdateDelete(const mg_graph::GraphView<> &graph, const std::pair<std::uint64_t, std::uint64_t> &removed_edge);

///
///@brief Deletes vertex from context. This is trivial because we are sure that no edge exists around that node.
///
///@param graph Graph for updating
///@param removed_vertex Removed vertex
///
void UpdateDelete(const mg_graph::GraphView<> &graph, std::uint64_t removed_vertex);

}  // namespace

///
///@brief Recreates context and calculates Pagerank based on method developed by Bahmani et. al.
///[http://snap.stanford.edu/class/cs224w-readings/bahmani10pagerank.pdf]. It creates R random walks from each node in
/// the graph and calculate pagerank approximation, depending in how many walks does a certain node appears in.
///
///@param graph Graph to calculate pagerank on
///@param R Number of random walks per node
///@param epsilon Stopping epsilong, walks of size (1/epsilon)
///@return std::vector<std::pair<std::uint64_t, double>>
///
std::vector<std::pair<std::uint64_t, double>> SetPagerank(const mg_graph::GraphView<> &graph, std::uint64_t R = 10,
                                                          double epsilon = 0.2);

///
///@brief Method for getting the values from the current Pagerank context. However if context is not set, method throws
/// an exception
///
///@param graph Graph to check consistency for
///@return std::vector<std::pair<std::uint64_t, double>>
///
std::vector<std::pair<std::uint64_t, double>> GetPagerank(const mg_graph::GraphView<> &graph);

///
///@brief Function called on vertex/edge creation or deletion. This method works with already changed graph. It
/// sequentially updates the pagerank by first updating deleted edges, deleted nodes and then adds both nodes and edges
/// among them
///
///@param graph Graph in the (t+1) step
///@param new_vertices Vertices created from the next step
///@param new_edges Edges created from the last step
///@param deleted_vertices Vertices deleted from the last iteration
///@param deleted_edges Edges deleted from the last iteration
///@return std::vector<std::pair<std::uint64_t, double>>
///
std::vector<std::pair<std::uint64_t, double>> UpdatePagerank(
    const mg_graph::GraphView<> &graph, const std::vector<std::uint64_t> &new_vertices,
    const std::vector<std::pair<std::uint64_t, uint64_t>> &new_edges,
    const std::vector<std::uint64_t> &deleted_vertices,
    const std::vector<std::pair<std::uint64_t, uint64_t>> &deleted_edges);

///
///@brief Method for resetting the context and initializing structures
///
///
void Reset();

}  // namespace pagerank_online_alg
