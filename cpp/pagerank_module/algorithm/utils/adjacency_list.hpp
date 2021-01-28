#include <vector>

namespace utils {

/// @tparam T Node ID data type.
template <typename T>
class AdjacencyList {
 public:
  AdjacencyList() = default;
  explicit AdjacencyList(int64_t node_count) : list_(node_count) {}
  /// If adjacency list size is unknown at the construction time this method
  /// can be used to reserve the required space. The method will also clear
  /// underlying storage if it contains something.
  void Init(int64_t node_count) {
    list_.clear();
    list_.resize(node_count);
  }

  auto GetNodeCount() const { return list_.size(); }
  /// AdjacentPair is a pair of T. Values of T have to be >= 0 and < node_count
  /// because they represent position in the underlying std::vector.
  void AddAdjacentPair(T left_node_id, T right_node_id,
                       bool undirected = false) {
    list_[left_node_id].push_back(right_node_id);
    if (undirected) list_[right_node_id].push_back(left_node_id);
  }
  /// Be careful and don't call AddAdjacentPair while you have reference to this
  /// vector because the referenced vector could be resized (moved) which means
  /// that the reference is going to become invalid.
  ///
  /// @return A reference to std::vector of adjecent node ids.
  const std::vector<T> &GetAdjacentNodes(T node_id) const {
    return list_[node_id];
  }

 private:
  std::vector<std::vector<T>> list_;
};

}