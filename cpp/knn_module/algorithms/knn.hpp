#include <mgp.hpp>

#include <fmt/format.h>
#include <omp.h>
#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <numeric>
#include <random>
#include <vector>

namespace knn_util {

// Aggregate methods for merging property values
enum class AggregateMethod { NONE, APPEND, MIN, MAX, AVG, SUM };

// Configuration for KNN algorithm
struct KNNConfig {
  int top_k = 1;
  double similarity_cutoff = 0.0;
  double delta_threshold = 0.001;
  int max_iterations = 100;
  int random_seed = 42;
  double sample_rate = 0.5;
  int concurrency = 1;
  std::vector<std::string> node_properties;
  AggregateMethod aggregate_method = AggregateMethod::NONE;
};

// Result structure for KNN
struct KNNResult {
  mgp::Id node1_id;
  mgp::Id node2_id;
  double similarity;

  // Default constructor for std::vector compatibility
  KNNResult() : similarity(0.0) {
    // Initialize with default constructed Ids
    node1_id = mgp::Id();
    node2_id = mgp::Id();
  }

  KNNResult(const mgp::Node &n1, const mgp::Node &n2, double sim)
      : node1_id(n1.Id()), node2_id(n2.Id()), similarity(sim) {}

  KNNResult(mgp::Id id1, mgp::Id id2, double sim) : node1_id(id1), node2_id(id2), similarity(sim) {}
};

}  // namespace knn_util

namespace knn_algs {

inline double CosineSimilarity(const std::vector<double> &vec1, const std::vector<double> &vec2, const double norm1,
                               const double norm2) {
  const double dot =
      std::transform_reduce(vec1.begin(), vec1.end(), vec2.begin(), 0.0, std::plus<>(), std::multiplies<>());

  const double denom = norm1 * norm2;
  if (denom < 1e-9) return 0.0;
  return dot / denom;
}

// Helper function to aggregate property values based on the specified method
std::vector<double> AggregatePropertyValues(const std::vector<std::vector<double>> &property_vectors,
                                            knn_util::AggregateMethod method) {
  if (property_vectors.empty()) {
    throw mgp::ValueException("Properties of the nodes for aggregation cannot be empty!");
  }

  // Validate vector sizes for methods that require same-sized vectors
  bool requires_same_size = (method == knn_util::AggregateMethod::MIN || method == knn_util::AggregateMethod::MAX ||
                             method == knn_util::AggregateMethod::AVG || method == knn_util::AggregateMethod::SUM);

  if (requires_same_size) {
    size_t expected_size = property_vectors[0].size();
    for (size_t i = 1; i < property_vectors.size(); ++i) {
      if (property_vectors[i].size() != expected_size) {
        throw mgp::ValueException(fmt::format(
            "All property vectors must have the same size for aggregation. Expected size: {}, but vector has size: {}",
            expected_size, property_vectors[i].size()));
      }
    }
  }

  std::vector<double> result;

  switch (method) {
    case knn_util::AggregateMethod::NONE: {
      throw mgp::ValueException(
          "Unexpected error: aggregation of property values happened without aggregation method specified! Please "
          "contact Memgraph support.");
    }
    case knn_util::AggregateMethod::APPEND: {
      // Concatenate all property vectors
      for (const auto &vec : property_vectors) {
        result.insert(result.end(), vec.begin(), vec.end());
      }
      break;
    }
    case knn_util::AggregateMethod::MIN: {
      // Take minimum value from each position across all vectors
      size_t vector_size = property_vectors[0].size();
      result.resize(vector_size);
      for (size_t i = 0; i < vector_size; ++i) {
        result[i] = property_vectors[0][i];
        for (size_t j = 1; j < property_vectors.size(); ++j) {
          result[i] = std::min(result[i], property_vectors[j][i]);
        }
      }
      break;
    }
    case knn_util::AggregateMethod::MAX: {
      // Take maximum value from each position across all vectors
      size_t vector_size = property_vectors[0].size();
      result.resize(vector_size);
      for (size_t i = 0; i < vector_size; ++i) {
        result[i] = property_vectors[0][i];
        for (size_t j = 1; j < property_vectors.size(); ++j) {
          result[i] = std::max(result[i], property_vectors[j][i]);
        }
      }
      break;
    }
    case knn_util::AggregateMethod::AVG: {
      // Take average value from each position across all vectors
      size_t vector_size = property_vectors[0].size();
      result.resize(vector_size);
      for (size_t i = 0; i < vector_size; ++i) {
        double sum = 0.0;
        for (const auto &vec : property_vectors) {
          sum += vec[i];
        }
        result[i] = sum / property_vectors.size();
      }
      break;
    }
    case knn_util::AggregateMethod::SUM: {
      // Take sum value from each position across all vectors
      size_t vector_size = property_vectors[0].size();
      result.resize(vector_size);
      for (size_t i = 0; i < vector_size; ++i) {
        result[i] = 0.0;
        for (const auto &vec : property_vectors) {
          result[i] += vec[i];
        }
      }
      break;
    }
  }

  return result;
}

// Structure to hold pre-loaded node data for efficient comparison
struct NodeData {
  mgp::Id node_id;
  std::vector<std::vector<double>>
      property_values;        // One vector per property (for NONE) or single aggregated vector (for aggregation)
  std::vector<double> norms;  // Norms for each property

  NodeData(const mgp::Node &n, const std::vector<std::vector<double>> &prop_values)
      : node_id(n.Id()), property_values(prop_values) {}
};

// Pre-load node properties into memory for efficient comparison
std::vector<NodeData> PreloadNodeData(const std::vector<mgp::Node> &nodes, const knn_util::KNNConfig &config) {
  std::vector<NodeData> node_data;
  node_data.reserve(nodes.size());

  for (const auto &node : nodes) {
    // Collect all property values first
    std::vector<std::vector<double>> property_values(config.node_properties.size());

    // Load all properties into temporary vectors
    for (size_t prop_idx = 0; prop_idx < config.node_properties.size(); ++prop_idx) {
      const std::string &prop_name = config.node_properties[prop_idx];
      mgp::Value prop_value = node.GetProperty(prop_name);
      std::vector<double> values;

      if (!prop_value.IsList()) {
        throw mgp::ValueException(
            fmt::format("Property {} must be a list of doubles for similarity calculation", prop_name));
      }

      mgp::List list = prop_value.ValueList();
      for (size_t i = 0; i < list.Size(); ++i) {
        if (!list[i].IsDouble()) {
          throw mgp::ValueException(
              fmt::format("Property {} must be a list of doubles for similarity calculation", prop_name));
        }
        values.push_back(list[i].ValueDouble());
      }

      if (values.empty()) {
        throw mgp::ValueException(fmt::format("Invalid property values: empty lists for property {}", prop_name));
      }

      property_values[prop_idx] = values;
    }

    // Handle aggregation if needed
    const bool is_aggregated = (config.aggregate_method != knn_util::AggregateMethod::NONE);
    if (is_aggregated) {
      // Aggregate the property values
      std::vector<double> aggregated = AggregatePropertyValues(property_values, config.aggregate_method);
      property_values.clear();
      property_values.push_back(aggregated);
    }

    // Create node_info at the end with the final property_values
    NodeData node_info(node, property_values);
    node_data.push_back(node_info);
  }

  // Validate vector sizes
  if (node_data.size() > 1) {
    // Validate that all property vectors have the same size
    for (size_t prop_idx = 0; prop_idx < node_data[0].property_values.size(); ++prop_idx) {
      size_t expected_size = node_data[0].property_values[prop_idx].size();
      for (size_t i = 1; i < node_data.size(); ++i) {
        if (node_data[i].property_values[prop_idx].size() != expected_size) {
          throw mgp::ValueException("Property vectors must have the same size for similarity calculation");
        }
      }
    }
  }

  return node_data;
}

void PreloadNorms(std::vector<NodeData> &node_data, const knn_util::KNNConfig &config) {
#pragma omp parallel for
  for (size_t ni = 0; ni < node_data.size(); ++ni) {
    auto &node = node_data[ni];

    // Calculate norms for each property vector
    node.norms.resize(node.property_values.size(), 0.0);
    for (size_t i = 0; i < node.property_values.size(); ++i) {
      const auto &v = node.property_values[i];
      node.norms[i] = std::sqrt(std::inner_product(v.begin(), v.end(), v.begin(), 0.0));
    }
  }
}

// Calculate similarity between pre-loaded node data
double CalculateNodeSimilarity(const NodeData &node1_data, const NodeData &node2_data,
                               const knn_util::KNNConfig &config) {
  double total_similarity = 0.0;
  const size_t num_properties = node1_data.property_values.size();

  for (size_t prop_idx = 0; prop_idx < num_properties; ++prop_idx) {
    const auto &values1 = node1_data.property_values[prop_idx];
    const auto &values2 = node2_data.property_values[prop_idx];

    // Use cosine similarity for each property
    double property_similarity =
        CosineSimilarity(values1, values2, node1_data.norms[prop_idx], node2_data.norms[prop_idx]);

    total_similarity += property_similarity;
  }

  // Return the mean of all property similarities
  return total_similarity / num_properties;
}

// Get candidate indices for comparison, excluding self
std::vector<size_t> GetCandidateIndices(const size_t node_idx, const size_t total_nodes,
                                        const knn_util::KNNConfig &config) {
  std::vector<size_t> comparison_indices;

  if (config.sample_rate < 1.0) {
    // Create indices for all nodes except self
    std::vector<size_t> all_indices;
    all_indices.reserve(total_nodes - 1);
    for (size_t i = 0; i < total_nodes; ++i) {
      if (i != node_idx) {  // Skip self
        all_indices.push_back(i);
      }
    }

    // Shuffle indices for uniform sampling
    std::mt19937 rng(config.random_seed);
    std::shuffle(all_indices.begin(), all_indices.end(), rng);

    // Calculate sample size
    const size_t sample_size = static_cast<size_t>(all_indices.size() * config.sample_rate);
    comparison_indices.reserve(sample_size);

    // Take the first sample_size indices
    for (size_t i = 0; i < sample_size; ++i) {
      comparison_indices.push_back(all_indices[i]);
    }
  } else {
    // Compare against all other nodes
    comparison_indices.reserve(total_nodes - 1);
    for (size_t j = 0; j < total_nodes; ++j) {
      if (j != node_idx) {  // Skip self-comparison
        comparison_indices.push_back(j);
      }
    }
  }

  return comparison_indices;
}

// Calculate similarity for one node against all candidates (parallel implementation)
std::vector<knn_util::KNNResult> CalculateSimilarityForNode(const size_t node_idx,
                                                            const std::vector<NodeData> &node_data,
                                                            const std::vector<size_t> &comparison_indices,
                                                            const knn_util::KNNConfig &config) {
  const auto &node1_data = node_data[node_idx];

  // Pre-allocate results vector
  std::vector<knn_util::KNNResult> results;
  results.reserve(comparison_indices.size());

  // Convert comparison_indices to array for OpenMP (similar to betweenness_centrality_online.cpp)
  const auto array_size = comparison_indices.size();
  std::vector<size_t> comparison_indices_array(array_size);
  std::copy(comparison_indices.begin(), comparison_indices.end(), comparison_indices_array.begin());

  // Pre-allocate parallel results vector
  std::vector<knn_util::KNNResult> parallel_results(array_size);

  // Set OpenMP parameters
  omp_set_dynamic(0);
  omp_set_num_threads(config.concurrency);

  // Parallel similarity calculation using OpenMP
#pragma omp parallel for
  for (size_t i = 0; i < array_size; ++i) {
    const size_t idx = comparison_indices_array[i];
    const auto &node2_data = node_data[idx];

    // Calculate similarity directly
    const double similarity = CalculateNodeSimilarity(node1_data, node2_data, config);

    // Store result
    parallel_results[i] = knn_util::KNNResult(node1_data.node_id, node2_data.node_id, similarity);
  }

  // Filter results based on similarity cutoff and add to final results
  for (const auto &result : parallel_results) {
    if (result.similarity >= config.similarity_cutoff) {
      results.push_back(result);
    }
  }

  const size_t k = std::min(results.size(), static_cast<size_t>(config.top_k));
  auto cmp = [](const knn_util::KNNResult &a, const knn_util::KNNResult &b) {
    return a.similarity > b.similarity;  // descending
  };

  if (k > 0 && results.size() > k) {
    std::nth_element(results.begin(), results.begin() + k, results.end(), cmp);
    results.resize(k);
    std::sort(results.begin(), results.end(), cmp);  // sort only top-k
  } else {
    std::sort(results.begin(), results.end(), cmp);  // small n or k >= n
  }

  return results;
}

// Insert top-k results into final results
void InsertTopKResults(const std::vector<knn_util::KNNResult> &top_k_results, const mgp::Graph &graph,
                       std::vector<std::tuple<mgp::Node, mgp::Node, double>> &final_results) {
  // Convert to final results with actual nodes (results are already sorted)
  for (const auto &result : top_k_results) {
    try {
      const auto node1 = graph.GetNodeById(result.node1_id);
      const auto node2 = graph.GetNodeById(result.node2_id);
      final_results.push_back(std::make_tuple(node1, node2, result.similarity));
    } catch (const std::exception &e) {
      // Skip if node not found
      continue;
    }
  }
}

// Main KNN algorithm implementation
std::vector<std::tuple<mgp::Node, mgp::Node, double>> CalculateKNN(const mgp::Graph &graph,
                                                                   const knn_util::KNNConfig &config) {
  std::vector<std::tuple<mgp::Node, mgp::Node, double>> results;
  std::vector<mgp::Node> nodes;

  // Collect all nodes
  for (const auto &node : graph.Nodes()) {
    nodes.push_back(node);
  }

  if (nodes.size() < 2) {
    return results;  // Need at least 2 nodes for similarity
  }

  // Pre-load node properties into memory for efficient comparison
  std::vector<NodeData> node_data = PreloadNodeData(nodes, config);
  PreloadNorms(node_data, config);

  // For each node, find its top-k most similar nodes
  for (size_t i = 0; i < node_data.size(); ++i) {
    // Get candidate indices for comparison
    const std::vector<size_t> comparison_indices = GetCandidateIndices(i, node_data.size(), config);

    // 2. Calculate similarity for one node
    const std::vector<knn_util::KNNResult> top_k_results =
        CalculateSimilarityForNode(i, node_data, comparison_indices, config);

    // 3. Insert sorted top-k results
    InsertTopKResults(top_k_results, graph, results);
  }

  return results;
}

}  // namespace knn_algs
