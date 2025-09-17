#include <mgp.hpp>

#include <fmt/format.h>
#include <omp.h>
#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <numeric>
#include <random>
#include <string_view>
#include <vector>

namespace knn_util {

// Similarity functions supported by KNN
enum class SimilarityFunction { COSINE, EUCLIDEAN, PEARSON, OVERLAP, JACCARD, DEFAULT };

// Initial sampler types
constexpr std::string_view kSamplerUniform = "uniform";
constexpr std::string_view kSamplerRandomWalk = "randomWalk";

// Property configuration for KNN
struct PropertyConfig {
  std::string name;
  SimilarityFunction metric;

  PropertyConfig(const std::string &prop_name, SimilarityFunction sim_func) : name(prop_name), metric(sim_func) {}
};

// Configuration for KNN algorithm
struct KNNConfig {
  int top_k = 1;
  double similarity_cutoff = 0.0;
  double delta_threshold = 0.001;
  int max_iterations = 100;
  int random_seed = 42;
  double sample_rate = 0.5;
  int concurrency = 1;
  std::string initial_sampler = "uniform";
  std::vector<PropertyConfig> node_properties;
  SimilarityFunction default_similarity_function = SimilarityFunction::COSINE;
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


// Extract property values from a node
std::vector<double> ExtractPropertyValues(const mgp::Node &node, const std::vector<std::string> &properties) {
  std::vector<double> values;
  values.reserve(properties.size());

  for (const auto &property : properties) {
    try {
      auto prop_value = node.GetProperty(property);
      if (prop_value.IsNull()) {
        throw mgp::ValueException(fmt::format("Node missing property: {}", property));
      }

      if (prop_value.IsList()) {
        // For vector properties, take the first element or average
        auto list = prop_value.ValueList();
        if (list.Size() > 0) {
          values.push_back(list[0].ValueDouble());
        } else {
          values.push_back(0.0);
        }
      } else if (prop_value.IsDouble()) {
        values.push_back(prop_value.ValueDouble());
      } else if (prop_value.IsInt()) {
        values.push_back(static_cast<double>(prop_value.ValueInt()));
      } else {
        throw mgp::ValueException(fmt::format("Property {} must be numeric or list of numbers", property));
      }
    } catch (const mgp::ValueException &e) {
      throw mgp::ValueException(fmt::format("Error extracting property {} : {}", property, e.what()));
    }
  }

  return values;
}

inline double CosineSimilarity(const std::vector<double> &vec1, const std::vector<double> &vec2, double norm1, double norm2) {
  const double dot =
      std::transform_reduce(vec1.begin(), vec1.end(), vec2.begin(), 0.0, std::plus<>(), std::multiplies<>());

  const double denom = norm1 * norm2;
  if (denom < 1e-9) return 0.0;
  return dot / denom;
}

// Euclidean similarity (1 / (1 + distance))
double EuclideanSimilarity(const std::vector<double> &vec1, const std::vector<double> &vec2) {
  double sum_squared_diff = 0.0;
  for (size_t i = 0; i < vec1.size(); ++i) {
    double diff = vec1[i] - vec2[i];
    sum_squared_diff += diff * diff;
  }

  double distance = std::sqrt(sum_squared_diff);
  return 1.0 / (1.0 + distance);
}

// Pearson correlation coefficient
double PearsonSimilarity(const std::vector<double> &vec1, const std::vector<double> &vec2) {
  if (vec1.size() < 2) {
    return 1.0;  // Perfect correlation for single values
  }

  // Calculate means
  double mean1 = 0.0, mean2 = 0.0;
  for (size_t i = 0; i < vec1.size(); ++i) {
    mean1 += vec1[i];
    mean2 += vec2[i];
  }
  mean1 /= vec1.size();
  mean2 /= vec2.size();

  // Calculate correlation
  double numerator = 0.0;
  double sum_sq1 = 0.0;
  double sum_sq2 = 0.0;

  for (size_t i = 0; i < vec1.size(); ++i) {
    double diff1 = vec1[i] - mean1;
    double diff2 = vec2[i] - mean2;
    numerator += diff1 * diff2;
    sum_sq1 += diff1 * diff1;
    sum_sq2 += diff2 * diff2;
  }

  double denominator = std::sqrt(sum_sq1 * sum_sq2);
  if (denominator < 1e-9) {
    return 0.0;
  }

  return numerator / denominator;
}

// Overlap similarity (intersection / min size)
double OverlapSimilarity(const std::vector<double> &vec1, const std::vector<double> &vec2) {
  // For numeric vectors, we consider values as "overlapping" if they're close
  const double threshold = 1e-6;
  int overlap_count = 0;

  for (size_t i = 0; i < vec1.size(); ++i) {
    if (std::abs(vec1[i] - vec2[i]) < threshold) {
      overlap_count++;
    }
  }

  int min_size = std::min(vec1.size(), vec2.size());
  if (min_size == 0) {
    return 0.0;
  }

  return static_cast<double>(overlap_count) / min_size;
}

// Jaccard similarity (intersection / union)
double JaccardSimilarity(const std::vector<double> &vec1, const std::vector<double> &vec2) {
  // For binary vectors (0 or 1), Jaccard = intersection / union
  int intersection_count = 0;
  int union_count = 0;

  for (size_t i = 0; i < vec1.size(); ++i) {
    bool has_v1 = vec1[i] > 0;
    bool has_v2 = vec2[i] > 0;

    if (has_v1 && has_v2) {
      intersection_count++;
    }
    if (has_v1 || has_v2) {
      union_count++;
    }
  }

  if (union_count == 0) {
    return 0.0;
  }

  return static_cast<double>(intersection_count) / union_count;
}

// Helper function to determine if a list contains integers
bool IsIntegerList(const mgp::List &list) {
  for (size_t i = 0; i < list.Size(); ++i) {
    if (list[i].IsNumeric()) {
      double val = list[i].ValueNumeric();
      // Check if the numeric value is actually an integer
      if (val != std::floor(val)) {
        return false;
      }
    } else {
      return false;
    }
  }
  return true;
}

// Helper function to get default similarity function based on data type
knn_util::SimilarityFunction GetDefaultSimilarityFunction(const mgp::Value &prop_value) {
  if (prop_value.IsNumeric()) {
    // Single numeric value - use cosine as default
    return knn_util::SimilarityFunction::COSINE;
  } else if (prop_value.IsList()) {
    mgp::List list = prop_value.ValueList();
    if (IsIntegerList(list)) {
      // List of integers - use Jaccard as default
      return knn_util::SimilarityFunction::JACCARD;
    } else {
      // List of floats - use Cosine as default
      return knn_util::SimilarityFunction::COSINE;
    }
  } else {
    // Non-numeric property - use Cosine as fallback
    return knn_util::SimilarityFunction::COSINE;
  }
}

// Structure to hold pre-loaded node data for efficient comparison
struct NodeData {
  mgp::Node node;
  std::vector<std::vector<double>> property_values;  // One vector per property
  std::vector<double> norms;
  std::vector<knn_util::SimilarityFunction> resolved_metrics;  // Resolved metrics per property

  NodeData(const mgp::Node &n, size_t num_properties)
      : node(n), property_values(num_properties), resolved_metrics(num_properties) {}
};

// Pre-load node properties into memory for efficient comparison
std::vector<NodeData> PreloadNodeData(const std::vector<mgp::Node> &nodes, const knn_util::KNNConfig &config) {
  std::vector<NodeData> node_data;
  node_data.reserve(nodes.size());

  if (config.node_properties.empty()) {
    throw mgp::ValueException("No node properties configured for similarity calculation");
  }

  for (const auto &node : nodes) {
    NodeData node_info(node, config.node_properties.size());

    // Load each property - throw immediately on any error
    for (size_t prop_idx = 0; prop_idx < config.node_properties.size(); ++prop_idx) {
      const auto &prop_config = config.node_properties[prop_idx];

      mgp::Value prop_value = node.GetProperty(prop_config.name);
      std::vector<double> values;

      // Resolve DEFAULT metric based on data type
      knn_util::SimilarityFunction resolved_metric = prop_config.metric;
      if (resolved_metric == knn_util::SimilarityFunction::DEFAULT) {
        if (prop_value.IsNumeric()) {
          // Single value - will use scalar formula
          resolved_metric = knn_util::SimilarityFunction::DEFAULT;  // Keep as DEFAULT for scalar
        } else if (prop_value.IsList()) {
          mgp::List list = prop_value.ValueList();
          if (list.Size() > 0 && list[0].IsNumeric()) {
            // Infer based on first element type
            resolved_metric =
                list[0].IsInt() ? knn_util::SimilarityFunction::JACCARD : knn_util::SimilarityFunction::COSINE;
          }
        }
      }

      // Determine expected data type based on resolved metric
      bool expects_integers = (resolved_metric == knn_util::SimilarityFunction::JACCARD ||
                               resolved_metric == knn_util::SimilarityFunction::OVERLAP);

      if (prop_value.IsNumeric()) {
        // For scalar numbers, validate type and store the single value
        if (expects_integers && !prop_value.IsInt()) {
          throw mgp::ValueException(
              fmt::format("Property {} must be integer for {} metric", prop_config.name,
                          (resolved_metric == knn_util::SimilarityFunction::JACCARD) ? "JACCARD" : "OVERLAP"));
        }
        if (!expects_integers && !prop_value.IsDouble()) {
          throw mgp::ValueException(fmt::format("Property {} must be double for {} metric", prop_config.name,
                                                (resolved_metric == knn_util::SimilarityFunction::COSINE) ? "COSINE"
                                                : (resolved_metric == knn_util::SimilarityFunction::EUCLIDEAN)
                                                    ? "EUCLIDEAN"
                                                    : "PEARSON"));
        }
        values.push_back(prop_value.ValueNumeric());
      } else if (prop_value.IsList()) {
        // For lists, validate type of first element and extract all numeric values
        mgp::List list = prop_value.ValueList();
        if (list.Size() > 0 && list[0].IsNumeric()) {
          // Check type of first element only
          if (expects_integers && !list[0].IsInt()) {
            throw mgp::ValueException(
                fmt::format("Property {} list elements must be integers for {} metric", prop_config.name,
                            (resolved_metric == knn_util::SimilarityFunction::JACCARD) ? "JACCARD" : "OVERLAP"));
          }
          if (!expects_integers && !list[0].IsDouble()) {
            throw mgp::ValueException(
                fmt::format("Property {} list elements must be doubles for {} metric", prop_config.name,
                            (resolved_metric == knn_util::SimilarityFunction::COSINE)      ? "COSINE"
                            : (resolved_metric == knn_util::SimilarityFunction::EUCLIDEAN) ? "EUCLIDEAN"
                                                                                           : "PEARSON"));
          }
        }

        // Extract all numeric values (trusting the rest are the same type)
        for (size_t i = 0; i < list.Size(); ++i) {
          if (list[i].IsNumeric()) {
            values.push_back(list[i].ValueNumeric());
          }
        }
      } else {
        throw mgp::ValueException(
            fmt::format("Property {} must be numeric or list of numbers for similarity calculation", prop_config.name));
      }

      if (values.empty()) {
        throw mgp::ValueException(
            fmt::format("Invalid property values: empty lists for property {}", prop_config.name));
      }

      node_info.property_values[prop_idx] = values;
      node_info.resolved_metrics[prop_idx] = resolved_metric;
    }

    node_data.push_back(node_info);
  }

  for (size_t i = 1; i < node_data.size(); i++) {
    for (size_t j = 0; j < node_data[i].property_values.size(); j++) {
      if (node_data[i].property_values[j].size() != node_data[0].property_values[j].size()) {
        throw mgp::ValueException("Vectors must have the same size for similarity calculation");
      }
    }
  }

  return node_data;
}

void PreloadNorms(std::vector<NodeData> &node_data, const knn_util::KNNConfig &config) {
  #pragma omp parallel for
  for (size_t ni = 0; ni < node_data.size(); ++ni) {
    auto &node = node_data[ni];
    node.norms.resize(node.property_values.size(), 0.0);
    for (size_t i = 0; i < node.property_values.size(); ++i) {
      if (node.resolved_metrics[i] == knn_util::SimilarityFunction::COSINE) {
        const auto &v = node.property_values[i];
        node.norms[i] = std::sqrt(std::inner_product(v.begin(), v.end(), v.begin(), 0.0));
      }
    }
  }
}

// Calculate similarity between pre-loaded node data
double CalculateNodeSimilarity(const NodeData &node1_data, const NodeData &node2_data,
                               const knn_util::KNNConfig &config) {
  double total_similarity = 0.0;
  const size_t num_properties = config.node_properties.size();

  // Calculate similarity for each property and compute the mean
  for (size_t prop_idx = 0; prop_idx < num_properties; ++prop_idx) {
    const auto &values1 = node1_data.property_values[prop_idx];
    const auto &values2 = node2_data.property_values[prop_idx];

    double property_similarity = 0.0;

    // Use the pre-resolved metric from NodeData
    knn_util::SimilarityFunction metric = node1_data.resolved_metrics[prop_idx];

    // For scalar numbers, use the formula: 1 / (1 + |a - b|)
    if (values1.size() == 1) {
      property_similarity = 1.0 / (1.0 + std::abs(values1[0] - values2[0]));
    } else {
      // For vectors, use the pre-resolved similarity function
      switch (metric) {
        case knn_util::SimilarityFunction::COSINE:
          property_similarity =
              CosineSimilarity(values1, values2, node1_data.norms[prop_idx], node2_data.norms[prop_idx]);
          break;
        case knn_util::SimilarityFunction::EUCLIDEAN:
          property_similarity = EuclideanSimilarity(values1, values2);
          break;
        case knn_util::SimilarityFunction::PEARSON:
          property_similarity = PearsonSimilarity(values1, values2);
          break;
        case knn_util::SimilarityFunction::OVERLAP:
          property_similarity = OverlapSimilarity(values1, values2);
          break;
        case knn_util::SimilarityFunction::JACCARD:
          property_similarity = JaccardSimilarity(values1, values2);
          break;
        default:
          property_similarity =
              CosineSimilarity(values1, values2, node1_data.norms[prop_idx], node2_data.norms[prop_idx]);
          break;
      }
    }

    total_similarity += property_similarity;
  }

  // Return the mean of all property similarities
  return total_similarity / num_properties;
}

// Validate configuration parameters
void ValidateConfig(const knn_util::KNNConfig &config) {
  if (config.initial_sampler == knn_util::kSamplerRandomWalk) {
    throw mgp::ValueException("Random walk sampling not implemented");
  } else if (config.initial_sampler != knn_util::kSamplerUniform) {
    throw mgp::ValueException(fmt::format("Unknown initial sampler: {}", config.initial_sampler));
  }
}

// Get candidate indices for comparison, excluding self
std::vector<size_t> GetCandidateIndices(size_t node_idx, size_t total_nodes, const knn_util::KNNConfig &config) {
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
    size_t sample_size = static_cast<size_t>(all_indices.size() * config.sample_rate);
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
std::vector<knn_util::KNNResult> CalculateSimilarityForNode(size_t node_idx, const std::vector<NodeData> &node_data,
                                                            const std::vector<size_t> &comparison_indices,
                                                            const knn_util::KNNConfig &config) {
  const auto &node1_data = node_data[node_idx];

  // Pre-allocate results vector
  std::vector<knn_util::KNNResult> results;
  results.reserve(comparison_indices.size());

  // Convert comparison_indices to array for OpenMP (similar to betweenness_centrality_online.cpp)
  auto array_size = comparison_indices.size();
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
    size_t idx = comparison_indices_array[i];
    const auto &node2_data = node_data[idx];
    
    // Calculate similarity directly
    double similarity = CalculateNodeSimilarity(node1_data, node2_data, config);

    // Store result
    parallel_results[i] = knn_util::KNNResult(node1_data.node.Id(), node2_data.node.Id(), similarity);
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
      auto node1 = graph.GetNodeById(result.node1_id);
      auto node2 = graph.GetNodeById(result.node2_id);
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

  // 1. Validate configuration
  ValidateConfig(config);

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
    std::vector<size_t> comparison_indices = GetCandidateIndices(i, node_data.size(), config);

    // 2. Calculate similarity for one node
    std::vector<knn_util::KNNResult> top_k_results =
        CalculateSimilarityForNode(i, node_data, comparison_indices, config);

    // 3. Insert sorted top-k results
    InsertTopKResults(top_k_results, graph, results);
  }

  return results;
}

}  // namespace knn_algs
