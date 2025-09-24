#include <fmt/format.h>
#include <mg_exceptions.hpp>
#include <mgp.hpp>
#include <random>
#include <string_view>

#include "algorithms/knn.hpp"

// Procedure names
constexpr std::string_view kProcedureGet = "get";

// Argument names
constexpr std::string_view kArgumentConfig = "config";
constexpr std::string_view kConfigNodeProperties = "nodeProperties";
constexpr std::string_view kConfigTopK = "topK";
constexpr std::string_view kConfigSimilarityCutoff = "similarityCutoff";
constexpr std::string_view kConfigDeltaThreshold = "deltaThreshold";
constexpr std::string_view kConfigMaxIterations = "maxIterations";
constexpr std::string_view kConfigRandomSeed = "randomSeed";
constexpr std::string_view kConfigSampleRate = "sampleRate";
constexpr std::string_view kConfigConcurrency = "concurrency";
constexpr std::string_view kConfigAggregateMethod = "aggregateMethod";

// Return field names
constexpr std::string_view kFieldNode = "node";
constexpr std::string_view kFieldNeighbour = "neighbour";
constexpr std::string_view kFieldSimilarity = "similarity";

// Default parameter values
constexpr int kDefaultTopK = 1;
constexpr double kDefaultSimilarityCutoff = 0.0;
constexpr double kDefaultDeltaThreshold = 0.001;
constexpr int kDefaultMaxIterations = 100;
constexpr int kDefaultConcurrency = 1;
constexpr double kDefaultSampleRate = 0.5;

// Aggregate method values
constexpr std::string_view kAggregateNone = "NONE";
constexpr std::string_view kAggregateAppend = "APPEND";
constexpr std::string_view kAggregateMin = "MIN";
constexpr std::string_view kAggregateMax = "MAX";
constexpr std::string_view kAggregateAvg = "AVG";
constexpr std::string_view kAggregateSum = "SUM";

// Helper function to validate if a string is a valid aggregate method
bool IsValidAggregateMethod(const std::string &method_str) {
  return method_str == kAggregateNone || method_str == kAggregateAppend || method_str == kAggregateMin ||
         method_str == kAggregateMax || method_str == kAggregateAvg || method_str == kAggregateSum;
}

// Helper function to parse aggregate method from string
knn_util::AggregateMethod ParseAggregateMethod(const std::string &method_str) {
  if (method_str == kAggregateNone) {
    return knn_util::AggregateMethod::NONE;
  } else if (method_str == kAggregateAppend) {
    return knn_util::AggregateMethod::APPEND;
  } else if (method_str == kAggregateMin) {
    return knn_util::AggregateMethod::MIN;
  } else if (method_str == kAggregateMax) {
    return knn_util::AggregateMethod::MAX;
  } else if (method_str == kAggregateAvg) {
    return knn_util::AggregateMethod::AVG;
  } else if (method_str == kAggregateSum) {
    return knn_util::AggregateMethod::SUM;
  } else {
    return knn_util::AggregateMethod::NONE;  // Default fallback
  }
}

// Helper function to validate parameter ranges
void ValidateParameterRanges(const knn_util::KNNConfig &config) {
  // Validate range [0, 1] parameters
  if (config.sample_rate < 0.0 || config.sample_rate > 1.0) {
    throw mgp::ValueException(fmt::format("sampleRate must be between 0 and 1, got {}", config.sample_rate));
  }

  if (config.delta_threshold < 0.0 || config.delta_threshold > 1.0) {
    throw mgp::ValueException(fmt::format("deltaThreshold must be between 0 and 1, got {}", config.delta_threshold));
  }

  if (config.similarity_cutoff < 0.0 || config.similarity_cutoff > 1.0) {
    throw mgp::ValueException(
        fmt::format("similarityCutoff must be between 0 and 1, got {}", config.similarity_cutoff));
  }

  // Validate positive integer parameters
  if (config.top_k <= 0) {
    throw mgp::ValueException(fmt::format("topK must be a positive integer, got {}", config.top_k));
  }

  if (config.concurrency <= 0) {
    throw mgp::ValueException(fmt::format("concurrency must be a positive integer, got {}", config.concurrency));
  }

  if (config.max_iterations <= 0) {
    throw mgp::ValueException(fmt::format("maxIterations must be a positive integer, got {}", config.max_iterations));
  }

  // randomSeed can be negative, so we only check it's not zero
  if (config.random_seed == 0) {
    throw mgp::ValueException("randomSeed cannot be 0");
  }
}

// Helper function to parse nodeProperties configuration
std::vector<std::string> ParseNodeProperties(const mgp::Value &node_props_value) {
  std::vector<std::string> properties;

  if (node_props_value.IsString()) {
    // Single property name
    const std::string prop_name = std::string(node_props_value.ValueString());
    if (prop_name.empty()) {
      throw mgp::ValueException("Property name cannot be empty");
    }
    properties.push_back(prop_name);
  } else if (node_props_value.IsList()) {
    // List of property names
    mgp::List prop_list = node_props_value.ValueList();
    if (prop_list.Size() == 0) {
      throw mgp::ValueException("Property list cannot be empty");
    }

    for (size_t i = 0; i < prop_list.Size(); ++i) {
      if (prop_list[i].IsString()) {
        const std::string prop_name = std::string(prop_list[i].ValueString());
        if (prop_name.empty()) {
          throw mgp::ValueException(fmt::format("Property name at index {} cannot be empty", i));
        }
        properties.push_back(prop_name);
      } else {
        throw mgp::ValueException(fmt::format("Property list element at index {} must be a string", i));
      }
    }
  } else {
    throw mgp::ValueException(
        "nodeProperties must be a string or list of strings defining properties to be used for similarity calculation. "
        "Each property must be a list of numbers.");
  }

  if (properties.empty()) {
    throw mgp::ValueException("No valid properties found in nodeProperties configuration");
  }

  return properties;
}

// Helper function to insert results into record factory
void InsertResults(const std::vector<std::tuple<mgp::Node, mgp::Node, double>> &results,
                   const mgp::RecordFactory &record_factory) {
  for (const auto &result : results) {
    auto new_record = record_factory.NewRecord();
    new_record.Insert(kFieldNode.data(), std::get<0>(result));
    new_record.Insert(kFieldNeighbour.data(), std::get<1>(result));
    new_record.Insert(kFieldSimilarity.data(), std::get<2>(result));
  }
}

// Get procedure - returns similarity pairs
void Get(mgp_list *args, mgp_graph *memgraph_graph, mgp_result *result, mgp_memory *memory) {
  mgp::MemoryDispatcherGuard guard{memory};
  const auto record_factory = mgp::RecordFactory(result);
  const auto &arguments = mgp::List(args);
  const auto &config_map = arguments[0].ValueMap();

  try {
    knn_util::KNNConfig config;

    // Parse node properties - required parameter
    if (!config_map.KeyExists(kConfigNodeProperties)) {
      throw mgp::ValueException("Required parameter 'nodeProperties' is missing from config");
    }

    config.node_properties = ParseNodeProperties(config_map[kConfigNodeProperties]);

    // Parse other parameters with defaults
    config.top_k =
        config_map.KeyExists(kConfigTopK) ? static_cast<int>(config_map[kConfigTopK].ValueInt()) : kDefaultTopK;
    config.similarity_cutoff = config_map.KeyExists(kConfigSimilarityCutoff)
                                   ? config_map[kConfigSimilarityCutoff].ValueDouble()
                                   : kDefaultSimilarityCutoff;
    config.delta_threshold = config_map.KeyExists(kConfigDeltaThreshold)
                                 ? config_map[kConfigDeltaThreshold].ValueDouble()
                                 : kDefaultDeltaThreshold;
    config.max_iterations = config_map.KeyExists(kConfigMaxIterations)
                                ? static_cast<int>(config_map[kConfigMaxIterations].ValueInt())
                                : kDefaultMaxIterations;
    // Parse concurrency first (needed for validation)
    config.concurrency = config_map.KeyExists(kConfigConcurrency)
                             ? static_cast<int>(config_map[kConfigConcurrency].ValueInt())
                             : kDefaultConcurrency;

    // Parse random seed with validation
    if (config_map.KeyExists(kConfigRandomSeed)) {
      config.random_seed = static_cast<int>(config_map[kConfigRandomSeed].ValueInt());
      // If seed is provided, concurrency must be 1 for deterministic results
      if (config.concurrency != 1) {
        throw mgp::ValueException(
            "When 'randomSeed' is specified, 'concurrency' must be set to 1 for deterministic results");
      }
    } else {
      // Generate completely random seed
      std::random_device rd;
      config.random_seed = static_cast<int>(rd());
    }

    config.sample_rate =
        config_map.KeyExists(kConfigSampleRate) ? config_map[kConfigSampleRate].ValueDouble() : kDefaultSampleRate;

    // Parse aggregate method
    if (config_map.KeyExists(kConfigAggregateMethod)) {
      const std::string method_str = std::string(config_map[kConfigAggregateMethod].ValueString());
      if (!IsValidAggregateMethod(method_str)) {
        throw mgp::ValueException(fmt::format(
            "Invalid aggregateMethod '{}'. Valid methods are: NONE, APPEND, MIN, MAX, AVG, SUM", method_str));
      }
      if (config.node_properties.size() < 2) {
        throw mgp::ValueException("aggregateMethod can only be used when nodeProperties has at least two properties");
      }
      config.aggregate_method = ParseAggregateMethod(method_str);
    } else {
      config.aggregate_method = knn_util::AggregateMethod::NONE;  // Default
    }

    // Validate all parameter ranges
    ValidateParameterRanges(config);

    const auto results = knn_algs::CalculateKNN(mgp::Graph(memgraph_graph), config);
    InsertResults(results, record_factory);
  } catch (const mgp::ValueException &e) {
    record_factory.SetErrorMessage(e.what());
  } catch (const std::exception &e) {
    record_factory.SetErrorMessage(fmt::format("Unexpected error: {}", e.what()));
  }
}

extern "C" int mgp_init_module(struct mgp_module *module, struct mgp_memory *memory) {
  try {
    mgp::MemoryDispatcherGuard guard{memory};

    // Return types for get procedure
    std::vector<mgp::Return> returns = {mgp::Return(kFieldNode, mgp::Type::Node),
                                        mgp::Return(kFieldNeighbour, mgp::Type::Node),
                                        mgp::Return(kFieldSimilarity, mgp::Type::Double)};

    // Single config parameter
    std::vector<mgp::Parameter> parameters = {mgp::Parameter(kArgumentConfig, mgp::Type::Map)};

    // Add the single get procedure
    mgp::AddProcedure(Get, kProcedureGet, mgp::ProcedureType::Read, parameters, returns, module, memory);

  } catch (const std::exception &e) {
    return 1;
  }
  return 0;
}

extern "C" int mgp_shutdown_module() { return 0; }
