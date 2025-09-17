#include <mgp.hpp>
#include <mg_exceptions.hpp>
#include <random>
#include <algorithm>
#include <string_view>
#include <fmt/format.h>

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
constexpr std::string_view kConfigSimilarityFunction = "similarityFunction";
constexpr std::string_view kConfigInitialSampler = "initialSampler";

// Return field names
constexpr std::string_view kFieldNode = "node";
constexpr std::string_view kFieldNeighbour = "neighbour";
constexpr std::string_view kFieldSimilarity = "similarity";

// Similarity function values
constexpr std::string_view kSimilarityCosine = "COSINE";
constexpr std::string_view kSimilarityEuclidean = "EUCLIDEAN";
constexpr std::string_view kSimilarityPearson = "PEARSON";
constexpr std::string_view kSimilarityOverlap = "OVERLAP";
constexpr std::string_view kSimilarityJaccard = "JACCARD";
constexpr std::string_view kSimilarityDefault = "DEFAULT";

// Default parameter values
constexpr int kDefaultTopK = 1;
constexpr double kDefaultSimilarityCutoff = 0.0;
constexpr double kDefaultDeltaThreshold = 0.001;
constexpr int kDefaultMaxIterations = 100;
constexpr int kDefaultConcurrency = 1;
constexpr double kDefaultSampleRate = 0.5;
constexpr std::string_view kDefaultInitialSampler = "uniform";

// Initial sampler values (using constants from knn.hpp)
// constexpr std::string_view kSamplerUniform = knn_util::kSamplerUniform;
// constexpr std::string_view kSamplerRandomWalk = knn_util::kSamplerRandomWalk;

// Helper function to validate if a string is a valid similarity function
bool IsValidSimilarityFunction(const std::string& func_str) {
    return func_str == kSimilarityCosine ||
           func_str == kSimilarityEuclidean ||
           func_str == kSimilarityPearson ||
           func_str == kSimilarityOverlap ||
           func_str == kSimilarityJaccard ||
           func_str == kSimilarityDefault;
}

// Helper function to validate if a string is a valid initial sampler
bool IsValidInitialSampler(const std::string& sampler_str) {
    std::string lower_sampler = sampler_str;
    std::transform(lower_sampler.begin(), lower_sampler.end(), lower_sampler.begin(), ::tolower);
    return lower_sampler == knn_util::kSamplerUniform || lower_sampler == knn_util::kSamplerRandomWalk;
}

// Helper function to validate parameter ranges
void ValidateParameterRanges(const knn_util::KNNConfig& config) {
    // Validate range [0, 1] parameters
    if (config.sample_rate < 0.0 || config.sample_rate > 1.0) {
        throw mgp::ValueException(fmt::format("sampleRate must be between 0 and 1, got {}", config.sample_rate));
    }
    
    if (config.delta_threshold < 0.0 || config.delta_threshold > 1.0) {
        throw mgp::ValueException(fmt::format("deltaThreshold must be between 0 and 1, got {}", config.delta_threshold));
    }
    
    if (config.similarity_cutoff < 0.0 || config.similarity_cutoff > 1.0) {
        throw mgp::ValueException(fmt::format("similarityCutoff must be between 0 and 1, got {}", config.similarity_cutoff));
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

// Helper function to parse similarity function from string
knn_util::SimilarityFunction ParseSimilarityFunction(const std::string& func_str) {
    if (func_str == kSimilarityCosine) {
        return knn_util::SimilarityFunction::COSINE;
    } else if (func_str == kSimilarityEuclidean) {
        return knn_util::SimilarityFunction::EUCLIDEAN;
    } else if (func_str == kSimilarityPearson) {
        return knn_util::SimilarityFunction::PEARSON;
    } else if (func_str == kSimilarityOverlap) {
        return knn_util::SimilarityFunction::OVERLAP;
    } else if (func_str == kSimilarityJaccard) {
        return knn_util::SimilarityFunction::JACCARD;
    } else if (func_str == kSimilarityDefault) {
        return knn_util::SimilarityFunction::DEFAULT;
    } else {
        return knn_util::SimilarityFunction::COSINE; // Default fallback
    }
}

// Helper function to parse nodeProperties configuration
std::vector<knn_util::PropertyConfig> ParseNodeProperties(const mgp::Value& node_props_value) {
    std::vector<knn_util::PropertyConfig> properties;
    
    if (node_props_value.IsString()) {
        // Single property name - use default similarity function
        std::string prop_name = std::string(node_props_value.ValueString());
        if (prop_name.empty()) {
            throw mgp::ValueException("Property name cannot be empty");
        }
        properties.emplace_back(prop_name, knn_util::SimilarityFunction::DEFAULT);
    } else if (node_props_value.IsMap()) {
        // Map of property names to metrics
        mgp::Map prop_map = node_props_value.ValueMap();
        if (prop_map.Size() == 0) {
            throw mgp::ValueException("Property map cannot be empty");
        }
        
        for (const auto& entry : prop_map) {
            // Validate property name
            std::string prop_name = std::string(entry.key);
            if (prop_name.empty()) {
                throw mgp::ValueException("Property name cannot be empty");
            }
            
            // Validate metric value
            if (!entry.value.IsString()) {
                throw mgp::ValueException(fmt::format("Metric value must be a string for property '{}'", prop_name));
            }
            
            std::string metric_str = std::string(entry.value.ValueString());
            if (metric_str.empty()) {
                throw mgp::ValueException(fmt::format("Metric value cannot be empty for property '{}'", prop_name));
            }
            
            if (!IsValidSimilarityFunction(metric_str)) {
                throw mgp::ValueException(fmt::format("Invalid metric '{}' for property '{}'. Valid metrics are: COSINE, EUCLIDEAN, PEARSON, OVERLAP, JACCARD, DEFAULT", metric_str, prop_name));
            }
            
            knn_util::SimilarityFunction metric = ParseSimilarityFunction(metric_str);
            properties.emplace_back(prop_name, metric);
        }
    } else if (node_props_value.IsList()) {
        // List of strings and/or maps
        mgp::List prop_list = node_props_value.ValueList();
        if (prop_list.Size() == 0) {
            throw mgp::ValueException("Property list cannot be empty");
        }
        
        for (size_t i = 0; i < prop_list.Size(); ++i) {
            if (prop_list[i].IsString()) {
                // String property name - use default similarity function
                std::string prop_name = std::string(prop_list[i].ValueString());
                if (prop_name.empty()) {
                    throw mgp::ValueException(fmt::format("Property name at index {} cannot be empty", i));
                }
                properties.emplace_back(prop_name, knn_util::SimilarityFunction::DEFAULT);
            } else if (prop_list[i].IsMap()) {
                // Map entry
                mgp::Map prop_map = prop_list[i].ValueMap();
                if (prop_map.Size() == 0) {
                    throw mgp::ValueException(fmt::format("Property map at index {} cannot be empty", i));
                }
                
                for (const auto& entry : prop_map) {
                    // Validate property name
                    std::string prop_name = std::string(entry.key);
                    if (prop_name.empty()) {
                        throw mgp::ValueException(fmt::format("Property name cannot be empty in map at index {}", i));
                    }
                    
                    // Validate metric value
                    if (!entry.value.IsString()) {
                        throw mgp::ValueException(fmt::format("Metric value must be a string for property '{}' in map at index {}", prop_name, i));
                    }
                    
                    std::string metric_str = std::string(entry.value.ValueString());
                    if (metric_str.empty()) {
                        throw mgp::ValueException(fmt::format("Metric value cannot be empty for property '{}' in map at index {}", prop_name, i));
                    }
                    
                    if (!IsValidSimilarityFunction(metric_str)) {
                        throw mgp::ValueException(fmt::format("Invalid metric '{}' for property '{}' in map at index {}. Valid metrics are: COSINE, EUCLIDEAN, PEARSON, OVERLAP, JACCARD, DEFAULT", metric_str, prop_name, i));
                    }
                    
                    knn_util::SimilarityFunction metric = ParseSimilarityFunction(metric_str);
                    properties.emplace_back(prop_name, metric);
                }
            } else {
                throw mgp::ValueException(fmt::format("Property list element at index {} must be a string or map", i));
            }
        }
    } else {
        throw mgp::ValueException("nodeProperties must be a string, map, or list");
    }
    
    if (properties.empty()) {
        throw mgp::ValueException("No valid properties found in nodeProperties configuration");
    }
    
    return properties;
}

// Helper function to insert results into record factory
void InsertResults(const std::vector<std::tuple<mgp::Node, mgp::Node, double>>& results, const mgp::RecordFactory& record_factory) {
    for (const auto& result : results) {
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
        config.top_k = config_map.KeyExists(kConfigTopK) ? 
            static_cast<int>(config_map[kConfigTopK].ValueInt()) : kDefaultTopK;
        config.similarity_cutoff = config_map.KeyExists(kConfigSimilarityCutoff) ? 
            config_map[kConfigSimilarityCutoff].ValueDouble() : kDefaultSimilarityCutoff;
        config.delta_threshold = config_map.KeyExists(kConfigDeltaThreshold) ? 
            config_map[kConfigDeltaThreshold].ValueDouble() : kDefaultDeltaThreshold;
        config.max_iterations = config_map.KeyExists(kConfigMaxIterations) ? 
            static_cast<int>(config_map[kConfigMaxIterations].ValueInt()) : kDefaultMaxIterations;
        // Parse concurrency first (needed for validation)
        config.concurrency = config_map.KeyExists(kConfigConcurrency) ? 
            static_cast<int>(config_map[kConfigConcurrency].ValueInt()) : kDefaultConcurrency;
        
        // Parse random seed with validation
        if (config_map.KeyExists(kConfigRandomSeed)) {
            config.random_seed = static_cast<int>(config_map[kConfigRandomSeed].ValueInt());
            // If seed is provided, concurrency must be 1 for deterministic results
            if (config.concurrency != 1) {
                throw mgp::ValueException("When 'randomSeed' is specified, 'concurrency' must be set to 1 for deterministic results");
            }
        } else {
            // Generate completely random seed
            std::random_device rd;
            config.random_seed = static_cast<int>(rd());
        }
        
        config.sample_rate = config_map.KeyExists(kConfigSampleRate) ? 
            config_map[kConfigSampleRate].ValueDouble() : kDefaultSampleRate;
        
        // Parse initial sampler
        if (config_map.KeyExists(kConfigInitialSampler)) {
            std::string sampler_str = std::string(config_map[kConfigInitialSampler].ValueString());
            if (!IsValidInitialSampler(sampler_str)) {
                throw mgp::ValueException(fmt::format("Invalid initialSampler '{}'. Valid values are: uniform, randomWalk", sampler_str));
            }
            // Convert to lowercase for consistency
            std::transform(sampler_str.begin(), sampler_str.end(), sampler_str.begin(), ::tolower);
            config.initial_sampler = sampler_str;
        } else {
            config.initial_sampler = kDefaultInitialSampler;
        }
        
        // Parse default similarity function
        if (config_map.KeyExists(kConfigSimilarityFunction)) {
            std::string func_str = std::string(config_map[kConfigSimilarityFunction].ValueString());
            if (!IsValidSimilarityFunction(func_str)) {
                throw mgp::ValueException(fmt::format("Invalid similarityFunction '{}'. Valid metrics are: COSINE, EUCLIDEAN, PEARSON, OVERLAP, JACCARD, DEFAULT", func_str));
            }
            config.default_similarity_function = ParseSimilarityFunction(func_str);
        } else {
            config.default_similarity_function = knn_util::SimilarityFunction::COSINE; // Default
        }
        
        // Validate all parameter ranges
        ValidateParameterRanges(config);
        
        auto results = knn_algs::CalculateKNN(mgp::Graph(memgraph_graph), config);
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
        std::vector<mgp::Return> returns = {
            mgp::Return(kFieldNode, mgp::Type::Node),
            mgp::Return(kFieldNeighbour, mgp::Type::Node),
            mgp::Return(kFieldSimilarity, mgp::Type::Double)
        };
        
        // Single config parameter
        std::vector<mgp::Parameter> parameters = {
            mgp::Parameter(kArgumentConfig, mgp::Type::Map)
        };
        
        // Add the single get procedure
        mgp::AddProcedure(Get, kProcedureGet, mgp::ProcedureType::Read, 
                         parameters, returns, module, memory);
        
    } catch(const std::exception &e) {
        return 1;
    } 
    return 0;
}

extern "C" int mgp_shutdown_module() { 
    return 0; 
}
