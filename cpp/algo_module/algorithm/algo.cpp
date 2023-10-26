#include "algo.hpp"

#include "algo.hpp"

void Algo::Cover(mgp_list *args, mgp_graph *memgraph_graph, mgp_result *result, mgp_memory *memory) {
  mgp::MemoryDispatcherGuard guard{memory};
  const auto arguments = mgp::List(args);
  const auto record_factory = mgp::RecordFactory(result);
  try {
    auto list_nodes = arguments[0].ValueList();
    std::unordered_set<mgp::Node> nodes;
    for (const auto &elem : list_nodes) {
      auto node = elem.ValueNode();
      nodes.insert(node);
    }

    for (const auto &node : nodes) {
      for (const auto rel : node.OutRelationships()) {
        if (nodes.find(rel.To()) != nodes.end()) {
          auto record = record_factory.NewRecord();
          record.Insert(std::string(kCoverRet1).c_str(), rel);
        }
      }
    }

  } catch (const std::exception &e) {
    record_factory.SetErrorMessage(e.what());
    return;
  }

}

void Algo::CheckConfigTypes(const mgp::Map &map) {
  if (!map.At("unweighted").IsNull() && !map.At("unweighted").IsBool()) {
    throw mgp::ValueException("unweighted config option should be bool!");
  }
  if (!map.At("epsilon").IsNull() && !map.At("epsilon").IsNumeric()) {
    throw mgp::ValueException("epsilon config option should be numeric!");
  }
  if (!map.At("distance_prop").IsNull() && !map.At("distance_prop").IsString()) {
    throw mgp::ValueException("distance_prop config option should be string!");
  }
  if (!map.At("heuristic_name").IsNull() && !map.At("heuristic_name").IsString()) {
    throw mgp::ValueException("heuristic_name config option should be string!");
  }
  if (!map.At("latitude_name").IsNull() && !map.At("latitude_name").IsString()) {
    throw mgp::ValueException("latitude_name config option should be string!");
  }
  if (!map.At("longitude_name").IsNull() && !map.At("longitude_name").IsString()) {
    throw mgp::ValueException("longitude_name config option should be string!");
  }
  if (!map.At("whitelisted_labels").IsNull() && !map.At("whitelisted_labels").IsList()) {
    throw mgp::ValueException("whitelisted_labels config option should be list!");
  }
  if (!map.At("whitelisted_labels").IsNull()) {
    auto list = map.At("whitelisted_labels").ValueList();
    for (const auto value : list) {
      if (!value.IsString()) {
        throw mgp::ValueException("Labels in the whitelisted_labels config list must be strings!");
      }
    }
  }
  if (!map.At("blacklisted_labels").IsNull() && !map.At("blacklisted_labels").IsList()) {
    throw mgp::ValueException("blacklisted_labels config option should be list!");
  }
  if (!map.At("blacklisted_labels").IsNull()) {
    auto list = map.At("blacklisted_labels").ValueList();
    for (const auto value : list) {
      if (!value.IsString()) {
        throw mgp::ValueException("Labels in the blacklisted_labels config list must be strings!");
      }
    }
  }
  if (!map.At("relationships_filter").IsNull() && !map.At("relationships_filter").IsList()) {
    throw mgp::ValueException("relationships_filter config option should be list!");
  } else if (!map.At("relationships_filter").IsNull() && map.At("relationships_filter").IsList()) {
    auto list = map.At("relationships_filter").ValueList();
    for (const auto value : list) {
      if (!value.IsString()) {
        throw mgp::ValueException("Elements in relationships_filter config list must be strings!");
      }
      auto rel_type = std::string(value.ValueString());
      const size_t size = rel_type.size();
      const char first_elem = rel_type[0];
      const char last_elem = rel_type[size - 1];

      if (first_elem == '<' && last_elem == '>') {
        throw mgp::ValueException("Wrong relationship format => <relationship> is not allowed!");
      }
    }
  }

  if (!map.At("duration").IsNull() && !map.At("duration").IsBool()) {
    throw mgp::ValueException("duration config option should be bool!");
  }
}
double Algo::GetRadians(double degrees) { return degrees * M_PI / 180.0; }

double Algo::GetHaversineDistance(double lat1, double lon1, double lat2, double lon2) {
  // IN KM
  const double earthRadius = 6371.0;
  lat1 = GetRadians(lat1);
  lon1 = GetRadians(lon1);
  lat2 = GetRadians(lat2);
  lon2 = GetRadians(lon2);

  double dLat = lat2 - lat1;
  double dLon = lon2 - lon1;
  double a = sin(dLat / 2) * sin(dLat / 2) + cos(lat1) * cos(lat2) * sin(dLon / 2) * sin(dLon / 2);
  double c = 2 * atan2(sqrt(a), sqrt(1 - a));
  double distance = earthRadius * c;

  // returns distance in km
  return distance;
}

// calculates the heuristic based on haversine, or returns the value if the heuristic is custom
double Algo::CalculateHeuristic(const Config &config, const mgp::Node &node, const GoalNodes &nodes) {
  if (config.heuristic_name != kDefaultHeuristic) {
    auto heuristic = node.GetProperty(config.heuristic_name);
    if (heuristic.IsNumeric()) {
      return heuristic.ValueNumeric();
    }
    if (heuristic.IsDuration() && config.duration) {
      return heuristic.ValueDuration().Microseconds();
    }
    throw mgp::ValueException("Custom heuristic property must be of a numeric, or duration data type!");
  }
  auto coordinate_pair = GetLatLon(node, config);
  auto &latitude_source = coordinate_pair.first;
  auto &longitude_source = coordinate_pair.second;
  return GetHaversineDistance(latitude_source, longitude_source, nodes.lat_lon.first, nodes.lat_lon.second);
}

std::pair<double, double> Algo::GetLatLon(const mgp::Node &target, const Config &config) {
  auto latitude = target.GetProperty(config.latitude_name);
  auto longitude = target.GetProperty(config.latitude_name);
  if (latitude.IsNull() || longitude.IsNull()) {
    throw mgp::ValueException(
        "Latitude and longitude properties, or a custom heuristic value, must be specified in every node!");
  }
  if (latitude.IsNumeric() && longitude.IsNumeric()) {
    return std::make_pair<double, double>(latitude.ValueNumeric(), longitude.ValueNumeric());
  }
  throw mgp::ValueException("Latitude and longitude must be numeric data types!");
}

double Algo::CalculateDistance(const Config &config, const mgp::Relationship &rel) {
  if (config.unweighted) {  // return same distance if unweighted
    return 10;
  }
  auto distance = rel.GetProperty(config.distance_prop);
  if (distance.IsNull()) {
    throw mgp::ValueException("If the graph is weighted, distance property of the relationship must be specified!");
  }
  if (distance.IsNumeric()) {
    return distance.ValueNumeric();
  }
  if (distance.IsDuration() && config.duration) {
    return distance.ValueDuration().Microseconds();
  }
  throw mgp::ValueException("Distance property must be a numeric or duration datatype!");
}

bool Algo::RelOk(const mgp::Relationship &rel, const Config &config,
                 const RelationshipType rel_type) {  // in true incoming, in false outgoing
  if (config.in_rels.size() == 0 && config.out_rels.size() == 0) {
    return true;
  }

  if (rel_type == RelationshipType::IN && config.in_rels.find(std::string(rel.Type())) != config.in_rels.end()) {
    return true;
  }

  if (rel_type == RelationshipType::OUT && config.out_rels.find(std::string(rel.Type())) != config.out_rels.end()) {
    return true;
  }

  return false;
}

bool Algo::IsLabelOk(const mgp::Node &node, const Config &config) {
  bool whitelist_empty = config.whitelist.empty();

  for (auto label : node.Labels()) {
    if (config.blacklist.find(std::string(label)) != config.blacklist.end()) {
      return false;
    }
    if (!whitelist_empty && config.whitelist.find(std::string(label)) == config.whitelist.end()) {
      return false;
    }
  }
  return true;
}
void Algo::ExpandRelationships(const std::shared_ptr<NodeObject> &prev, const RelationshipType rel_type,
                               const GoalNodes &nodes, TrackingLists &lists, const Config &config) {
  auto rels = rel_type == RelationshipType::IN ? prev->node.InRelationships() : prev->node.OutRelationships();
  for (const auto rel : rels) {
    if (!RelOk(rel, config, rel_type)) {
      continue;
    }
    const auto node = rel_type == RelationshipType::IN ? rel.From() : rel.To();
    if (!IsLabelOk(node, config)) {
      continue;
    }
    auto heuristic = CalculateHeuristic(config, node, nodes) * config.epsilon;  // epsilon 0 == UCS
    auto distance = CalculateDistance(config, rel);
    auto nb = std::make_shared<NodeObject>(heuristic, distance + prev->total_distance, node, rel, prev);
    if (!lists.closed.FindAndCompare(nb)) {
      continue;
    }
    lists.open.InsertOrUpdate(nb);
  }
}

std::shared_ptr<Algo::NodeObject> Algo::InitializeStart(const mgp::Node &start) {
  if (start.InDegree() == 0 && start.OutDegree() == 0) {
    throw mgp::ValueException("Start node must have in or out relationships!");
  }

  if (start.InDegree() != 0) {
    return std::make_shared<NodeObject>(0, 0, start, *start.InRelationships().begin(), nullptr);
  }

  return std::make_shared<NodeObject>(0, 0, start, *start.OutRelationships().begin(), nullptr);
}

std::pair<mgp::Path, double> Algo::HelperAstar(const GoalNodes &nodes, const Config &config) {
  TrackingLists lists = TrackingLists();

  auto start_nb = InitializeStart(nodes.start);
  lists.open.InsertOrUpdate(start_nb);

  while (!lists.open.Empty()) {
    auto nb = lists.open.Top();
    lists.open.Pop();
    if (nb->node == nodes.target) {
      return BuildResult(nb, nodes.start);
    }
    lists.closed.Insert(nb);
    ExpandRelationships(nb, RelationshipType::OUT, nodes, lists, config);
    ExpandRelationships(nb, RelationshipType::IN, nodes, lists, config);
  }
  return std::pair<mgp::Path, double>(mgp::Path(nodes.start), 0);
}

std::pair<mgp::Path, double> Algo::BuildResult(std::shared_ptr<NodeObject> final_node, const mgp::Node &start) {
  mgp::Path path = mgp::Path(start);
  std::vector<mgp::Relationship> rels;

  double weight = final_node->total_distance;
  while (final_node->prev) {
    rels.push_back(final_node->rel);
    final_node = final_node->prev;
  }

  for (auto it = rels.rbegin(); it != rels.rend(); ++it) {
    path.Expand(std::move(*it));
  }

  return std::pair<mgp::Path, double>(std::move(path), weight);
}

void Algo::AStar(mgp_list *args, mgp_graph *memgraph_graph, mgp_result *result, mgp_memory *memory) {
  mgp::MemoryDispatcherGuard guard{memory};
  const auto arguments = mgp::List(args);
  const auto record_factory = mgp::RecordFactory(result);

  try {
    auto start = arguments[0].ValueNode();
    auto target = arguments[1].ValueNode();
    auto config_map = arguments[2].ValueMap();
    CheckConfigTypes(config_map);
    auto config = Config(config_map);
    // if there is a custom heuristic, there is no need for target latitude and lognitude
    auto nodes = config.heuristic_name == kDefaultHeuristic ? GoalNodes(start, target, GetLatLon(target, config))
                                                            : GoalNodes(start, target);
    std::pair<mgp::Path, double> pair = HelperAstar(nodes, config);
    mgp::Path &path = pair.first;
    const double weight = pair.second;

    auto record = record_factory.NewRecord();
    record.Insert(std::string(kAStarPath).c_str(), std::move(path));
    record.Insert(std::string(kAStarWeight).c_str(), weight);

  } catch (const std::exception &e) {
    record_factory.SetErrorMessage(e.what());
    return;
  }
}
