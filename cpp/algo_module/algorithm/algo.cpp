#include "algo.hpp"

double Algo::toRadians(double degrees) { return degrees * M_PI / 180.0; }

double Algo::haversineDistance(double lat1, double lon1, double lat2, double lon2) {
  const double earthRadius = 6371.0;  // IN KM

  lat1 = toRadians(lat1);
  lon1 = toRadians(lon1);
  lat2 = toRadians(lat2);
  lon2 = toRadians(lon2);

  double dLat = lat2 - lat1;
  double dLon = lon2 - lon1;
  double a = sin(dLat / 2) * sin(dLat / 2) + cos(lat1) * cos(lat2) * sin(dLon / 2) * sin(dLon / 2);
  double c = 2 * atan2(sqrt(a), sqrt(1 - a));
  double distance = earthRadius * c;

  return distance;  // returns distance in km
}

/*calculates the heuristic based on haversine, or returns the value if the heuristic is custom*/
double Algo::CalculateHeuristic(const Config &config, const mgp::Node &node, const GoalNodes &nodes) {
  if (config.heuristic_name != "") {
    auto heuristic = node.GetProperty(config.heuristic_name);
    if (heuristic.IsNumeric()) {
      return heuristic.ValueNumeric();
    }
    if (heuristic.IsDuration() && config.duration) {
      return heuristic.ValueDuration().Microseconds();
    }
    throw mgp::ValueException("Custom heuristic property must be of a numeric, or duration data type!");
  }

  auto latitude_source = node.GetProperty(config.latitude_name);
  auto longitude_source = node.GetProperty(config.latitude_name);
  if (latitude_source.IsNull() || longitude_source.IsNull()) {
    throw mgp::ValueException(
        "Latitude and longitude properties, or a custom heuristic value, must be specified in every node!");
  }
  if (latitude_source.IsNumeric() && longitude_source.IsNumeric()) {
    return haversineDistance(latitude_source.ValueNumeric(), longitude_source.ValueNumeric(), nodes.latLon.first,
                             nodes.latLon.second);
  }
  throw mgp::ValueException("Latitude and longitude must be numeric data types!");
}

std::pair<double, double> Algo::TargetLatLon(const mgp::Node &target, const Config &config) {
  if (config.heuristic_name != "") {  // if custom heuristic, dont return latitude and longitude
    return std::make_pair<double, double>(0, 0);
  }

  auto latitude = target.GetProperty(config.latitude_name);
  auto longitude = target.GetProperty(config.latitude_name);
  if (latitude.IsNull() || longitude.IsNull()) {
    throw mgp::ValueException("Latitude and longitude properties of the target node must be specified!");
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
                 const bool in) {  // in true incoming, in false outgoing
  if (config.in_rels.size() == 0 && config.out_rels.size() == 0) {
    return true;
  }

  if (in && config.in_rels.find(std::string(rel.Type())) != config.in_rels.end()) {
    return true;
  }

  if (!in && config.out_rels.find(std::string(rel.Type())) != config.out_rels.end()) {
    return true;
  }

  return false;
}

bool Algo::LabelOk(const mgp::Node &node, const Config &config) {
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
void Algo::ParseRelationships(const std::shared_ptr<NodeObject> &prev, bool in, const GoalNodes &nodes, Lists &lists,
                              const Config &config) {
  auto rels = in ? prev->node.InRelationships() : prev->node.OutRelationships();
  for (const auto rel : rels) {
    if (!RelOk(rel, config, in)) {
      continue;
    }
    const auto node = in ? rel.From() : rel.To();
    if (!LabelOk(node, config)) {
      continue;
    }
    auto heuristic = CalculateHeuristic(config, node, nodes) * config.epsilon;  // epsilon 0 == UCS
    auto distance = CalculateDistance(config, rel);
    std::shared_ptr<NodeObject> nb =
        std::make_shared<NodeObject>(heuristic, distance + prev->total_distance, node, rel, prev);
    if (!lists.closed.FindAndCompare(nb)) {
      continue;
    }
    lists.open.Insert(nb);
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
  Lists lists = Lists();

  std::shared_ptr<NodeObject> start_nb = InitializeStart(nodes.start);
  lists.open.Insert(start_nb);

  while (!lists.open.Empty()) {
    auto nb = lists.open.Top();
    lists.open.Pop();
    if (nb->node == nodes.target) {
      return BuildResult(nb, nodes.start);
    }
    lists.closed.Insert(nb);
    ParseRelationships(nb, false, nodes, lists, config);
    ParseRelationships(nb, true, nodes, lists, config);
  }
  return std::pair<mgp::Path, double>(mgp::Path(nodes.start), 0);
}

std::pair<mgp::Path, double> Algo::BuildResult(std::shared_ptr<NodeObject> final, const mgp::Node &start) {
  mgp::Path path = mgp::Path(start);
  std::vector<mgp::Relationship> rels;

  double weight = final->total_distance;
  while (final->prev) {
    rels.push_back(final->rel);
    final = final->prev;
  }

  for (auto it = rels.rbegin(); it != rels.rend(); ++it) {
    path.Expand(*it);
  }

  return std::pair<mgp::Path, double>(path, weight);
}

void Algo::AStar(mgp_list *args, mgp_graph *memgraph_graph, mgp_result *result, mgp_memory *memory) {
  mgp::MemoryDispatcherGuard guard{memory};
  const auto arguments = mgp::List(args);
  const auto record_factory = mgp::RecordFactory(result);

  try {
    auto start = arguments[0].ValueNode();
    auto target = arguments[1].ValueNode();
    auto config = Config(arguments[2].ValueMap());
    std::pair<double, double> latLon = TargetLatLon(target, config);
    auto nodes = GoalNodes(start, target, latLon);
    std::pair<mgp::Path, double> pair = HelperAstar(nodes, config);
    const mgp::Path path = pair.first;
    const double weight = pair.second;

    auto record = record_factory.NewRecord();
    record.Insert(std::string(kAStarRet1).c_str(), path);
    record.Insert(std::string(kAStarRet2).c_str(), weight);

  } catch (const std::exception &e) {
    record_factory.SetErrorMessage(e.what());
    return;
  }
}
