#include <mgp.hpp>
#include <math.h>
# define M_PI           3.14159265358979323846  /* pi */


const char *kProcedureSingle = "single";
const char *kProcedureMultiple = "multiple";

const char *kReturnDistance = "distance";
const char *kReturnDistances = "distances";


const char *kArgumentStartPoint = "start_point";
const char *kArgumentEndPoint = "end_point";
const char *kArgumentStart = "start";
const char *kArgumentEnd = "end";


double distance_calc(const mgp::Node &node1, const mgp::Node &node2){

//   def calculate_distance_between_points(
//     start: Dict[str, float], end: Dict[str, float], metrics="m"
// ):
    // """
    // Returns distance based on the metrics between 2 points.
    // :param start: Start node - dictionary with lat and lng
    // :param end: End node - dictionary with lat and lng
    // :param metrics: m - in metres, km - in kilometres
    // :return: float distance
    // """

    // if (
    //     LATITUDE not in start.keys()
    //     or LONGITUDE not in start.keys()
    //     or LATITUDE not in end.keys()
    //     or LONGITUDE not in end.keys()
    // ):
    //     raise InvalidCoordinatesException("Latitude/longitude not specified!")

    // lat_1_str = start[LATITUDE]
    // lng_1_str = start[LONGITUDE]
    // lat_2_str = end[LATITUDE]
    // lng_2_str = end[LONGITUDE]

    // if not all([lat_1_str, lng_1_str, lat_2_str, lng_2_str]):
    //     raise InvalidCoordinatesException("Latitude/longitude not specified!")

    // try:
    //     lat_1 = float(lat_1_str)
    //     lat_2 = float(lat_2_str)
    //     lng_1 = float(lng_1_str)
    //     lng_2 = float(lng_2_str)
    // except ValueError:
    //     raise InvalidCoordinatesException("Latitude/longitude not in numerical format!")

    // if not isinstance(metrics, str) or metrics.lower() not in VALID_METRICS:
    //     raise InvalidMetricException("Invalid metric exception!")

    // R = 6371e3
    // pi_radians = math.pi / 180.00

    // phi_1 = lat_1 * pi_radians
    // phi_2 = lat_2 * pi_radians
    // delta_phi = (lat_2 - lat_1) * pi_radians
    // delta_lambda = (lng_2 - lng_1) * pi_radians

    // sin_delta_phi = math.sin(delta_phi / 2.0)
    // sin_delta_lambda = math.sin(delta_lambda / 2.0)

    // a = sin_delta_phi**2 + math.cos(phi_1) * math.cos(phi_2) * (sin_delta_lambda**2)
    // c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))

    // # Distance in metres
    // distance = R * c

    // if metrics.lower() == "km":
    //     distance *= KM_MULTIPLIER

    // return distance  
  auto prop_node1 = node1.Properties();
  auto prop_node2 = node2.Properties();

  prop_node1["lng"]=mgp::Value(60.0);
  prop_node1["lat"]=mgp::Value(60.0);

  prop_node2["lng"]=mgp::Value(60.0);
  prop_node2["lat"]=mgp::Value(60.0);

  auto lng1 = prop_node1["lng"].ValueDouble();
  auto lat1 = prop_node1["lat"].ValueDouble();
  
  auto lat2 = prop_node2["lat"].ValueDouble();
  auto lng2 = prop_node2["lng"].ValueDouble();

  double pi_rad = M_PI / 180.0;
  double R = 6371000.0;

  double phi_1 = lat1 * pi_rad;
  double phi_2 = lat2 * pi_rad;

  double delta_phi = (lat2 - lat1) * pi_rad;
  double delta_lambda = (lng2 - lng1) * pi_rad;

  double sin_delta_phi = sin(delta_phi / 2);
  double sin_delta_lambda = sin(delta_lambda /2);

  double a = sin_delta_phi*sin_delta_phi + cos(phi_1)*cos(phi_2)* (sin_delta_lambda*sin_delta_lambda);
  double c = 2 * atan2(sqrt(a),sqrt(1-a));


  return R*c;
}

void Single(mgp_list *args, mgp_graph *memgraph_graph, mgp_result *result, mgp_memory *memory) {
  mgp::memory = memory;
  const auto arguments = mgp::List(args);
  const auto record_factory = mgp::RecordFactory(result);
  
  const mgp::Node &node1 = arguments[0].ValueNode();
  const mgp::Node &node2 = arguments[1].ValueNode();

  auto record = record_factory.NewRecord();

  record.Insert(kReturnDistance, distance_calc(node1, node2));

}

void Multiple(mgp_list *args, mgp_graph *memgraph_graph, mgp_result *result, mgp_memory *memory) {
  mgp::memory = memory;
  const auto arguments = mgp::List(args);
  const auto record_factory = mgp::RecordFactory(result);
  
  mgp::List distances= mgp::List();

  const auto &start = arguments[0].ValueList();
  const auto &end = arguments[1].ValueList();

  for(const auto &node: start){
    distances.Append(mgp::Value(static_cast<double>(1)));
  }

  auto record = record_factory.NewRecord();
  record.Insert(kReturnDistances, distances);

}



extern "C" int mgp_init_module(struct mgp_module *module, struct mgp_memory *memory) {
  try {
    mgp::memory = memory;

    AddProcedure(Single, kProcedureSingle, mgp::ProcedureType::Read,
                 {mgp::Parameter(kArgumentStartPoint, mgp::Type::Node), mgp::Parameter(kArgumentEndPoint, mgp::Type::Node)}, 
                 {mgp::Return(kReturnDistance, mgp::Type::Double)},
                 module, memory);

    const auto multiple_input = std::make_pair(mgp::Type::List, mgp::Type::Node);
    const auto multiple_return = std::make_pair(mgp::Type::List, mgp::Type::Double);
    AddProcedure(Multiple, kProcedureMultiple, mgp::ProcedureType::Read,
                 {mgp::Parameter(kArgumentStart, multiple_input), mgp::Parameter(kArgumentEnd, multiple_input)}, 
                 {mgp::Return(kReturnDistances, multiple_return)},
                 module, memory);

  } catch (const std::exception &e) {
    return 1;
  }

  return 0;
}

extern "C" int mgp_shutdown_module() { return 0; }
