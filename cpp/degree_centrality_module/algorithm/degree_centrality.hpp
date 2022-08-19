#pragma once

#include <mg_graph.hpp>

namespace degree_cenntrality_alg {

std::vector<double> GetDegreeCentrality(const mg_graph::GraphView<> &graph);

}  // namespace degree_cenntrality_alg