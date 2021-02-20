#include "pagerank.hpp"

#include <iostream>
#include <vector>

// TODO(gitbuda): Add unit test lib to the pagerank module.
// TODO(gitbuda): Add unit tests to the pagerank module.

int main() {
  pagerank::PageRankGraph graph(4, 5, {{0, 1}, {1, 2}, {0, 2}, {2, 0}, {3, 2}});
  std::vector<double> correct_ranks{0.372474, 0.195798, 0.394097, 0.0375};
  auto ranks = pagerank::ParallelIterativePageRank(graph);
  if (!std::equal(ranks.begin(), ranks.end(), correct_ranks.begin(),
                  [](const auto first, const auto second) {
                    if (std::abs(first - second) < 10e-2) {
                      return true;
                    }
                    return false;
                  })) {
    std::cerr << "Simple Pagerank example is NOT correct." << std::endl;
    std::exit(1);
  } else {
    std::cout << "Simple Pagerank example is correct." << std::endl;
  }

  return 0;
}
