#include <tuple>
#include "gtest/gtest.h"
#include "mg_test_utils.hpp"

#include "algorithm/hits.hpp"

class HitsTests : public testing::TestWithParam<std::tuple<hits_alg::HitsGraph, std::tuple<std::vector<double>, std::vector<double> > >>{};

TEST_P(HitsTests, ParametrizedTest){
    auto graph =std::get<0>(GetParam());
    auto [hub, auth] = std::get<1>(GetParam());
    auto [hub_result, auth_result] = hits_alg::ParallelIterativeHits(graph);
    ASSERT_TRUE(mg_test_utility::TestEqualVectors(hub_result, hub));
    ASSERT_TRUE(mg_test_utility::TestEqualVectors(auth_result, auth));

}

INSTANTIATE_TEST_SUITE_P( Hits, HitsTests,

        testing::Values(
                std::make_tuple(hits_alg::HitsGraph(1, 0, {}),std::make_tuple(std::vector<double>{0}, std::vector<double>{0})),
                std::make_tuple(hits_alg::HitsGraph(2, 1, {{0, 1}}),std::make_tuple(std::vector<double>{1, 0}, std::vector<double>{0, 1})),
                std::make_tuple(hits_alg::HitsGraph(0, 0, {}),std::make_tuple(std::vector<double>{}, std::vector<double>{})),
                std::make_tuple(hits_alg::HitsGraph(1, 1, {{0, 0}}),std::make_tuple(std::vector<double>{1}, std::vector<double>{1})),
                std::make_tuple(hits_alg::HitsGraph(2, 2, {{0, 1},{0, 1}}),std::make_tuple(std::vector<double>{1, 0}, std::vector<double>{0, 1})),
                std::make_tuple(hits_alg::HitsGraph(2, 1, {{1, 1}}),std::make_tuple(std::vector<double>{0, 1}, std::vector<double>{0, 1})),
                std::make_tuple(hits_alg::HitsGraph(5, 11, {{0, 2},{0, 0},{2, 3},{3, 1},{1, 3},{1, 0},{1, 2},{3, 0},{0, 1},{3, 2},{4, 0}}),
                                std::make_tuple(std::vector<double>{0.293869, 0.257972, 0.0361229, 0.293869, 0.118168},
                                                std::vector<double>{0.358123, 0.218354, 0.314219, 0.109304, 0.0000})),
                std::make_tuple(hits_alg::HitsGraph(4, 4, {{1, 0},{3, 0},{2, 0},{3, 0}}),
                                std::make_tuple(std::vector<double>{0, 0.25, 0.25, 0.5},
                                                std::vector<double>{1, 0, 0, 0})),

                std::make_tuple(hits_alg::HitsGraph(7, 30, {{0, 6},{3, 0},{6, 2},{0, 3},{2, 3},
                        {6, 4},{1, 1},{2, 0},{0, 3},{5, 0},{0, 4},{5, 2},{1, 5},{5, 3},{2, 3},{6, 1},{2, 0},
                        {6, 1},{2, 6},{2, 2},{0, 0},{6, 0},{6, 0},{0, 6},{3, 3},{6, 3},{1, 3},{4, 0},{1, 2},
                        {2, 1}}), std::make_tuple(
                                std::vector<double>{0.189179, 0.0958985, 0.247642, 0.0898859, 0.0446254, 0.111874, 0.220895},
                                std::vector<double>{0.256163, 0.146562, 0.126215, 0.259812, 0.076528, 0.0178975, 0.116822})),

                std::make_tuple(hits_alg::HitsGraph(10, 9, {{8, 8}, {2, 2}, {0, 8}, {7, 8}, {1, 6}, {0, 0}, {1, 1}, {6, 3}, {9, 5}}),
                std::make_tuple(std::vector<double>{0.414214  ,0  ,0  ,0  ,0  ,0  ,0  ,0.292893  ,0.292893  ,0  },
                                std::vector<double>{0.292893  ,0  ,0  ,0  ,0  ,0  ,0  ,0  ,0.707107  ,0}))

));




int main(int argc, char **argv) {
    testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}