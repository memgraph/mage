// Copyright (c) 2016-2022 Memgraph Ltd. [https://memgraph.com]
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include <stdio.h>
#include <iostream>

#include <cugraph/algorithms.hpp>
#include <mg_exceptions.hpp>
#include <mg_utils.hpp>
#include <raft/distance/distance.hpp>
#include <raft/handle.hpp>
#include <rmm/device_uvector.hpp>

// TODO(gitbuda): Just a RAPIDS example, put something useful here, buda ❤ cuda + rapids + cugraph!

constexpr char const *kProcedureRapidsExample = "rapids_example";
constexpr char const *kProcedureCugraphExample = "cugraph_example";
constexpr char const *kResultValue = "value";

void InsertRecord(mgp_graph *, mgp_result *result, mgp_memory *memory, const double value) {
  auto *record = mgp::result_new_record(result);
  mg_utility::InsertDoubleValueResult(record, kResultValue, value, memory);
}

void ExampleRapidsProc(mgp_list *args, mgp_graph *memgraph_graph, mgp_result *result, mgp_memory *memory) {
  raft::handle_t handle{};
  auto stream = handle.get_stream_view();

  int n_samples = 3;
  int n_features = 2;
  rmm::device_uvector<float> input(n_samples * n_features, stream);
  for (int i = 0; i < input.size(); ++i) {
    float value = i;
    input.set_element_async(i, value, stream);
  }
  stream.synchronize_no_throw();
  rmm::device_uvector<float> output(n_samples * n_samples, stream);
  auto metric = raft::distance::DistanceType::L1;  // Sum of distances in each feature vector.
  raft::distance::pairwise_distance(handle, input.data(), input.data(), output.data(), n_samples, n_samples, n_features,
                                    metric);

  for (int i = 0; i < output.size(); ++i) {
    auto value = output.element(i, stream);
    InsertRecord(memgraph_graph, result, memory, value);
  }
}

void ExampleCugraphProc(mgp_list *args, mgp_graph *memgraph_graph, mgp_result *result, mgp_memory *memory) {
  // TODO(gitbuda): Implement example cuGraph Memgraph procedure.
}

extern "C" int mgp_init_module(struct mgp_module *module, struct mgp_memory *memory) {
  try {
    struct mgp_proc *example_rapids_proc =
        mgp::module_add_read_procedure(module, kProcedureRapidsExample, ExampleRapidsProc);
    mgp::proc_add_result(example_rapids_proc, kResultValue, mgp::type_float());
    struct mgp_proc *example_cugraph_proc =
        mgp::module_add_read_procedure(module, kProcedureCugraphExample, ExampleRapidsProc);
    mgp::proc_add_result(example_cugraph_proc, kResultValue, mgp::type_float());
  } catch (std::exception &e) {
    return 1;
  }
  return 0;
}

extern "C" int mgp_shutdown_module() { return 0; }
