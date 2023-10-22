#include <mgp.hpp>

#include "algorithm/label.hpp"

extern "C" int mgp_init_module(struct mgp_module *module, struct mgp_memory *memory) {
  try {
    mgp::MemoryDispatcherGuard guard{memory};;

    mgp::AddFunction(Label::Exists, Label::kFunctionExists,
                     {mgp::Parameter(Label::kArgumentsNode, mgp::Type::Any),
                      mgp::Parameter(Label::kArgumentsLabel, mgp::Type::String)},
                     module, memory);

  } catch (const std::exception &e) {
    return 1;
  }

  return 0;
}

extern "C" int mgp_shutdown_module() { return 0; }
