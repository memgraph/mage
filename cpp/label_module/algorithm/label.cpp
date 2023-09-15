#include "label.hpp"
#include "mg_procedure.h"
#include "mgp.hpp"

void Label::Exists(mgp_list *args, mgp_func_context *ctx, mgp_func_result *res, mgp_memory *memory) {
  mgp::memory = memory;
  const auto arguments = mgp::List(args);
  auto result = mgp::Result(res);
  try {
    bool exists = false;

    const auto label = arguments[1].ValueString();
    if (arguments[0].IsNode()) {
      const auto node = arguments[0].ValueNode();
      exists = node.HasLabel(label);
    }
    result.SetValue(exists);

  } catch (const std::exception &e) {
    result.SetErrorMessage(e.what());
    return;
  }
}
