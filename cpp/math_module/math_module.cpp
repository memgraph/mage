#include <cstdint>
#include <mgp.hpp>

#include "algorithm/math.hpp"

extern "C" int mgp_init_module(struct mgp_module *module, struct mgp_memory *memory) {
  try {
    mgp::MemoryDispatcherGuard guard{memory};

    mgp::AddFunction(
        Math::Round, Math::kProcedureRound,
        {mgp::Parameter(std::string(Math::kArgumentValue).c_str(), mgp::Type::Double, 0.0),
         mgp::Parameter(std::string(Math::kArgumentPrecision).c_str(), mgp::Type::Int, static_cast<int64_t>(0)),
         mgp::Parameter(std::string(Math::kArgumentMode).c_str(), mgp::Type::String, "HALF_UP")},
        module, memory);

  } catch (const std::exception &e) {
    return 1;
  }

  return 0;
}

extern "C" int mgp_shutdown_module() { return 0; }
