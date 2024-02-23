#include <mgp.hpp>

#include "algorithm/util.hpp"

extern "C" int mgp_init_module(struct mgp_module *module, struct mgp_memory *memory) {
  try {
    mgp::MemoryDispatcherGuard guard{memory};
    AddProcedure(Util::Md5Procedure, std::string(Util::kProcedureMd5).c_str(), mgp::ProcedureType::Write,
                 {mgp::Parameter(std::string(Util::kArgumentValuesMd5).c_str(), mgp::Type::Any)},
                 {mgp::Return(std::string(Util::kArgumentResultMd5).c_str(), mgp::Type::String)}, module, memory);

    mgp::AddFunction(Util::Md5Function, "md5", {mgp::Parameter("stringToHash", mgp::Type::Any)}, module, memory);
  } catch (const std::exception &e) {
    return 1;
  }

  return 0;
}

extern "C" int mgp_shutdown_module() { return 0; }
