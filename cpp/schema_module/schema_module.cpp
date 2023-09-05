#include <mgp.hpp>

#include "algorithm/schema.hpp"

extern "C" int mgp_init_module(struct mgp_module *module, struct mgp_memory *memory) {
  try {
    mgp::memory = memory;

    AddProcedure(Schema::NodeTypeProperties, std::string(Schema::kProcedureNodeType).c_str(), mgp::ProcedureType::Read,
                 {},
                 {mgp::Return(std::string(Schema::kReturnLabels).c_str(), {mgp::Type::List, mgp::Type::String}),
                  mgp::Return(std::string(Schema::kReturnPropertyName).c_str(), mgp::Type::String),
                  mgp::Return(std::string(Schema::kReturnPropertyType).c_str(), mgp::Type::String),
                  mgp::Return(std::string(Schema::kReturnMandatory).c_str(), mgp::Type::Bool)},
                 module, memory);

    AddProcedure(Schema::RelTypeProperties, std::string(Schema::kProcedureRelType).c_str(), mgp::ProcedureType::Read,
                 {},
                 {mgp::Return(std::string(Schema::kReturnRelType).c_str(), mgp::Type::String),
                  mgp::Return(std::string(Schema::kReturnPropertyName).c_str(), mgp::Type::String),
                  mgp::Return(std::string(Schema::kReturnPropertyType).c_str(), mgp::Type::String),
                  mgp::Return(std::string(Schema::kReturnMandatory).c_str(), mgp::Type::Bool)},
                 module, memory);

  } catch (const std::exception &e) {
    return 1;
  }

  return 0;
}

extern "C" int mgp_shutdown_module() { return 0; }