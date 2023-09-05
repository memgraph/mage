#include <mgp.hpp>

#include "algorithm/create.hpp"

extern "C" int mgp_init_module(struct mgp_module *module, struct mgp_memory *memory) {
  try {
    mgp::memory = memory;

    AddProcedure(Create::SetRelProperty, Create::kProcedureSetRelProp, mgp::ProcedureType::Write,
                 {mgp::Parameter(Create::kArgumentsRelationship, mgp::Type::Any),
                  mgp::Parameter(Create::kArgumentsKey, mgp::Type::String),
                  mgp::Parameter(Create::kArgumentsValue, mgp::Type::Any)},
                 {mgp::Return(Create::kReturnRelProp, mgp::Type::Relationship)}, module, memory);

    AddProcedure(Create::RemoveLabels, Create::kProcedureRemoveLabels, mgp::ProcedureType::Write,
                 {mgp::Parameter(Create::kArgumentsNodes, mgp::Type::Any),
                  mgp::Parameter(Create::kArgumentsLabels, {mgp::Type::List, mgp::Type::String})},
                 {mgp::Return(Create::kReturnRemoveLabels, mgp::Type::Node)}, module, memory);

    AddProcedure(Create::SetProperties, Create::kProcedureSetProperties, mgp::ProcedureType::Write,
                 {mgp::Parameter(Create::kArgumentsNodes, mgp::Type::Any),
                  mgp::Parameter(Create::kArgumentsKeys, {mgp::Type::List, mgp::Type::String}),
                  mgp::Parameter(Create::kArgumentsValues, {mgp::Type::List, mgp::Type::Any})},
                 {mgp::Return(Create::kReturnProperties, mgp::Type::Node)}, module, memory);

    AddProcedure(Create::Node, Create::kProcedureNode, mgp::ProcedureType::Write,
                 {mgp::Parameter(Create::kArgumentsLabelsList, {mgp::Type::List, mgp::Type::String}),
                  mgp::Parameter(Create::kArgumentsProperties, mgp::Type::Map)},
                 {mgp::Return(Create::kReturnNode, mgp::Type::Node)}, module, memory);

    AddProcedure(
        Create::Nodes, std::string(Create::kProcedureNodes).c_str(), mgp::ProcedureType::Write,
        {mgp::Parameter(std::string(Create::kArgumentLabelsNodes).c_str(), {mgp::Type::List, mgp::Type::String}),
         mgp::Parameter(std::string(Create::kArgumentPropertiesNodes).c_str(), {mgp::Type::List, mgp::Type::Map})},
        {mgp::Return(std::string(Create::kReturnNodes).c_str(), mgp::Type::Node)}, module, memory);

    AddProcedure(Create::RemoveProperties, std::string(Create::kProcedureRemoveProperties).c_str(),
                 mgp::ProcedureType::Write,
                 {mgp::Parameter(std::string(Create::kArgumentNodeRemoveProperties).c_str(), mgp::Type::Any),
                  mgp::Parameter(std::string(Create::kArgumentKeysRemoveProperties).c_str(),
                                 {mgp::Type::List, mgp::Type::String})},
                 {mgp::Return(std::string(Create::kReturntRemoveProperties).c_str(), mgp::Type::Node)}, module, memory);

    AddProcedure(Create::SetProperty, std::string(Create::kProcedureSetProperty).c_str(), mgp::ProcedureType::Write,
                 {mgp::Parameter(std::string(Create::kArgumentNodeSetProperty).c_str(), mgp::Type::Any),
                  mgp::Parameter(std::string(Create::kArgumentKeySetProperty).c_str(), mgp::Type::String),
                  mgp::Parameter(std::string(Create::kArgumentValueSetProperty).c_str(), mgp::Type::Any)},
                 {mgp::Return(std::string(Create::kReturntSetProperty).c_str(), mgp::Type::Node)}, module, memory);

    AddProcedure(Create::RemoveRelProperties, Create::kProcedureRemoveRelProperties, mgp::ProcedureType::Write,
                 {mgp::Parameter(Create::kRemoveRelPropertiesArg1, mgp::Type::Any),
                  mgp::Parameter(Create::kRemoveRelPropertiesArg2, {mgp::Type::List, mgp::Type::String})},
                 {mgp::Return(Create::kResultRemoveRelProperties, mgp::Type::Relationship)}, module, memory);

    AddProcedure(Create::SetRelProperties, Create::kProcedureSetRelProperties, mgp::ProcedureType::Write,
                 {mgp::Parameter(Create::kSetRelPropertiesArg1, mgp::Type::Any),
                  mgp::Parameter(Create::kSetRelPropertiesArg2, {mgp::Type::List, mgp::Type::String}),
                  mgp::Parameter(Create::kSetRelPropertiesArg3, {mgp::Type::List, mgp::Type::Any})},
                 {mgp::Return(Create::kResultSetRelProperties, mgp::Type::Relationship)}, module, memory);

    AddProcedure(Create::Relationship, Create::kProcedureRelationship, mgp::ProcedureType::Write,
                 {mgp::Parameter(Create::kRelationshipArg1, mgp::Type::Node),
                  mgp::Parameter(Create::kRelationshipArg2, mgp::Type::String),
                  mgp::Parameter(Create::kRelationshipArg3, mgp::Type::Map),
                  mgp::Parameter(Create::kRelationshipArg4, mgp::Type::Node)},
                 {mgp::Return(Create::kResultRelationship, mgp::Type::Relationship)}, module, memory);

  } catch (const std::exception &e) {
    return 1;
  }

  return 0;
}

extern "C" int mgp_shutdown_module() { return 0; }
