#include <mgp.hpp>

#include "algorithms/collections.hpp"

const char *kProcedureContains= "contains";


const char *kReturnContains="output";


const char *kArgumentList="list";
const char *kArgumentValue="value";



extern "C" int mgp_init_module(struct mgp_module *module, struct mgp_memory *memory) {
  try {
    mgp::memory= memory;
    


    AddProcedure(contains,kProcedureContains, mgp::ProcedureType::Read,
    {mgp::Parameter(kArgumentList, {mgp::Type::List, mgp::Type::Any}), mgp::Parameter(kArgumentValue, mgp::Type::Any)},
    {mgp::Return(kReturnContains,mgp::Type::Bool)},module,memory);


  } catch (const std::exception &e) {
    return 1;
  }

  return 0;
}

extern "C" int mgp_shutdown_module() { return 0; }