#include <mgp.hpp>

#include "algorithms/collections.hpp"

const char *kProcedureUnionALl = "unionAll";

const char *kReturnUnionAll="return_list";


const char *kArgumentList1= "list1";
const char *kArgumentList2= "list2";



extern "C" int mgp_init_module(struct mgp_module *module, struct mgp_memory *memory) {
  try {
    mgp::memory= memory;
    AddProcedure(unionAll,kProcedureUnionALl, mgp::ProcedureType::Read,
    {mgp::Parameter(kArgumentList1, {mgp::Type::List, mgp::Type::Any}), mgp::Parameter(kArgumentList2, {mgp::Type::List, mgp::Type::Any})},
    {mgp::Return(kReturnUnionAll,{mgp::Type::List, mgp::Type::Any})},module,memory);

  } catch (const std::exception &e) {
    return 1;
  }

  return 0;
}

extern "C" int mgp_shutdown_module() { return 0; }

