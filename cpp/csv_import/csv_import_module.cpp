#include <mg_utils.hpp>
#include <mgp.hpp>

#include "algorithm/csv_import.hpp"

extern "C" int mgp_init_module(mgp_module *module, mgp_memory *memory) {
  try {
    mgp::memory = memory;
    mgp::AddProcedure(CsvImport::CreateCsvFile, "create_csv_file", mgp::ProcedureType::Read,
                      {
                          mgp::Parameter("string", mgp::Type::String),
                          mgp::Parameter("string", mgp::Type::String),
                      },
                      {mgp::Return(CsvImport::kFilepath, {mgp::Type::String})}, module, memory);
    mgp::AddProcedure(CsvImport::DeleteFile, "delete_file", mgp::ProcedureType::Read,
                      {
                          mgp::Parameter("string", mgp::Type::String),
                      },
                      {}, module, memory);
  } catch (const std::exception &e) {
    return 1;
  }
  return 0;
}