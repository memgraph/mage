#include <mg_utils.hpp>
#include <mgp.hpp>

#include "algorithm/csv_utils.hpp"

extern "C" int mgp_init_module(mgp_module *module, mgp_memory *memory) {
  try {
    mgp::memory = memory;
    mgp::AddProcedure(CsvUtils::CreateCsvFile, CsvUtils::kProcedureCreateCsvFile, mgp::ProcedureType::Read,
                      {
                          mgp::Parameter(CsvUtils::kArgumentCreateCsvFile1, mgp::Type::String),
                          mgp::Parameter(CsvUtils::kArgumentCreateCsvFile2, mgp::Type::String),
                          mgp::Parameter(CsvUtils::kArgumentCreateCsvFile3, mgp::Type::Bool, false),
                      },
                      {mgp::Return(CsvUtils::kArgumentDeleteCsvFile1, {mgp::Type::String})}, module, memory);
    mgp::AddProcedure(CsvUtils::DeleteCsvFile, CsvUtils::kProcedureDeleteCsvFile, mgp::ProcedureType::Read,
                      {
                          mgp::Parameter(CsvUtils::kArgumentDeleteCsvFile1, mgp::Type::String),
                      },
                      {}, module, memory);
  } catch (const std::exception &e) {
    return 1;
  }
  return 0;
}