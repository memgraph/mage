#pragma once

#include <stdio.h>
#include <stdlib.h>
#include <filesystem>
#include <fstream>
#include <mgp.hpp>
namespace fs = std::filesystem;

namespace CsvUtils {

/* create_csv_file constants */
constexpr std::string_view kProcedureCreateCsvFile = "create_csv_file";
constexpr std::string_view kArgumentCreateCsvFile1 = "filepath";
constexpr std::string_view kArgumentCreateCsvFile2 = "content";
constexpr std::string_view kArgumentCreateCsvFile3 = "is_append";

void CreateCsvFile(mgp_list *args, mgp_graph *memgraph_graph, mgp_result *result, mgp_memory *memory);

/* delete_csv_file constants */
constexpr std::string_view kProcedureDeleteCsvFile = "delete_csv_file";
constexpr std::string_view kArgumentDeleteCsvFile1 = "filepath";

void DeleteCsvFile(mgp_list *args, mgp_graph *memgraph_graph, mgp_result *result, mgp_memory *memory);

}  // namespace CsvUtils