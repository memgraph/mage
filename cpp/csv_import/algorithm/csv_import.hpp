#pragma once

#include <stdio.h>
#include <stdlib.h>
#include <filesystem>
#include <fstream>
#include <mgp.hpp>
namespace fs = std::filesystem;

namespace CsvImport {

constexpr std::string_view kFilepath = "filepath";

void CreateCsvFile(mgp_list *args, mgp_graph *memgraph_graph, mgp_result *result, mgp_memory *memory);
void DeleteCsvFile(mgp_list *args, mgp_graph *memgraph_graph, mgp_result *result, mgp_memory *memory);

}  // namespace CsvImport