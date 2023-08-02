#pragma once

#include <mgp.hpp>

namespace Date {

constexpr std::string_view kReturnParsed = "parsed";

constexpr std::string_view kProcedureParse = "parse";

constexpr std::string_view kArgumentsTime = "time";
constexpr std::string_view kArgumentsUnit = "unit";
constexpr std::string_view kArgumentsFormat = "format";
constexpr std::string_view kArgumentsTimezone = "timezone";

constexpr std::string_view kResultParsed = "parsed";

void Parse(mgp_list *args, mgp_graph *memgraph_graph, mgp_result *result, mgp_memory *memory);

}  // namespace Date
