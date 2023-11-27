#pragma once

#include <mgp.hpp>

namespace Text {

/* join constants */
constexpr std::string_view kProcedureJoin = "join";
constexpr std::string_view kJoinArg1 = "strings";
constexpr std::string_view kJoinArg2 = "delimiter";
constexpr std::string_view kResultJoin = "string";
/* format constants */
constexpr std::string_view kProcedureFormat = "format";
constexpr std::string_view kStringToFormat = "text";
constexpr std::string_view kParameters = "params";
constexpr std::string_view kResultFormat = "result";
/* regex constants */
constexpr std::string_view kProcedureRegexGroups = "regexGroups";
constexpr std::string_view kInput = "input";
constexpr std::string_view kRegex = "regex";
constexpr std::string_view kResultRegexGroups = "results";

void Join(mgp_list *args, mgp_graph *memgraph_graph, mgp_result *result, mgp_memory *memory);
void Format(mgp_list *args, mgp_graph *memgraph_graph, mgp_result *result, mgp_memory *memory);
void RegexGroups(mgp_list *args, mgp_graph *memgraph_graph, mgp_result *result, mgp_memory *memory);
}  // namespace Text
