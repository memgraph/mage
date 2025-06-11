#pragma once

#include <string>
#include <string_view>

#include <mgp.hpp>

namespace Text {

/* join constants */
constexpr std::string_view kProcedureJoin = "join";
constexpr std::string_view kJoinArg1 = "strings";
constexpr std::string_view kJoinArg2 = "delimiter";
constexpr std::string_view kResultJoin = "join";
/* format constants */
constexpr std::string_view kProcedureFormat = "format";
constexpr std::string_view kStringToFormat = "format";
constexpr std::string_view kParameters = "params";
constexpr std::string_view kResultFormat = "format";
/* regex constants */
constexpr std::string_view kProcedureRegexGroups = "regex_groups";
constexpr std::string_view kInput = "input";
constexpr std::string_view kRegex = "regex";
constexpr std::string_view kResultRegexGroups = "groups";
/* replace constants */
constexpr std::string_view kProcedureReplace = "replace";
constexpr std::string_view kText = "text";
constexpr std::string_view kRegexReplace = "search";
constexpr std::string_view kReplacement = "replacement";
constexpr std::string_view kResultReplace = "result";
/* regreplace constants */
constexpr std::string_view kProcedureRegReplace = "regreplace";
constexpr std::string_view kResultRegReplace = "result";
/* distance constants */
constexpr std::string_view kProcedureDistance = "distance";
constexpr std::string_view kText1 = "text1";
constexpr std::string_view kText2 = "text2";

void Join(mgp_list *args, mgp_graph *memgraph_graph, mgp_result *result, mgp_memory *memory);
void Format(mgp_list *args, mgp_graph *memgraph_graph, mgp_result *result, mgp_memory *memory);
void RegexGroups(mgp_list *args, mgp_graph *memgraph_graph, mgp_result *result, mgp_memory *memory);
void Replace(mgp_list *args, mgp_func_context *ctx, mgp_func_result *result, mgp_memory *memory);
void RegReplace(mgp_list *args, mgp_func_context *ctx, mgp_func_result *result, mgp_memory *memory);
void Distance(mgp_list *args, mgp_func_context *ctx, mgp_func_result *result, mgp_memory *memory);
}  // namespace Text
