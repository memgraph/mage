#include <mgp.hpp>
#include "mgclient.h"
#include <fmt/core.h>

const char *kProcedurePeriodic = "call";
const char *kProcedureFormatQuery = "format_query";
const char *kArgumentChunkSize = "chunk_size";
const char *kArgumentQuery = "query";
const char *kArgumentHost = "host";
const char *kArgumentPort = "port";
const char *kReturnStatus = "status";
const char *kReturnQuery = "query";


// CREATE (:Label{{prop:row}});
std::string format_query(const int num, const std::string &user_string){
  return fmt::format("CALL migration.yield_records({}) YIELD row {} ", std::to_string(num), user_string);
}

void MgClientCall(mgp_list *args, mgp_graph *memgraph_graph, mgp_result *result, mgp_memory *memory) {
  mgp::memory = memory;
  const auto arguments = mgp::List(args);
  const auto num = arguments[0].ValueInt();
  const auto user_query = std::string(arguments[1].ValueString());
  const auto host = std::string(arguments[2].ValueString());
  const auto port = arguments[3].ValueInt();
  const auto record_factory = mgp::RecordFactory(result);
  auto record = record_factory.NewRecord();
  try{
    mg_session *session = NULL;
    mg_init();
    mg_session_params *params = mg_session_params_make();
    mg_session_params_set_host(params, host.c_str());
    mg_session_params_set_port(params, port);
    mg_session_params_set_sslmode(params, MG_SSLMODE_DISABLE);
    
    int status = mg_connect(params, &session);
    const std::string query = format_query(num, user_query);
    if (mg_session_run(session, query.c_str() , NULL, NULL, NULL, NULL) < 0) {
        mg_session_destroy(session);
        record.Insert(kReturnStatus, false);
        return;
    }

    if (mg_session_pull(session, NULL)) {
        mg_session_destroy(session);
        record.Insert(kReturnStatus, false);
        return;
    }

    mg_result *result;
    int rows = 0;
    while ((status = mg_session_fetch(session, &result)) == 1) {
        rows++;
    }

    mg_session_destroy(session);
    mg_finalize();
    record.Insert(kReturnStatus, true);
    return;
  }
  catch (const std::exception &e) {
    record_factory.SetErrorMessage(e.what());
    record.Insert(kReturnStatus, false);
    return;
  }
}


void MgClientGetQuery(mgp_list *args, mgp_graph *memgraph_graph, mgp_result *result, mgp_memory *memory) {
  mgp::memory = memory;
  const auto arguments = mgp::List(args);
  const auto num = arguments[0].ValueInt();
  const auto user_query = std::string(arguments[1].ValueString());
  const auto record_factory = mgp::RecordFactory(result);
  auto record = record_factory.NewRecord();
  try{
    const std::string formatted_query = format_query(num, user_query);
    record.Insert(kReturnQuery, formatted_query);
    return;
  }
  catch (const std::exception &e) {
    record_factory.SetErrorMessage(e.what());
    record.Insert(kReturnQuery, "Can't format query.");
    return;
  }
}



extern "C" int mgp_init_module(struct mgp_module *module, struct mgp_memory *memory) {
    try{
        mgp::memory = memory;
        mgp::AddProcedure(MgClientCall, kProcedurePeriodic, mgp::ProcedureType::Read, 
        {mgp::Parameter(kArgumentChunkSize, mgp::Type::Int), mgp::Parameter(kArgumentQuery, mgp::Type::String),
        mgp::Parameter(kArgumentHost, mgp::Type::String), mgp::Parameter(kArgumentPort, mgp::Type::Int)}, {mgp::Return(kReturnStatus, mgp::Type::Bool)}, module, memory);

        mgp::AddProcedure(MgClientGetQuery, kProcedureFormatQuery, mgp::ProcedureType::Read, 
        {mgp::Parameter(kArgumentChunkSize, mgp::Type::Int), mgp::Parameter(kArgumentQuery, mgp::Type::String)}, {mgp::Return(kReturnQuery, mgp::Type::String)}, module, memory);
    } catch (const std::exception &e) {
    return 1;
    }
    return 0;
}

extern "C" int mgp_shutdown_module() { return 0; }