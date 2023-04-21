#include <mgp.hpp>
#include "mgclient.h"
#include <fmt/core.h>

const char *kProcedureHacky = "hacky_migration_cpp";
const char *kArgumentChunkSize = "chunk_size";
const char *kReturnStatus = "status";


void GQLACall(mgp_list *args, mgp_graph *memgraph_graph, mgp_result *result, mgp_memory *memory) {
  mgp::memory = memory;
  const auto arguments = mgp::List(args);
  const auto num = arguments[0].ValueInt();
  const auto record_factory = mgp::RecordFactory(result);
  auto record = record_factory.NewRecord();
  std::cout << num << std::endl;
  try{
    mg_session *session = NULL;
    mg_init();
    mg_session_params *params = mg_session_params_make();
    mg_session_params_set_host(params, "localhost");
    mg_session_params_set_port(params, 7687);
    mg_session_params_set_sslmode(params, MG_SSLMODE_DISABLE);
    
    int status = mg_connect(params, &session);
    const std::string query = fmt::format("CALL hacky_migration.yield_records({}) YIELD row CREATE (:Label{{prop:row}});", std::to_string(num));
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


extern "C" int mgp_init_module(struct mgp_module *module, struct mgp_memory *memory) {
    try{
        mgp::memory = memory;
        mgp::AddProcedure(GQLACall, kProcedureHacky, mgp::ProcedureType::Read, 
        {mgp::Parameter(kArgumentChunkSize, mgp::Type::Int)}, {mgp::Return(kReturnStatus, mgp::Type::Bool)}, module, memory);
    } catch (const std::exception &e) {
    return 1;
    }
    return 0;
}

extern "C" int mgp_shutdown_module() { return 0; }