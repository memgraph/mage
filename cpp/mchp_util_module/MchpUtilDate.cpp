// Copyright 2023 Microchip Technology.
//
// Use of this software is governed by the Business Source License
// included in the file licenses/BSL.txt; by using this file, you agree to be bound by the terms of the Business Source
// License, and you may not use this file except in compliance with the Business Source License.
//
// As of the Change Date specified in that file, in accordance with
// the Business Source License, use of this software will be governed
// by the Apache License, Version 2.0, included in the file
// licenses/APL.txt.

#include <chrono>
#include <ctime>
#include <mgp.hpp>
#include "mchputil/DateHelper.hpp"
#include <string>


void convertToDate(mgp_list *args, mgp_func_context *ctx, mgp_func_result *res, mgp_memory *memory){
  mgp::MemoryDispatcherGuard guard{memory};


  std::vector<mgp::Value> arguments;
  for (size_t i = 0; i < mgp::list_size(args); i++){
    arguments.push_back(mgp::Value(mgp::list_at(args, i)));
  }

  auto result = mgp::Result(res);

  try{
    //date format
    std::string formatting{std::string(arguments[0].ValueString())};
    //date string
    std::string datetime{std::string(arguments[1].ValueString())};

    //convert date string to time
    std::tm t = DateHelper::convertDateTime(datetime.c_str(), formatting.c_str());
    //convert to localDateTime
    //mgp::LocalDateTime epochdatetime(t.tm_year+1900, t.tm_mon+1, t.tm_mday, t.tm_hour, t.tm_min, t.tm_sec, 0, 0);
    mgp::Date dt(t.tm_year+1900, t.tm_mon+1, t.tm_mday);

    result.SetValue(dt);

  }
  catch (const std::exception &e){
    result.SetErrorMessage(e.what());
    //result.SetValue("ERROR");
    //mgp::result_set_error_msg(result, e.what());
    return;
  }
}


/// @brief convert date string to LocalDateTime
/// @param args 
/// @param ctx 
/// @param res 
/// @param memory 
void convertToLocalDateTime(mgp_list *args, mgp_func_context *ctx, mgp_func_result *res, mgp_memory *memory){
  mgp::MemoryDispatcherGuard guard{memory};

  std::vector<mgp::Value> arguments;
  for (size_t i = 0; i < mgp::list_size(args); i++){
    arguments.push_back(mgp::Value(mgp::list_at(args, i)));
  }

  auto result = mgp::Result(res);

  try{
    //date format
    std::string formatting{arguments[0].ValueString()};
    //date string
    std::string datetime{arguments[1].ValueString()};

    //convert date string to time
    std::tm t = DateHelper::convertDateTime(datetime.c_str(), formatting.c_str());
    //convert to localDateTime
    mgp::LocalDateTime epochdatetime(t.tm_year+1900, t.tm_mon+1, t.tm_mday, t.tm_hour, t.tm_min, t.tm_sec, 0, 0);

    result.SetValue(epochdatetime);

  }
  catch (const std::exception &e){
    result.SetErrorMessage(e.what());
    //result.SetValue("ERROR");
    //mgp::result_set_error_msg(result, e.what());
    return;
  }
 
}

/// @brief convert date string to Epoch Microseconds
/// @param args 
/// @param ctx 
/// @param res 
/// @param memory 
void convertToEpochTime(mgp_list *args, mgp_func_context *ctx, mgp_func_result *res, mgp_memory *memory){
  mgp::MemoryDispatcherGuard guard{memory};

  mgp::List arguments{args};


  auto result = mgp::Result(res);

  try{

    //date format
    std::string formatting{arguments[0].ValueString()};
    //date string
    std::string datetime{arguments[1].ValueString()};

    int64_t t = DateHelper::convertTimeToEpoch(datetime.c_str(), formatting.c_str());
    //convert to Epoch time in microsecond
    t = DateHelper::convertEpochSecToMicrosec(t);

    result.SetValue(t);
  }
  catch (const std::exception &e){
    result.SetErrorMessage(e.what());
    return;
  }
}

/// @brief convert localDateTime to date string
/// @param args 
/// @param ctx 
/// @param res 
/// @param memory 
void convertDateTime(mgp_list *args, mgp_func_context *ctx, mgp_func_result *res, mgp_memory *memory){
  mgp::MemoryDispatcherGuard guard{memory};

  mgp::List arguments{args};

  auto result = mgp::Result(res);

  try{
 
    if (!arguments[1].IsLocalDateTime()){
      result.SetValue("ERROR");
      return;
    }

    //date format
    std::string formatting{arguments[0].ValueString()};
    //epoch time
    mgp::LocalDateTime datetime = arguments[1].ValueLocalDateTime();

    //convert Epoch time to date string
    std::string formattedDate = DateHelper::convertEpochMicrosecToString(datetime.Timestamp(), formatting, "");

    result.SetValue(formattedDate);
  }
  catch (const std::exception &e){
    result.SetErrorMessage(e.what());
    return;
  }
}

/// @brief convert Epoch time in Microsecond to date string
/// @param args 
/// @param ctx 
/// @param res 
/// @param memory 
void convertEpochTime(mgp_list *args, mgp_func_context *ctx, mgp_func_result *res, mgp_memory *memory){
  mgp::MemoryDispatcherGuard guard{memory};

  mgp::List arguments{args};

  auto result = mgp::Result(res);

  try{
    if (arguments[1].IsInt() == false){
      result.SetErrorMessage("ERROR");
      return;
    }
  
    //date format
    std::string formatting{arguments[0].ValueString()};
    //epoch time
    float epochTime = arguments[1].ValueInt();

    // convert Epoch time in microseconds to date string
    std::string formattedDate = DateHelper::convertEpochMicrosecToString(epochTime, formatting, "");

    result.SetValue(formattedDate);
  }
  catch (const std::exception &e)
  {
    result.SetErrorMessage(e.what());
    return;
  }
}

/// @brief Add minutes to Epoch time
/// @param args 
/// @param ctx 
/// @param res 
/// @param memory 
void dateAddMinutes(mgp_list *args, mgp_func_context *ctx, mgp_func_result *res, mgp_memory *memory){
  mgp::MemoryDispatcherGuard guard{memory};

  mgp::List arguments{args};


  auto result = mgp::Result(res);

  try{

    std::string formattedDate = "NA";

    std::string formatting{arguments[0].ValueString()};

    //epoch time
    if (!arguments[1].IsInt()){
      result.SetErrorMessage("ERROR");
      return;
    }
    //minutes to add
    if (!arguments[2].IsInt()) {
      result.SetErrorMessage("ERROR");
      return;
    }

    //get epoch time
    float epochTime = arguments[1].ValueInt();
    //get days to add
    int days = arguments[2].ValueInt();

    //return the date string with Added minutes
    result.SetValue(DateHelper::dateAdd(epochTime, "MINUTES", days, formatting, ""));
  }
  catch (const std::exception &e)
  {
    result.SetErrorMessage(e.what());
    return;
  }
}

/// @brief Add hours to Epoch time
/// @param args 
/// @param ctx 
/// @param res 
/// @param memory 
void dateAddHours(mgp_list *args, mgp_func_context *ctx, mgp_func_result *res, mgp_memory *memory){
  mgp::MemoryDispatcherGuard guard{memory};

  mgp::List arguments{args};


  auto result = mgp::Result(res);

  try{

    std::string formatting{arguments[0].ValueString()};

    if (arguments[1].IsInt() == false){
      result.SetErrorMessage("ERROR");
      return;
    }

    if (arguments[2].IsInt() == false){
      result.SetErrorMessage("ERROR");
      return;
    }


    float epochTime = arguments[1].ValueInt();

    int days = arguments[2].ValueInt();

    result.SetValue(DateHelper::dateAdd(epochTime, "HOURS", days, formatting, ""));
  }
  catch (const std::exception &e)
  {
    result.SetErrorMessage(e.what());
    return;
  }
}

/// @brief Add days to Epoch time
/// @param args 
/// @param ctx 
/// @param res 
/// @param memory 
void dateAddDays(mgp_list *args, mgp_func_context *ctx, mgp_func_result *res, mgp_memory *memory){
  mgp::MemoryDispatcherGuard guard{memory};

  mgp::List arguments{args};

  auto result = mgp::Result(res);

  try{

    std::string formatting{arguments[0].ValueString()};

    if (arguments[1].IsInt() == false){
      result.SetErrorMessage("ERROR");
      return;
    }

    if (arguments[2].IsInt() == false){
      result.SetErrorMessage("ERROR");
      return;
    }

    float epochTime = arguments[1].ValueInt();

    int days = arguments[2].ValueInt();

    result.SetValue(DateHelper::dateAdd(epochTime, "DAYS", days, formatting, ""));
  }
  catch (const std::exception &e)
  {
    result.SetErrorMessage(e.what());
    return;
  }
}

extern "C" int mgp_init_module(struct mgp_module *module, struct mgp_memory *memory){
  mgp::MemoryDispatcherGuard guard{memory};
  try{
    mgp::AddFunction(convertToLocalDateTime, "convertToLocalDateTime",
                     {mgp::Parameter("format", mgp::Type::String), mgp::Parameter("date", mgp::Type::String)}, module,
                     memory);
    
    mgp::AddFunction(convertToDate, "convertToDate",
                     {mgp::Parameter("format", mgp::Type::String), mgp::Parameter("date", mgp::Type::String)}, module,
                     memory);

     mgp::AddFunction(convertToEpochTime, "convertToEpochTime",
                     {mgp::Parameter("format", mgp::Type::String), mgp::Parameter("date", mgp::Type::String)}, module,
                     memory);

    mgp::AddFunction(convertDateTime, "convertDateTime",
                     {mgp::Parameter("format", mgp::Type::String), mgp::Parameter("localdatetime", mgp::Type::LocalDateTime)}, module,
                     memory);

     mgp::AddFunction(convertEpochTime, "convertEpochTime",
                     {mgp::Parameter("format", mgp::Type::String), mgp::Parameter("epochTime", mgp::Type::Int)}, module,
                     memory);

    mgp::AddFunction(dateAddDays, "dateAddDays",
                     {mgp::Parameter("format", mgp::Type::String), mgp::Parameter("epochTime", mgp::Type::Int), mgp::Parameter("days", mgp::Type::Int)}, module,
                     memory);
    mgp::AddFunction(dateAddHours, "dateAddHours",
                     {mgp::Parameter("format", mgp::Type::String), mgp::Parameter("epochTime", mgp::Type::Int), mgp::Parameter("hours", mgp::Type::Int)}, module,
                     memory);

     mgp::AddFunction(dateAddMinutes, "dateAddMinutes",
                     {mgp::Parameter("format", mgp::Type::String), mgp::Parameter("epochTime", mgp::Type::Int), mgp::Parameter("minutes", mgp::Type::Int)}, module,
                     memory);
  }
  catch (const std::exception &e){
    return 1;
  }

  return 0;
}

extern "C" int mgp_shutdown_module() { return 0; }
