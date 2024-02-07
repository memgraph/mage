#include <iostream>
#include <vector>
#include <string>

#include <chrono>
#include <ctime>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <cstring>
#include <regex>
#include <ctime>
#include <iomanip>

#include "DateHelper.hpp"



/// @brief Add Days, Hours or Minutes to the given Epoch time; will not throw exception if timeSpanType is not support, instead default to DAYS
/// @param epoch_time 
/// @param timeSpanType 
/// @param timeSpan 
/// @param strftime_format 
/// @param timezone 
/// @return 
std::string DateHelper::dateAdd(int64_t epoch_time, const std::string &timeSpanType, int timeSpan, const std::string &strftime_format, const std::string &timezone) {
    std::chrono::microseconds microseconds(epoch_time);
    auto time_point = std::chrono::system_clock::from_time_t(std::chrono::duration_cast<std::chrono::seconds>(microseconds).count());

    std::string format =  DateHelper::convertDateFormatToCPP(strftime_format);

    if(timeSpanType == "MINUTES")
      //Minutes
      time_point += std::chrono::minutes(timeSpan);
    else {
      if(timeSpanType == "HOURS")
      {
        //HOURS
        time_point += std::chrono::hours(timeSpan);
      }
      else
      {
      //default to days
      time_point += std::chrono::hours(timeSpan * 24);
      }
    }

    std::time_t time = std::chrono::system_clock::to_time_t(time_point);

    std::tm* tm_info;
    if (timezone.empty()) {
        tm_info = std::localtime(&time);
    } else {
        tzset();
        setenv("TZ", timezone.c_str(), 1);
        tm_info = std::localtime(&time);
    }

    char buffer[100];
    std::strftime(buffer, sizeof(buffer), format.c_str(), tm_info);
    return buffer;
}


/// @brief convert the time to formatted date string
/// @param time 
/// @param strftime_format 
/// @param timezone 
/// @return 
std::string DateHelper::convertTimeToString(std::time_t time, const std::string &strftime_format, const std::string &timezone ) {

    std::tm* tm_info;
    std::string format =  DateHelper::convertDateFormatToCPP(strftime_format);

    //convert the time zone
    if (timezone.empty()) {
        tm_info = std::localtime(&time);
    } else {
        tzset();
        setenv("TZ", timezone.c_str(), 1);
        tm_info = std::localtime(&time);
    }
   //format the date string
    char buffer[100];
    std::strftime(buffer, sizeof(buffer), format.c_str(), tm_info);
    return buffer;

}


/// @brief convertd date format to CPP formatting 
/// @param strftime_format 
/// @return 
std::string DateHelper::convertDateFormatToCPP(const std::string &strftime_format){
   std::string cppFormat = strftime_format;

   //if % is found, assume that date format is CPP ready
   //otherwise update the format
   if(cppFormat.find('%') != std::string::npos)
      return strftime_format;
   
 
   //2 or 4 digit year
   cppFormat = std::regex_replace(cppFormat, std::regex("YYYY"), "%Y"); 
   cppFormat = std::regex_replace(cppFormat, std::regex("YY"), "%y"); 

   //2 digit month
   cppFormat = std::regex_replace(cppFormat, std::regex("MM"), "%m"); 
   //February
   cppFormat = std::regex_replace(cppFormat, std::regex("MONTH"), "%B"); 
   //Feb
   cppFormat = std::regex_replace(cppFormat, std::regex("MON"), "%b"); 

   //Week starting on Monday
   //week 01-53
   cppFormat = std::regex_replace(cppFormat, std::regex("WW"), "%W"); 

   //julian day
   cppFormat = std::regex_replace(cppFormat, std::regex("DDD"), "%j"); 

   //2 digit day
   cppFormat = std::regex_replace(cppFormat, std::regex("DD"), "%d"); 

   //Monday
   cppFormat = std::regex_replace(cppFormat, std::regex("DAY"), "%A"); 
   //Mon
   cppFormat = std::regex_replace(cppFormat, std::regex("DY"), "%a"); 

   //where Sunday is 0 (range [0-6])
   cppFormat = std::regex_replace(cppFormat, std::regex("D"), "%w"); 

   //2 digit Hour
   cppFormat = std::regex_replace(cppFormat, std::regex("HH24"), "%H"); 
   //24 hour clock (range [00-23])
   cppFormat = std::regex_replace(cppFormat, std::regex("HH"), "%I"); 

   //2 digit Minute 
   cppFormat = std::regex_replace(cppFormat, std::regex("MI"), "%M"); 

   //2 digit Second
   cppFormat = std::regex_replace(cppFormat, std::regex("SS"), "%S"); 
   //used with HH
   cppFormat = std::regex_replace(cppFormat, std::regex("AM\\|PM"), "%p"); 

   return cppFormat;
}

/// @brief return a formatted date string from the given Epoch time in microseconds
/// @param epoch_time 
/// @param strftime_format 
/// @param timezone 
/// @return 
std::string DateHelper::convertEpochMicrosecToString(int64_t epoch_time, const std::string &strftime_format, const std::string &timezone) {
    using namespace std::chrono;
    std::chrono::microseconds epochTimeMicroseconds(epoch_time);
    //convert the Microsecond to Second
    auto time_point = std::chrono::system_clock::from_time_t(std::chrono::duration_cast<std::chrono::seconds>(epochTimeMicroseconds).count());

    std::string format = DateHelper::convertDateFormatToCPP(strftime_format);

    std::time_t time = system_clock::to_time_t(time_point);
   //return the date string for the time
   return convertTimeToString(time, format, timezone);

}

/// @brief get the date time formatted string from Epoch time (Seconds)
/// @param epoch_time 
/// @param strftime_format 
/// @param timezone 
/// @return 
std::string DateHelper::convertEpochSecToString(int64_t epoch_time, const std::string &strftime_format, const std::string &timezone) {
    using namespace std::chrono;
    std::chrono::seconds seconds(epoch_time);
   //convert the Second to Microsecond
    auto time_point = std::chrono::system_clock::from_time_t(std::chrono::duration_cast<std::chrono::seconds>(seconds).count());
    std::time_t time = std::chrono::system_clock::to_time_t(time_point);

    std::string format = DateHelper::convertDateFormatToCPP(strftime_format);

   //return the date string for the time
   return convertTimeToString(time, format, timezone);
}

/// @brief convert Epoch time from Seconds to Microseconds
/// @param epoch_time 
/// @return 
std::int64_t DateHelper::convertEpochSecToMicrosec(int64_t epoch_time) {
    using namespace std::chrono;
    std::chrono::seconds seconds(epoch_time);

   return std::chrono::duration_cast<std::chrono::microseconds>(seconds).count();
}

/// @brief convert Epoch time from Microseconds to Seconds
/// @param epoch_time 
/// @return 
std::int64_t DateHelper::convertEpochMicrosecToSec(int64_t epoch_time) {
    using namespace std::chrono;
    std::chrono::microseconds microseconds(epoch_time);

   return std::chrono::duration_cast<std::chrono::seconds>(microseconds).count();
   
}


/// @brief return a formatted string based on given epoch Time
/// @param epochTime 
/// @param format 
/// @return 
std::string DateHelper::getTimeStamp(time_t epochTime, const char* format){
      char timestamp[64] = {0};
      std::string dateformat = DateHelper::convertDateFormatToCPP(format);
      strftime(timestamp, sizeof(timestamp), dateformat.c_str(), localtime(&epochTime));
      return timestamp;
}

   /// @brief return the epoch time based on the date in string
   /// @param theTime 
   /// @param format 
   /// @return 
time_t DateHelper::convertTimeToEpoch(const char* theTime, const char* format){
      std::tm tmTime;
      std::string dateformat = DateHelper::convertDateFormatToCPP(format);

      memset(&tmTime, 0, sizeof(tmTime));
      strptime(theTime, dateformat.c_str(), &tmTime);
      return mktime(&tmTime);
   }

/// @brief convert the date string to time
/// @param theTime 
/// @param format 
/// @return 
std::tm DateHelper::convertDateTime(const char* theTime, const char* format){
      std::tm tmTime;
      std::string dateformat = DateHelper::convertDateFormatToCPP(format);
      //std::string dateformat = format;

      memset(&tmTime, 0, sizeof(tmTime));
      strptime(theTime, dateformat.c_str(), &tmTime);

      //time_t t = mktime(&tmTime);
      //std::tm* tm_info = std::localtime(&t);
      //std::tm* tm_info = std::gmtime(&t);

      return tmTime;
}

/// @brief convert the date string into time stamp (Do NOT USE)
/// @param dateTime 
/// @param dateTimeFormat 
/// @return 
std::tm DateHelper::convertDateTime2(const std::string& dateTime, const std::string& dateTimeFormat){
   // Create a stream which we will use to parse the string,
   // which we provide to constructor of stream to fill the buffer.
   std::istringstream ss{ dateTime };

   // Create a tm object to store the parsed date and time.
   std::tm dt;

   //std::string dateformat = dateTimeFormat;
   std::string dateformat = DateHelper::convertDateFormatToCPP(dateTimeFormat);

   // Now we read from buffer using get_time manipulator
   // and formatting the input appropriately.
   ss >> std::get_time(&dt, dateformat.c_str());

   return dt;
}
