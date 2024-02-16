
#ifndef DATEHELPER_H
#define DATEHELPER_H

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


class DateHelper {
   public :
      static std::string dateAdd(int64_t epoch_time, const std::string &timeSpanType, int timeSpan, const std::string &strftime_format, const std::string &timezone = ""); 
 
      /// @brief format the 
      /// @param strftime_format 
      /// @return 
      static std::string convertDateFormatToCPP(const std::string &strftime_format);


      /// @brief convert the time to formatted date string
      /// @param time 
      /// @param strftime_format 
      /// @param timezone 
      /// @return 
      static std::string convertTimeToString(std::time_t time, const std::string &strftime_format, const std::string &timezone = "");


      /// @brief return a formatted date string from the Epoch time in microseconds
      /// @param epoch_time 
      /// @param strftime_format 
      /// @param timezone 
      /// @return 
      static std::string convertEpochMicrosecToString(int64_t epoch_time, const std::string &strftime_format, const std::string &timezone = "");


      /// @brief get the date time formatted string from Epoch time (Seconds)
      /// @param epoch_time 
      /// @param strftime_format 
      /// @param timezone 
      /// @return 
      static std::string convertEpochSecToString(int64_t epoch_time, const std::string &strftime_format, const std::string &timezone = "");


      /// @brief convert Epoch time from Seconds to Microseconds
      /// @param epoch_time 
      /// @return 
      static std::int64_t convertEpochSecToMicrosec(int64_t epoch_time);


      /// @brief convert Epoch time from Microseconds to Seconds
      /// @param epoch_time 
      /// @return 
      static std::int64_t convertEpochMicrosecToSec(int64_t epoch_time);



      /// @brief return a formatted string based on given epoch Time
      /// @param epochTime 
      /// @param format 
      /// @return 
      static std::string getTimeStamp(time_t epochTime, const char* format = "%Y-%m-%d %H:%M:%S");


      /// @brief return the epoch time based on the date in string
      /// @param theTime 
      /// @param format 
      /// @return 
      static time_t convertTimeToEpoch(const char* theTime, const char* format = "%Y-%m-%d %H:%M:%S");


      /// @brief convert the date string into time stamp
      /// @param dateTime 
      /// @param dateTimeFormat 
      /// @return 
      static std::tm convertDateTime(const char* theTime, const char* format);

      static std::tm convertDateTime2(const std::string& dateTime, const std::string& dateTimeFormat);

};


#endif