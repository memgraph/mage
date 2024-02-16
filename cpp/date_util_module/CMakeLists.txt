set(util_date_src
    UtilDate.cpp
    util/DateHelper.cpp)


add_query_module(UtilDate 1 "${util_date_src}")
