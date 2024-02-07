set(mchp_util_date_src
    MchpUtilDate.cpp
    mchputil/DateHelper.cpp)


add_query_module(MchpUtilDate 1 "${mchp_util_date_src}")
