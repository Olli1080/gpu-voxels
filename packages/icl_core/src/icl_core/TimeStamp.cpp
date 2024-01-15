// this is for emacs file handling -*- mode: c++; indent-tabs-mode: nil -*-

// -- BEGIN LICENSE BLOCK ----------------------------------------------
// This program is free software licensed under the CDDL
// (COMMON DEVELOPMENT AND DISTRIBUTION LICENSE Version 1.0).
// You can find a copy of this license in LICENSE in the top
// directory of the source code.
//
// © Copyright 2018 FZI Forschungszentrum Informatik, Karlsruhe, Germany
// -- END LICENSE BLOCK ------------------------------------------------

//----------------------------------------------------------------------
/*!\file
 *
 * \author  Kay-Ulrich Scholl <scholl@fzi.de>
 * \author  Klaus Uhl <uhl@fzi.de>
 * \date    2006-06-10
 */
//----------------------------------------------------------------------

#include "icl_core/TimeStamp.h"

namespace icl_core {

    const TimeStamp TimeStamp::cZERO(0, 0);

    TimeStamp TimeStamp::now()
    {
        return std::chrono::system_clock::now();
    }

    TimeStamp TimeStamp::futureMSec(uint64_t msec)
    {
        return now() + std::chrono::milliseconds(msec);
    }

    TimeStamp TimeStamp::fromIso8601BasicUTC(const std::string& str)
    {
        int32_t tm_sec = 0;
        int32_t tm_min = 0;
        int32_t tm_hour = 0;
        int32_t tm_mday = 1;
        int32_t tm_mon = 1;
        int32_t tm_year = 1970;
        if (str.size() >= 4)
        {
            tm_year = std::stoi(str.substr(0, 4));
        }
        if (str.size() >= 6)
        {
            tm_mon = std::stoi(str.substr(4, 2));
        }
        if (str.size() >= 8)
        {
            tm_mday = std::stoi(str.substr(6, 2));
        }
        // Here comes the 'T', which we ignore and skip
        if (str.size() >= 11)
        {
            tm_hour = std::stoi(str.substr(9, 2));
        }
        if (str.size() >= 13)
        {
            tm_min = std::stoi(str.substr(11, 2));
        }
        if (str.size() >= 15)
        {
            tm_sec = std::stoi(str.substr(13, 2));
        }
        uint32_t nsecs = 0;
        // Here comes the comma, which we ignore and skip
        if (str.size() > 16)
        {
            std::string nsec_str = (str.substr(16, 9) + "000000000").substr(0, 9);
            nsecs = std::stoi(nsec_str);
        }

        // Jump to beginning of given year
        uint64_t days_since_epoch = 0;
        for (int32_t y = 1970; y < tm_year; ++y)
        {
            bool leap_year = (y % 400 == 0
                || (y % 4 == 0
                    && y % 100 != 0));
            days_since_epoch += leap_year ? 366 : 365;
        }
        // Now add months in that year
        bool leap_year = (tm_year % 400 == 0
            || (tm_year % 4 == 0
                && tm_year % 100 != 0));
        int32_t days_per_month[] = { 31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31 };
        if (leap_year)
        {
            days_per_month[1] = 29;
        }
        for (int32_t m = 1; m < tm_mon; ++m)
        {
            days_since_epoch += days_per_month[m - 1];
        }
        // Add day of month
        days_since_epoch += tm_mday - 1;
        uint64_t secs = 86400 * days_since_epoch + 3600 * tm_hour + 60 * tm_min + tm_sec;

        return { secs, nsecs };
    }

    void TimeStamp::strfTime(char* dest, size_t max_len, const char* format) const
    {
        time_t time = std::chrono::system_clock::to_time_t(internal_timestamp);
        auto newtime = gmtime(&time);
        strftime(dest, max_len, format, newtime);
    }

    void TimeStamp::strfLocaltime(char* dest, size_t max_len, const char* format) const
    {
        time_t time = std::chrono::system_clock::to_time_t(internal_timestamp);
        auto newtime = localtime(&time);
        if (newtime)
            strftime(dest, max_len, format, newtime);
    }

    std::string TimeStamp::formatIso8601() const
    {
        char date_time_sec[20];
        strfLocaltime(date_time_sec, 20, "%Y-%m-%d %H:%M:%S");
        return { date_time_sec };
    }

    std::string TimeStamp::formatIso8601UTC() const
    {
        char date_time_sec[20];
        strfTime(date_time_sec, 20, "%Y-%m-%d %H:%M:%S");
        return { date_time_sec };
    }

    std::string TimeStamp::formatIso8601Basic() const
    {
        char date_time_sec[16], date_time_nsec[10];
        TimeStamp adjust_nsec(*this);

        adjust_nsec.strfLocaltime(date_time_sec, 16, "%Y%m%dT%H%M%S");
        std::snprintf(date_time_nsec, 10, "%09lli", adjust_nsec.tsNSec().count());
        return std::string(date_time_sec) + "," + std::string(date_time_nsec);
    }

    std::string TimeStamp::formatIso8601BasicUTC() const
    {
        char date_time_sec[16], date_time_nsec[10];
        TimeStamp adjust_nsec(*this);

        adjust_nsec.strfTime(date_time_sec, 16, "%Y%m%dT%H%M%S");
        std::snprintf(date_time_nsec, 10, "%09lli", adjust_nsec.tsNSec().count());
        return std::string(date_time_sec) + "," + std::string(date_time_nsec);
    }
}