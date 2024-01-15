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
 * \date    2006-09-04
 *
 * \brief   Contains TimeStamp
 *
 * \b tTime
 *
 * Contains the definitions of a generic interface  to the system time
 *
 */
//----------------------------------------------------------------------
#ifndef ICL_CORE_TIME_STAMP_H_INCLUDED
#define ICL_CORE_TIME_STAMP_H_INCLUDED

#include <string>
#include <chrono>

#include "icl_core/ImportExport.h"

namespace icl_core {

#if __cplusplus < 202002L
    using days = std::chrono::duration<int, std::ratio_multiply<std::ratio<24>, std::chrono::hours::period>>;
#else
    using days = std::chrono::days;
#endif

    //! Represents absolute times.
    /*! Use this class whenever you want to deal with times, as it
     *  provides a number of useful operators and functions.
     */
    class ICL_CORE_IMPORT_EXPORT TimeStamp
    {
    public:
        //! Standard constructor, creates a null time.
        TimeStamp() = default;

        //! Constructor, takes a timeval for creation.
        //TimeStamp(timeval time) { secs = time.tv_sec; nsecs = time.tv_usec * 1000; }

        //! Constructor that gets a time in seconds plus nanoseconds.
        TimeStamp(uint64_t sec, uint32_t nsec)
            : internal_timestamp(std::chrono::system_clock::time_point(std::chrono::duration_cast<std::chrono::system_clock::duration>(std::chrono::seconds(sec) + std::chrono::nanoseconds(nsec))))
        { }

        /*
      TimeStamp(const struct timespec& ts)
        : TimeBase(ts)
      { }
      */
        explicit TimeStamp(time_t timestamp)
            : internal_timestamp(std::chrono::seconds(timestamp))
        { }

        TimeStamp(std::chrono::system_clock::time_point tp)
            : internal_timestamp(tp)
        { }

        /*! This static function returns a TimeStamp that contains the
         *  current System time (as UTC).
         */
        static TimeStamp now();

        //! Returns a time stamp which lies \a msec ms in the future.
        static TimeStamp futureMSec(uint64_t msec);

        /*! Returns a time stamp parsed from an ISO 8601 basic UTC timestamp
         *  (YYYYMMDDTHHMMSS,fffffffff).
         */
        static TimeStamp fromIso8601BasicUTC(const std::string& str);

        /*! Return a formatted time std::string.
         *  \note While TimeStamp uses a 64-bit unsigned integer to store
         *        the seconds, the time formatting methods only support
         *        32-bit signed integers and will therefore render
         *        TimeStamps beyond 03:14:08 UTC on 19 January 2038
         *        incorrectly.
         */
        void strfTime(char* dest, size_t max_len, const char* format) const;
        /*! Return a formatted time std::string converted to the local timezone.
         *  \note While TimeStamp uses a 64-bit unsigned integer to store
         *        the seconds, the time formatting methods only support
         *        32-bit signed integers and will therefore render
         *        TimeStamps beyond 03:14:08 UTC on 19 January 2038
         *        incorrectly.
         */
        void strfLocaltime(char* dest, size_t max_len, const char* format) const;
        /*! Return the TimeStamp as a std::string in ISO 8601 format, in the
         *  local timezone.
         *  \note While TimeStamp uses a 64-bit unsigned integer to store
         *        the seconds, the time formatting methods only support
         *        32-bit signed integers and will therefore render
         *        TimeStamps beyond 03:14:08 UTC on 19 January 2038
         *        incorrectly.
         */
        [[nodiscard]] std::string formatIso8601() const;
        /*! Return the TimeStamp as a std::string in ISO 8601 format, in UTC.
         *  \note While TimeStamp uses a 64-bit unsigned integer to store
         *        the seconds, the time formatting methods only support
         *        32-bit signed integers and will therefore render
         *        TimeStamps beyond 03:14:08 UTC on 19 January 2038
         *        incorrectly.
         */
        [[nodiscard]] std::string formatIso8601UTC() const;
        /*! Return the TimeStamp as a std::string in the ISO 8601 basic format
         *  (YYYYMMDDTHHMMSS,fffffffff), in the local timezone.
         *  \note While TimeStamp uses a 64-bit unsigned integer to store
         *        the seconds, the time formatting methods only support
         *        32-bit signed integers and will therefore render
         *        TimeStamps beyond 03:14:08 UTC on 19 January 2038
         *        incorrectly.
         */
        [[nodiscard]] std::string formatIso8601Basic() const;
        /*! Return the TimeStamp as a std::string in the ISO 8601 basic format
         *  (YYYYMMDDTHHMMSS,fffffffff), in UTC.
         *  \note While TimeStamp uses a 64-bit unsigned integer to store
         *        the seconds, the time formatting methods only support
         *        32-bit signed integers and will therefore render
         *        TimeStamps beyond 03:14:08 UTC on 19 January 2038
         *        incorrectly.
         */
        [[nodiscard]] std::string formatIso8601BasicUTC() const;

        //! Adds a TimeSpan.
        TimeStamp& operator += (const std::chrono::system_clock::duration& span)
        {
            internal_timestamp += span;
            return *this;
        }

        //! Substracts a TimeSpan.
        TimeStamp& operator -= (const std::chrono::system_clock::duration& span)
        {
            internal_timestamp -= span;
            return *this;
        }

        /*! Compares two variables of type TimeStamp.
         *  \returns \c true if they are not equal.
         */
        bool operator != (const TimeStamp& other) const
        {
            return internal_timestamp != other.internal_timestamp;
        }

        /*! Compares two variables of type TimeStamp.
         *  \returns \c true if they are equal.
         */
        bool operator == (const TimeStamp& other) const
        {
            return internal_timestamp == other.internal_timestamp;
        }

        /*! Compares two variables of type TimeStamp.
         *  \returns \c true if the first one is earlier than the second
         *           one.
         */
        bool operator < (const TimeStamp& other) const
        {
            return internal_timestamp < other.internal_timestamp;
        }

        /*! Compares two variables of type TimeStamp.
         *  \returns \c true if the first one is later than the second one.
         */
        bool operator > (const TimeStamp& other) const
        {
            return internal_timestamp > other.internal_timestamp;
        }

        /*! Compares two variables of type TimeStamp.
         *  \returns \c true if the first one is earlier than or equal to
         *           the second one.
         */
        bool operator <= (const TimeStamp& other) const
        {
            return internal_timestamp <= other.internal_timestamp;
        }

        /*! Compares two variables of type TimeStamp.
         *  \returns \c true if the first one is later than or equal to the
         *           second one.
         */
        bool operator >= (const TimeStamp& other) const
        {
            return internal_timestamp >= other.internal_timestamp;
        }

        [[nodiscard]] std::chrono::seconds tsSec() const { return std::chrono::duration_cast<std::chrono::seconds>(internal_timestamp.time_since_epoch()); }
        [[nodiscard]] std::chrono::milliseconds tsMSec() const { return std::chrono::duration_cast<std::chrono::milliseconds>(internal_timestamp.time_since_epoch()) - tsSec(); }
        [[nodiscard]] std::chrono::microseconds tsUSec() const { return std::chrono::duration_cast<std::chrono::microseconds>(internal_timestamp.time_since_epoch()) - tsMSec(); }
        [[nodiscard]] std::chrono::nanoseconds tsNSec() const { return internal_timestamp.time_since_epoch() - tsUSec(); }
        [[nodiscard]] days tsNDays() const { return std::chrono::duration_cast<days>(internal_timestamp.time_since_epoch()); }
        [[nodiscard]] std::chrono::system_clock::time_point getInternal() const { return internal_timestamp; }

        static const TimeStamp cZERO;

    private:

        std::chrono::system_clock::time_point internal_timestamp{};
    };


    inline TimeStamp operator + (const std::chrono::system_clock::duration& span, const TimeStamp& time)
    {
        TimeStamp a(time);
        return a += span;
    }

    inline TimeStamp operator + (const TimeStamp& time, const std::chrono::system_clock::duration& span)
    {
        TimeStamp a(time);
        return a += span;
    }

    inline TimeStamp operator - (const TimeStamp& time, const std::chrono::system_clock::duration& span)
    {
        TimeStamp a(time);
        return a -= span;
    }

    inline std::chrono::system_clock::duration operator - (const TimeStamp& time_1, const TimeStamp& time_2)
    {
        return time_1.getInternal() - time_2.getInternal();
    }

}
#endif