// this is for emacs file handling -*- mode: c++; indent-tabs-mode: nil -*-
//----------------------------------------------------------------------
/*!\file
 *
 * \author  Kay-Ulrich Scholl <scholl@fzi.de>
 * \author  Klaus Uhl <uhl@fzi.de>
 * \date    2006-06-10
 */
//----------------------------------------------------------------------
#include "icl_core_logging/ThreadStream.h"

#include "icl_core_logging/LogStream.h"
#include "icl_core_logging/LogOutputStream.h"

#include <cstdarg>
#include <cstring>

#if defined ICL_CORE_QT_SUPPORT
#include <QString>
#endif

namespace icl_core {
    namespace logging {

        void ThreadStream::setClassname(const char* classname)
        {
            strncpy(m_classname, classname, cDEFAULT_LOG_SIZE);
        }

        void ThreadStream::setFilename(const char* filename)
        {
            strncpy(m_filename, filename, cDEFAULT_LOG_SIZE);
        }

        void ThreadStream::setObjectname(const char* objectname)
        {
            strncpy(m_objectname, objectname, cDEFAULT_LOG_SIZE);
        }

        void ThreadStream::setFunction(const char* function)
        {
            strncpy(m_function, function, cDEFAULT_LOG_SIZE);
        }

        void ThreadStream::setLine(size_t line)
        {
            m_line = line;
        }

        void ThreadStream::setLineLogLevel(icl_core::logging::LogLevel line_log_level)
        {
            m_line_log_level = line_log_level;
        }

        void ThreadStream::write(const char* source, size_t number_of_bytes, size_t protected_buffer_size)
        {
            // Protect the last byte in the thread stream's buffer!
            const size_t writable_length = cDEFAULT_LOG_SIZE - m_write_index - 1;
            if (number_of_bytes + protected_buffer_size > writable_length)
            {
                if (writable_length > protected_buffer_size)
                {
                    number_of_bytes = writable_length - protected_buffer_size;
                }
                else
                {
                    number_of_bytes = 0;
                }
            }
            memcpy(&m_data[m_write_index], source, number_of_bytes);

            m_write_index += number_of_bytes;
        }

        void ThreadStream::printf(char const* const fmt, ...)
        {
            // Protect the last byte in the thread stream's buffer!
            const size_t writable_length = cDEFAULT_LOG_SIZE - m_write_index - 1;

            va_list argptr;
            va_start(argptr, fmt);
            const int32_t bytes_printed = vsnprintf(&m_data[m_write_index], writable_length, fmt, argptr);
            va_end(argptr);

            if (bytes_printed >= 0)
            {
                if (static_cast<size_t>(bytes_printed) > writable_length)
                {
                    m_write_index += writable_length;
                }
                else
                {
                    m_write_index += bytes_printed;
                }
            }

            flush();
        }

        void ThreadStream::flush()
        {
            m_data[m_write_index] = '\0';
            {
                std::unique_lock lock(m_parent->m_mutex);
                for (const auto iter : m_parent->m_output_stream_list)
                {
                    iter->push(m_line_log_level, m_parent->nameCStr(), m_filename, m_line,
                        m_classname, m_objectname, m_function, m_data);
                }

                m_parent->releaseThreadStream(this);
            }
            m_write_index = 0;
        }

        ThreadStream::ThreadStream(LogStream* parent)
            : m_parent(parent),
            m_level(parent->m_initial_level),
            m_line_log_level(parent->m_initial_level),
            m_line(0),
            m_write_index(0)
        {
            memset(m_classname, 0, cDEFAULT_LOG_SIZE + 1);
            memset(m_function, 0, cDEFAULT_LOG_SIZE + 1);
            memset(m_data, 0, cDEFAULT_LOG_SIZE + 1);
        }


        ThreadStream& operator << (ThreadStream& stream, uint8_t value)
        {
            char local_buffer[8];
            const size_t length = snprintf(local_buffer, 8, "%hu", value);
            stream.write(local_buffer, length);
            return stream;
        }

        ThreadStream& operator << (ThreadStream& stream, uint16_t value)
        {
            char local_buffer[8];
            const size_t length = snprintf(local_buffer, 8, "%hu", value);
            stream.write(local_buffer, length);
            return stream;
        }

        ThreadStream& operator << (ThreadStream& stream, uint32_t value)
        {
            char local_buffer[13];
            const size_t length = snprintf(local_buffer, 13, "%u", value);
            stream.write(local_buffer, length);
            return stream;
        }

#if __WORDSIZE != 64
        ThreadStream& operator << (ThreadStream& stream, unsigned long value)
        {
            char local_buffer[13];
            const size_t length = snprintf(local_buffer, 13, "%lu", value);
            stream.write(local_buffer, length);
            return stream;
        }
#endif

        ThreadStream& operator << (ThreadStream& stream, uint64_t value)
        {
            char local_buffer[23];
            const size_t length = snprintf(local_buffer, 23, "%llu", value);
            stream.write(local_buffer, length);
            return stream;
        }

        ThreadStream& operator << (ThreadStream& stream, int8_t value)
        {
            char local_buffer[8];
            const size_t length = snprintf(local_buffer, 8, "%hd", static_cast<int16_t>(value));
            stream.write(local_buffer, length);
            return stream;
        }

        ThreadStream& operator << (ThreadStream& stream, int16_t value)
        {
            char local_buffer[8];
            const size_t length = snprintf(local_buffer, 8, "%hd", value);
            stream.write(local_buffer, length);
            return stream;
        }

        ThreadStream& operator << (ThreadStream& stream, int32_t value)
        {
            char local_buffer[13];
            const size_t length = snprintf(local_buffer, 13, "%d", value);
            stream.write(local_buffer, length);
            return stream;
        }

#if __WORDSIZE != 64
        ThreadStream& operator << (ThreadStream& stream, long value)
        {
            char local_buffer[13];
            const size_t length = snprintf(local_buffer, 13, "%ld", value);
            stream.write(local_buffer, length);
            return stream;
        }
#endif

        ThreadStream& operator << (ThreadStream& stream, int64_t value)
        {
            char local_buffer[23];
            const size_t length = snprintf(local_buffer, 23, "%lld", value);
            stream.write(local_buffer, length);
            return stream;
        }

        ThreadStream& operator << (ThreadStream& stream, const char* text)
        {
            stream.write(text, strlen(text));
            return stream;
        }

        ThreadStream& operator << (ThreadStream& stream, const std::string& text)
        {
            stream.write(text.c_str(), text.size());
            return stream;
        }

        ThreadStream& operator << (ThreadStream& stream, double value)
        {
            char local_buffer[100];
            const size_t length = snprintf(local_buffer, 100, "%f", value);
            stream.write(local_buffer, length);
            return stream;
        }

        ThreadStream& operator << (ThreadStream& stream, bool value)
        {
            if (value)
            {
                stream.write("True", 4);
            }
            else
            {
                stream.write("False", 5);
            }
            return stream;
        }

        ThreadStream& operator << (ThreadStream& stream, void* value)
        {
            char local_buffer[25];
            const size_t length = snprintf(local_buffer, 25, "%p", value);
            stream.write(local_buffer, length);
            return stream;
        }

#if defined ICL_CORE_QT_SUPPORT
        ThreadStream& operator << (ThreadStream& stream, const QString& value)
        {
            return operator << (stream, value.toLatin1().constData());
        }
#endif

    }
}