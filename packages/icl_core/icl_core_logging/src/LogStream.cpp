// this is for emacs file handling -*- mode: c++; indent-tabs-mode: nil -*-
//----------------------------------------------------------------------
/*!\file
 *
 * \author  Kay-Ulrich Scholl <scholl@fzi.de>
 * \author  Klaus Uhl <uhl@fzi.de>
 * \date    2006-06-10
 */
//----------------------------------------------------------------------
#include "LogStream.h"

#include <iostream>

#include "icl_core_logging/LoggingManager.h"
#include "icl_core_logging/LogOutputStream.h"
#include "icl_core_logging/ThreadStream.h"

namespace icl_core {
    namespace logging {
        
        LogStream::LogStream(std::string name, icl_core::logging::LogLevel initial_level)
            : m_initial_level(initial_level),
            m_name(std::move(name)),
            m_active(true)
        {
            LoggingManager::instance().assertInitialized();

            for (size_t i = 0; i < cDEFAULT_LOG_THREAD_STREAM_POOL_SIZE; ++i)
            {
                m_thread_stream_map.push_back(ThreadStreamInfo(m_empty_thread_id, eLL_MUTE,
                    new icl_core::logging::ThreadStream(this)));
            }
        }

        LogStream::~LogStream()
        {
            LoggingManager::instance().removeLogStream(m_name);

            ThreadStreamMap::const_iterator it = m_thread_stream_map.begin();
            for (; it != m_thread_stream_map.end(); ++it)
            {
                delete it->thread_stream;
            }
            m_thread_stream_map.clear();
        }

        icl_core::logging::LogLevel LogStream::getLogLevel() const
        {
            // TODO: Implement individual log levels for each thread.
            return m_initial_level;
        }

        void LogStream::addOutputStream(LogOutputStream* new_stream)
        {
            std::unique_lock lock(m_mutex);
            m_output_stream_list.insert(new_stream);
        }

        void LogStream::removeOutputStream(LogOutputStream* stream)
        {
            std::unique_lock lock(m_mutex);
            m_output_stream_list.erase(stream);
        }

        icl_core::logging::ThreadStream& LogStream::threadStream(icl_core::logging::LogLevel log_level)
        {
            icl_core::logging::ThreadStream* thread_stream = nullptr;

            std::unique_lock lock(m_mutex);

            const auto thread_id = std::this_thread::get_id();

            // Try to find the stream for the current thread, if it has already been assigned.
            for (const auto find_it : m_thread_stream_map)
            {
                if (find_it.thread_id == thread_id && find_it.log_level == log_level)
                {
                    thread_stream = find_it.thread_stream;
                    break;
                }
            }

            // Take a thread stream from the pool, if one is available.
            if (thread_stream == nullptr)
            {
                for (auto& find_it : m_thread_stream_map)
                {
                    if (find_it.thread_id == m_empty_thread_id)
                    {
                        find_it.thread_id = thread_id;
                        find_it.log_level = log_level;
                        thread_stream = find_it.thread_stream;
                        break;
                    }
                }
            }

            // There are no more threads streams available, so create a new one.
            if (thread_stream == nullptr)
            {
                thread_stream = new icl_core::logging::ThreadStream(this);
                m_thread_stream_map.push_back(ThreadStreamInfo(thread_id, log_level, thread_stream));
            }
            lock.unlock();

            // Set the log level for the thread stream.
            thread_stream->changeLevel(this->getLogLevel());

            return *thread_stream;
        }

        void LogStream::printConfiguration() const
        {
            for (const auto it : m_output_stream_list)
                std::cerr << it->name() << " ";
        }

        void LogStream::releaseThreadStream(const icl_core::logging::ThreadStream* thread_stream)
        {
            for (auto& find_it : m_thread_stream_map)
            {
                if (find_it.thread_stream == thread_stream)
                {
                    find_it.thread_id = m_empty_thread_id;
                    break;
                }
            }
        }
    }
}