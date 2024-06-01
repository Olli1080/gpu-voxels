// this is for emacs file handling -*- mode: c++; indent-tabs-mode: nil -*-
//----------------------------------------------------------------------
/*!\file
 *
 * \author  Kay-Ulrich Scholl <scholl@fzi.de>
 * \author  Klaus Uhl <uhl@fzi.de>
 * \date    2006-06-10
 *
 */
//----------------------------------------------------------------------
#include "LogOutputStream.h"

#include <cassert>
#include <cctype>
#include <cstring>
#include <iostream>

#include <icl_core_config/Config.h>

#include "LoggingManager.h"

namespace icl_core {
    namespace logging {

        const std::string LogOutputStream::m_default_log_format = "<~T.~3M> ~S(~L)~ C~(O~::D: ~E";

        LogOutputStream::LogOutputStream(std::string name,
                                         const std::string& config_prefix,
                                         icl_core::logging::LogLevel log_level,
                                         bool use_worker_thread)
            : m_name(std::move(name)),
            m_log_level(log_level),
            m_time_format("%F %X"),
            m_use_worker_thread(use_worker_thread)
        {
            LoggingManager::instance().assertInitialized();

            std::string log_format = m_default_log_format;
            icl_core::config::get<std::string>(config_prefix + "/Format", log_format);
            changeLogFormat(log_format.c_str());

            if (m_use_worker_thread)
            {
                //icl_core::ThreadPriority priority = m_default_worker_thread_priority;
                //icl_core::config::get<icl_core::ThreadPriority>(config_prefix + "/ThreadPriority", priority);

#ifdef ICL_CORE_LOG_OUTPUT_STREAM_USE_FIXED_QUEUE
                size_t message_queue_size = cDEFAULT_FIXED_OUTPUT_STREAM_QUEUE_SIZE;
                icl_core::config::get<size_t>(config_prefix + "/MessageQueueSize", message_queue_size);

                m_worker_thread = std::make_unique<WorkerThread>(this, message_queue_size);
#else
                m_worker_thread = std::make_unique<WorkerThread>(this);
#endif
            }
        }

        LogOutputStream::LogOutputStream(std::string name,
                                         icl_core::logging::LogLevel log_level,
                                         bool use_worker_thread)
            : m_name(std::move(name)),
            m_log_level(log_level),
            m_time_format("%F %X"),
            m_use_worker_thread(use_worker_thread)
        {
            LoggingManager::instance().assertInitialized();
            changeLogFormat(m_default_log_format.c_str());
            if (m_use_worker_thread)
            {
#ifdef ICL_CORE_LOG_OUTPUT_STREAM_USE_FIXED_QUEUE
                m_worker_thread = std::make_unique<WorkerThread>(this, cDEFAULT_FIXED_OUTPUT_STREAM_QUEUE_SIZE,
                    m_default_worker_thread_priority);
#else
                m_worker_thread = std::make_unique<WorkerThread>(this/*, m_default_worker_thread_priority*/);
#endif
            }
        }

        LogOutputStream::~LogOutputStream()
        {
            if (!m_use_worker_thread)
                return;
            
            if (m_worker_thread->running())
            {
                std::cerr << "WARNING: Destroyed LogOutputStream while thread is still alive. "
                    << "Please call Shutdown() before destruction." << '\n';
            }
            m_worker_thread.reset();
        }

        void LogOutputStream::changeLogFormat(const char* format)
        {
            // Stop processing at the end of the format string.
            if (format[0] == 0)
                return;

            parseLogFormat(format);
            {
                std::unique_lock lock(m_format_mutex);

                std::swap(m_log_format, m_new_log_format);
                m_new_log_format.clear();
            }
        }

        void LogOutputStream::push(icl_core::logging::LogLevel log_level,
            const char* log_stream_description, const char* filename,
            size_t line, const char* classname, const char* objectname,
            const char* function, const char* text)
        {
            if (log_level < getLogLevel())
                return;

            const LogMessage new_entry(std::chrono::system_clock::now(), log_level, log_stream_description,
                                       filename, line, classname, objectname, function, text);

            if (m_use_worker_thread)
            {
                // Hand the log text over to the output implementation.
                {
                    std::unique_lock lock(m_worker_thread->m_queue_mtx);
                    m_worker_thread->m_push_count_cv.wait(lock, [&]()
                        {
                            return !m_worker_thread->isMessageQueueFull() || !m_worker_thread->m_execute;
                        });

                    if (!m_worker_thread->m_execute)
                        return;

                    m_worker_thread->m_message_queue.push_back(new_entry);
                    m_worker_thread->m_fill_count_cv.notify_one();
                }
            }
            else
            {
                std::unique_lock lock(m_no_worker_thread_push_mutex);
                pushImpl(new_entry);
            }
        }

        void LogOutputStream::pushImpl(const LogMessage& log_message)
        {
            std::unique_lock lock(m_format_mutex);

            std::stringstream msg;
            for (const auto& it : m_log_format)
            {
                switch (it.type)
                {
                case LogFormatEntry::eT_TEXT:
                {
                    msg << it.text;
                    break;
                }
                case LogFormatEntry::eT_CLASSNAME:
                {
                    if (std::strcmp(log_message.class_name, "") != 0)
                    {
                        msg << it.text << log_message.class_name;
                    }
                    break;
                }
                case LogFormatEntry::eT_OBJECTNAME:
                {
                    if (std::strcmp(log_message.object_name, "") != 0)
                    {
                        msg << it.text << log_message.object_name << it.suffix;
                    }
                    break;
                }
                case LogFormatEntry::eT_FUNCTION:
                {
                    if (std::strcmp(log_message.function_name, "") != 0)
                    {
                        msg << it.text << log_message.function_name;
                    }
                    break;
                }
                case LogFormatEntry::eT_MESSAGE:
                {
                    msg << log_message.message_text;
                    break;
                }
                case LogFormatEntry::eT_FILENAME:
                {
                    msg << log_message.filename;
                    break;
                }
                case LogFormatEntry::eT_LINE:
                {
                    msg << log_message.line;
                    break;
                }
                case LogFormatEntry::eT_LEVEL:
                {
                    msg << logLevelDescription(log_message.log_level);
                    break;
                }
                case LogFormatEntry::eT_STREAM:
                {
                    msg << log_message.log_stream;
                    break;
                }
                case LogFormatEntry::eT_TIMESTAMP:
                {
                    msg << std::vformat("{:" + std::string(m_time_format) + "}", std::make_format_args(log_message.timestamp));
                    break;
                }
                case LogFormatEntry::eT_TIMESTAMP_MS:
                {
                    auto msec = std::chrono::duration_cast<std::chrono::milliseconds>(log_message.timestamp.time_since_epoch());
                    size_t msec_len = 1;
                    if (msec >= std::chrono::milliseconds(100))
                    {
                        msec_len = 3;
                    }
                    else if (msec >= std::chrono::milliseconds(10))
                    {
                        msec_len = 2;
                    }
                    for (size_t i = it.width; i > msec_len; --i)
                    {
                        msg << "0";
                    }
                    msg << msec;
                    break;
                }
                }
            }
            lock.unlock();
            pushImpl(msg.str());
        }

        void LogOutputStream::pushImpl(const std::string&)
        {
            std::cerr << "LOG OUTPUT STREAM ERROR: pushImpl() is not implemented!!!" << '\n';
        }

        void LogOutputStream::printConfiguration() const
        {
            std::cerr << "    " << name() << " : " << logLevelDescription(m_log_level) << '\n';
        }

        void LogOutputStream::parseLogFormat(const char* format)
        {
            LogFormatEntry new_entry;

            // The format string starts with a field specifier.
            if (format[0] == '~')
            {
                ++format;

                // Read the field width.
                while (format[0] != 0 && std::isdigit(format[0]))
                {
                    new_entry.width = 10 * new_entry.width + (format[0] - '0');
                    ++format;
                }

                // Read optional prefix text.
                char* prefix_ptr = new_entry.text;
                while (format[0] != 0 && format[0] != 'C' && format[0] != 'O' && format[0] != 'D'
                    && format[0] != 'E' && format[0] != 'F' && format[0] != 'G' && format[0] != 'L'
                    && format[0] != 'S' && format[0] != 'T' && format[0] != 'M')
                {
                    *prefix_ptr = format[0];
                    ++prefix_ptr;
                    ++format;
                }

                // Read the field type.
                if (format[0] == 'C')
                {
                    new_entry.type = LogFormatEntry::eT_CLASSNAME;
                }
                else if (format[0] == 'O')
                {
                    new_entry.type = LogFormatEntry::eT_OBJECTNAME;
                    if (new_entry.text[0] == '(')
                    {
                        std::strncpy(new_entry.suffix, ")", 100);
                    }
                    else if (new_entry.text[0] == '[')
                    {
                        std::strncpy(new_entry.suffix, "]", 100);
                    }
                    else if (new_entry.text[0] == '{')
                    {
                        std::strncpy(new_entry.suffix, "}", 100);
                    }
                }
                else if (format[0] == 'D')
                {
                    new_entry.type = LogFormatEntry::eT_FUNCTION;
                }
                else if (format[0] == 'E')
                {
                    new_entry.type = LogFormatEntry::eT_MESSAGE;
                }
                else if (format[0] == 'F')
                {
                    new_entry.type = LogFormatEntry::eT_FILENAME;
                }
                else if (format[0] == 'G')
                {
                    new_entry.type = LogFormatEntry::eT_LINE;
                }
                else if (format[0] == 'L')
                {
                    new_entry.type = LogFormatEntry::eT_LEVEL;
                }
                else if (format[0] == 'S')
                {
                    new_entry.type = LogFormatEntry::eT_STREAM;
                }
                else if (format[0] == 'T')
                {
                    new_entry.type = LogFormatEntry::eT_TIMESTAMP;
                }
                else if (format[0] == 'M')
                {
                    new_entry.type = LogFormatEntry::eT_TIMESTAMP_MS;
                }

                if (format[0] != 0)
                {
                    m_new_log_format.push_back(new_entry);
                }

                ++format;
            }
            else
            {
                char* text_ptr = new_entry.text;
                while (format[0] != '~' && format[0] != 0)
                {
                    *text_ptr = format[0];
                    ++text_ptr;
                    ++format;
                }

                if (new_entry.text[0] != 0)
                {
                    m_new_log_format.push_back(new_entry);
                }
            }

            // Stop processing at the end of the format string.
            if (format[0] == 0)
            {
                return;
            }
            else
            {
                parseLogFormat(format);
            }
        }

        void LogOutputStream::start()
        {
            if (!m_use_worker_thread)
                return;

            m_worker_thread->launch();
        }

        void LogOutputStream::shutdown()
        {
            if (m_use_worker_thread && m_worker_thread->running())
                m_worker_thread->stop();
        }
        
        LogOutputStream::WorkerThread::WorkerThread(LogOutputStream* output_stream)
            : m_output_stream(output_stream), m_max_queue_size(nullptr)
        {}

        LogOutputStream::WorkerThread::WorkerThread(LogOutputStream* output_stream, size_t message_queue_size)
            : m_output_stream(output_stream), m_max_queue_size(std::make_unique<size_t>(message_queue_size))
        {}

        LogOutputStream::WorkerThread::~WorkerThread()
        {
            stop();
        }

        void LogOutputStream::WorkerThread::run()
        {
            m_output_stream->onStart();

            // Wait for new messages to arrive.
            while (m_execute)
            {
                LogMessage log_message;
                {
                    //handle empty queue and pop of element
                    std::unique_lock lock(m_queue_mtx);
                    m_fill_count_cv.wait(lock, [&]()
                        {
                            return !isMessageQueueEmpty() || !m_execute;
                        });
                    if (!m_execute)
                    {
                        //just one single thread
                        //m_fill_count_cv.notify_one();
                        break;
                    }
                    log_message = m_message_queue.front();
                    m_message_queue.pop_front();
                }
                //notify outer stream producer
                m_push_count_cv.notify_one();
                //push to output
                m_output_stream->pushImpl(log_message);
            }
            {
                // Write out all remaining log messages.
                // Assuming no writes to m_message_queue anymore
                std::unique_lock lock(m_queue_mtx);
                while (!isMessageQueueEmpty())
                {
                    LogMessage log_message = m_message_queue.front();
                    m_message_queue.pop_front();
                    m_output_stream->pushImpl(log_message);
                }
            }
            m_output_stream->onShutdown();
        }

        void LogOutputStream::WorkerThread::launch()
        {
            m_execute = true;
            m_thread = std::make_unique<std::thread>([this]()
                {
                    run();
                    m_execute = false;
                    m_done = true;
                });
        }
        
        bool LogOutputStream::WorkerThread::isMessageQueueEmpty() const
        {
            return m_message_queue.empty();
        }

        bool LogOutputStream::WorkerThread::isMessageQueueFull() const
        {
            if (!m_max_queue_size)
                return false;

            return m_message_queue.size() >= *m_max_queue_size;
        }

        void LogOutputStream::WorkerThread::stop()
        {
	        const bool was_executing = std::atomic_exchange(&m_execute, false);
            if (!was_executing)
                return;

            m_fill_count_cv.notify_all();
            m_push_count_cv.notify_all();

            m_thread->join();
        }

        bool LogOutputStream::WorkerThread::running() const
        {
            return m_execute;
        }

        bool LogOutputStream::WorkerThread::done() const
        {
            return m_done;
        }

        LogOutputStream::LogMessage::LogMessage(const std::chrono::system_clock::time_point& timestamp,
            icl_core::logging::LogLevel log_level,
            const char* log_stream, const char* filename,
            size_t line, const char* class_name,
            const char* object_name, const char* function_name,
            const char* message_text)
            : timestamp(timestamp),
            log_level(log_level),
            line(line)
        {
            std::strncpy(LogMessage::log_stream, log_stream, cMAX_IDENTIFIER_LENGTH + 1);
            std::strncpy(LogMessage::filename, filename, cMAX_DESCRIPTION_LENGTH + 1);
            std::strncpy(LogMessage::class_name, class_name, cMAX_IDENTIFIER_LENGTH + 1);
            std::strncpy(LogMessage::object_name, object_name, cMAX_DESCRIPTION_LENGTH + 1);
            std::strncpy(LogMessage::function_name, function_name, cMAX_IDENTIFIER_LENGTH + 1);
            std::strncpy(LogMessage::message_text, message_text, cDEFAULT_LOG_SIZE + 1);
        }

    }
}