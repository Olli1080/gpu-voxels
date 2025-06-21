// this is for emacs file handling -*- mode: c++; indent-tabs-mode: nil -*-
//----------------------------------------------------------------------
/*!\file
 *
 * \author  Klaus Uhl <uhl@fzi.de>
 * \date    2007-10-02
 *
 */
//----------------------------------------------------------------------
#include "icl_core_logging/LoggingManager.h"

#include <cassert>
#include <iostream>
#include <sstream>
#include <ranges>

//#include <icl_core/os_lxrt.h>
#include <icl_core_config/Config.h>
#include "icl_core_logging/FileLogOutput.h"
#include "icl_core_logging/LogStream.h"
#include "icl_core_logging/StdLogOutput.h"
#include "icl_core_config/GetoptParser.h"
#include "icl_core_config/GetoptParameter.h"

namespace icl_core {
    namespace logging {

        void LoggingManager::initialize()
        {
            if (!m_initialized)
            {
                m_initialized = true;

                // Read the log output stream configuration.
                ::icl_core::config::ConfigIterator output_stream_it =
                    ::icl_core::config::find(R"(\/IclCore\/Logging\/(OutputStream.*)\/(.*))");
                while (output_stream_it.next())
                {
                    ::std::string entry_name = output_stream_it.matchGroup(1);
                    ::std::string value_name = output_stream_it.matchGroup(2);
                    if (value_name == "OutputStreamName")
                    {
                        m_output_stream_config[entry_name].output_stream_name = output_stream_it.value();
                    }
                    else if (value_name == "Name")
                    {
                        m_output_stream_config[entry_name].name = output_stream_it.value();
                    }
                    else if (value_name == "LogLevel")
                    {
                        if (!stringToLogLevel(output_stream_it.value(), m_output_stream_config[entry_name].log_level))
                        {
                            std::cerr << "LOGGING CONFIG ERROR: Illegal log level in " << output_stream_it.key() << std::endl;
                        }
                    }
                    else if (value_name.substr(0, 9) == "LogStream")
                    {
                        m_output_stream_config[entry_name].log_streams.push_back(output_stream_it.value());
                    }
                }

                // Read the log stream configuration.
                ::icl_core::config::ConfigIterator log_stream_it =
                    ::icl_core::config::find(R"(\/IclCore\/Logging\/(LogStream.*)\/(.*))");
                while (log_stream_it.next())
                {
                    ::std::string entry_name = log_stream_it.matchGroup(1);
                    ::std::string value_name = log_stream_it.matchGroup(2);
                    if (value_name == "Name")
                    {
                        m_log_stream_config[entry_name].name = log_stream_it.value();
                    }
                    else if (value_name == "LogLevel")
                    {
                        if (!stringToLogLevel(log_stream_it.value(), m_log_stream_config[entry_name].log_level))
                        {
                            std::cerr << "LOGGING CONFIG ERROR: Illegal log level in " << log_stream_it.key() << std::endl;
                        }
                    }
                }
            }

            configure();

            // Configure the "QuickDebug" log stream and log output stream.
            std::string quick_debug_filename;
            if (icl_core::config::paramOpt<std::string>("quick-debug", quick_debug_filename))
            {
                // Find an unused name for the QuickDebug[0-9]* log output stream.
                std::string output_stream_name = "QuickDebug";
                LogOutputStreamMap::const_iterator find_it = m_log_output_streams.find(output_stream_name);
                if (find_it != m_log_output_streams.end())
                {
                    size_t count = 0;
                    do
                    {
                        ++count;
                        find_it = m_log_output_streams.find(output_stream_name
                            + std::to_string(count));
                    } while (find_it != m_log_output_streams.end());
                    output_stream_name = output_stream_name + std::to_string(count);
                }

                // Create the log output stream and connect the log stream.
                LogOutputStream* output_stream = new FileLogOutput(output_stream_name, quick_debug_filename,
                    eLL_TRACE, true);
                m_log_output_streams[output_stream_name] = output_stream;
                QuickDebug::instance().addOutputStream(output_stream);
                QuickDebug::instance().m_initial_level = eLL_TRACE;
            }

            // Run the log output stream threads.
            if (m_default_log_output != nullptr)
                m_default_log_output->start();

            for (const auto& val : m_log_output_streams | std::views::values)
                val->start();
        }

        void LoggingManager::configure()
        {
            // Create the default log output stream, if necessary.
            if (m_output_stream_config.empty() && m_default_log_output == nullptr)
            {
                m_default_log_output = StdLogOutput::create("Default", "/IclCore/Logging/Default");
            }

            // Create log stream instances, if necessary.
            for (auto& [name, factory] : m_log_stream_factories)
            {
                if (!m_log_streams.contains(name))
                    registerLogStream((*factory)());
            }

            // Delete the default log output stream, if necessary.
            if (!m_output_stream_config.empty() && m_default_log_output != nullptr)
            {
                for (const auto& val : m_log_streams | std::views::values)
                    val->removeOutputStream(m_default_log_output);

                m_default_log_output->shutdown();
                delete m_default_log_output;
                m_default_log_output = nullptr;
            }

            // Run through the log output stream configuration
            for (auto& loc_it : m_output_stream_config)
            {
                // Auto-generate a suitable name for the log output stream, if it
                // has not been set in the configuration.
                if (loc_it.second.name.empty())
	                loc_it.second.name = loc_it.second.output_stream_name;

                // Create the configured log output stream, if necessary.
                LogOutputStreamMap::const_iterator find_log_output_stream =
                    m_log_output_streams.find(loc_it.second.name);
                if (find_log_output_stream == m_log_output_streams.end())
                {
                    LogOutputStreamFactoryMap::const_iterator find_log_output_stream_factory =
                        m_log_output_stream_factories.find(loc_it.second.output_stream_name);
                    if (find_log_output_stream_factory == m_log_output_stream_factories.end())
                    {
                        // If the log output stream cannot be created then skip to the
                        // next configuration entry.
                        continue;
                    }
                    LogOutputStream* log_output_stream =
                        (*find_log_output_stream_factory->second)(loc_it.second.name,
                                                                  "/IclCore/Logging/" + loc_it.first,
                                                                  loc_it.second.log_level);
                    std::tie(find_log_output_stream, std::ignore) =
                        m_log_output_streams.emplace(loc_it.second.name, log_output_stream);
                }

                // Check again, just to be sure!
                if (find_log_output_stream != m_log_output_streams.end())
                {
                    // Connect the configured log streams (either the list from the
                    // commandline or all available log streams).
                    if (loc_it.second.log_streams.empty())
                    {
                        for (const auto& val : m_log_streams | std::views::values)
                            val->addOutputStream(find_log_output_stream->second);
                    }
                    else
                    {
                        for (const auto& log_stream : loc_it.second.log_streams)
                        {
                            auto find_it = m_log_streams.find(log_stream);
                            if (find_it == m_log_streams.end())
                            {
                                // If the log stream cannot be found then skip to the next
                                // entry.  Maybe there will be a second call to configure()
                                // in the future and the log stream is available then.
                                continue;
                            }
                            else
                            {
                                find_it->second->addOutputStream(find_log_output_stream->second);
                            }
                        }
                    }
                }
            }

            // Set the log level of the configured log streams (either the list
            // from the commandline or all available log streams).
            for (const auto& val : m_log_stream_config | std::views::values)
            {
                auto find_it = m_log_streams.find(val.name);
                if (find_it == m_log_streams.end())
                {
                    // If the log stream cannot be found then skip to the next
                    // entry.  Maybe there will be a second call to configure() in
                    // the future and the log stream is available then.
                    continue;
                }
                else
                {
                    find_it->second->m_initial_level = val.log_level;
                }
            }


            if (icl_core::config::Getopt::instance().paramOptPresent("log-level"))
            {
                LogLevel initial_level = cDEFAULT_LOG_LEVEL;
                const std::string log_level = icl_core::config::Getopt::instance().paramOpt("log-level");
                if (!stringToLogLevel(log_level, initial_level))
                {
                    std::cerr << "Illegal log level " << log_level << std::endl;
                    std::cerr << "Valid levels are 'Trace', 'Debug', 'Info', 'Warning', 'Error' and 'Mute'." << std::endl;
                }
                else
                {
                    if (m_default_log_output == nullptr)
                    {
                        m_default_log_output = StdLogOutput::create("Default", "/IclCore/Logging/Default");
                    }
                    m_default_log_output->setLogLevel(initial_level);

                    for (const auto& val : m_log_streams | std::views::values)
                    {
                        val->m_initial_level = initial_level;
                        val->addOutputStream(m_default_log_output);
                    }

                    for (const auto& val : m_log_output_streams | std::views::values)
                    {
                        val->setLogLevel(initial_level);
                    }
                }
            }
        }

        void LoggingManager::setLogLevel(icl_core::logging::LogLevel log_level)
        {
            for (const auto& val : m_log_streams | std::views::values)
	            val->m_initial_level = log_level;

            for (const auto& val : m_log_output_streams | std::views::values)
	            val->setLogLevel(log_level);
        }

        void LoggingManager::assertInitialized() const
        {
            if (!initialized())
                assert(0);
        }

        void LoggingManager::registerLogOutputStream(const ::std::string& name, LogOutputStreamFactory factory)
        {
            m_log_output_stream_factories[name] = factory;
        }

        void LoggingManager::removeLogOutputStream(LogOutputStream* log_output_stream, bool remove_from_list)
        {
            for (const auto& val : m_log_streams | std::views::values)
                val->removeOutputStream(log_output_stream);

            if (remove_from_list)
                m_log_output_streams.erase(log_output_stream->name());
        }

        void LoggingManager::registerLogStream(const std::string& name, LogStreamFactory factory)
        {
            m_log_stream_factories[name] = factory;
        }

        void LoggingManager::registerLogStream(LogStream* log_stream)
        {
            m_log_streams[log_stream->name()] = log_stream;

            if (m_default_log_output != nullptr)
                log_stream->addOutputStream(m_default_log_output);
        }

        void LoggingManager::removeLogStream(const std::string& log_stream_name)
        {
            if (!m_shutdown_running)
                m_log_streams.erase(log_stream_name);
        }

        LoggingManager::LoggingManager()
        {
            m_initialized = false;
            m_shutdown_running = false;
            m_default_log_output = nullptr;

            const std::string help_text =
                "Override the log level of all streams and connect them to stdout. "
                "Possible values are 'Trace', 'Debug', 'Info', 'Warning', 'Error' and 'Mute'.";
            icl_core::config::addParameter(icl_core::config::GetoptParameter("log-level:", "l", help_text));
            icl_core::config::addParameter(icl_core::config::GetoptParameter(
                "quick-debug:", "qd",
                "Activate the QuickDebug log stream and write it "
                "to the specified file."));
        }

        LoggingManager::~LoggingManager()
        {
            shutdown();
        }

        void LoggingManager::printConfiguration() const
        {
            std::cerr << "LoggingManager configuration:" << std::endl;

            std::cerr << "  Log output stream factories:" << std::endl;
            for (const auto& key : m_log_output_stream_factories | std::views::keys)
                std::cerr << "    " << key << std::endl;

            std::cerr << "  Log output streams:" << std::endl;
            if (m_default_log_output)
                m_default_log_output->printConfiguration();

            for (const auto& val : m_log_output_streams | std::views::values)
                val->printConfiguration();

            std::cerr << "  Log streams:" << std::endl;
            for (const auto& [name, stream] : m_log_streams)
            {
                std::cerr << "    " << name << " -> ";
                stream->printConfiguration();
                std::cerr << std::endl;
            }
        }

        void LoggingManager::changeLogFormat(const ::std::string& name, const char* format)
        {
            for (const auto& m_log_output_stream : m_log_output_streams)
            {
                if (m_log_output_stream.first == name)
                    m_log_output_stream.second->changeLogFormat(format);
            }
        }

        void LoggingManager::shutdown()
        {
            m_initialized = false;
            m_shutdown_running = true;

            // If the default log output stream exists then remove it from all connected
            // log streams and delete it afterwards.
            if (m_default_log_output != nullptr)
            {
                removeLogOutputStream(m_default_log_output, false);
                m_default_log_output->shutdown();
                delete m_default_log_output;
                m_default_log_output = nullptr;
            }

            // Remove all log output streams from all connected log streams and delete
            // the output streams afterwards.
            for (const auto& val : m_log_output_streams | std::views::values)
            {
                removeLogOutputStream(val, false);
                val->shutdown();
                delete val;
            }

            // Clear the log output stream map.
            m_log_output_streams.clear();

            // Delete all log streams.
            for (const auto& val : m_log_streams | std::views::values)
                delete val;

            // Clear the log stream map.
            m_log_streams.clear();

            m_shutdown_running = false;
        }

        namespace hidden {

            LogOutputStreamRegistrar::LogOutputStreamRegistrar(const ::std::string& name,
                LogOutputStreamFactory factory)
            {
                LoggingManager::instance().registerLogOutputStream(name, factory);
            }

            LogStreamRegistrar::LogStreamRegistrar(const ::std::string& name, LogStreamFactory factory)
            {
                LoggingManager::instance().registerLogStream(name, factory);
            }

        }

        LifeCycle::LifeCycle(int& argc, char* argv[])
        {
            icl_core::config::initialize(argc, argv, icl_core::config::Getopt::eCLC_Cleanup, icl_core::config::Getopt::ePRC_Relaxed);
            LoggingManager::instance().initialize();
        }

        LifeCycle::~LifeCycle()
        {
            LoggingManager::instance().shutdown();
        }
    }
}