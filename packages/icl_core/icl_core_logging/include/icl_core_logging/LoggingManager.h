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
 * \author  Klaus Uhl <uhl@fzi.de>
 * \date    2007-10-02
 *
 * \brief   Contains icl_logging::LoggingManager
 *
 * \b icl_logging::LoggingManager
 *
 */
//----------------------------------------------------------------------
#ifndef ICL_CORE_LOGGING_LOGGING_MANAGER_H_INCLUDED
#define ICL_CORE_LOGGING_LOGGING_MANAGER_H_INCLUDED

#include <string>
#include <list>
#include <map>
#include <memory>

#include "icl_core_logging/ImportExport.h"
#include "icl_core_logging/LogLevel.h"

#ifdef _IC_BUILDER_DEPRECATED_STYLE_
# include "icl_core/Deprecate.h"
#endif

namespace icl_core {
    namespace logging {

        class LogOutputStream;
        class LogStream;

        typedef LogOutputStream* (*LogOutputStreamFactory)(const std::string& name,
            const std::string& config_prefix,
            LogLevel log_level);
        typedef LogStream* (*LogStreamFactory)();

        /*! \brief Manages the logging framework.
         *
         *  The logging framework can be configured through a call to
         *  Initialize(). It will then read its configuration via
         *  icl_core_config. See
         *  http://www.mca2.org/wiki/index.php/ProgMan:Logging for the
         *  details.
         */
        class ICL_CORE_LOGGING_IMPORT_EXPORT LoggingManager
        {
        public:

            // Forbid copying logging manager objects.
            LoggingManager(const LoggingManager&) = delete;
            LoggingManager& operator = (const LoggingManager&) = delete;

            static LoggingManager& instance()
            {
                static LoggingManager manager_instance;
                return manager_instance;
            }

            /*! Configures log streams and log output streams.
             *
             *  This function is only useful if log streams are created
             *  dynamically after the logging manager has been initialized.
             */
            void configure();

            /*! Initializes the logging manager.
             *
             *  Remark: It is preferred to use the convenience functions
             *  ::icl_core::logging::initialize(),
             *  ::icl_core::logging::initialize(int&, char *[], bool) or
             *  ::icl_core::logging::initialize(int&, char *[],
             *  ::icl_core::config::Getopt::CommandLineCleaning,
             *  ::icl_core::config::Getopt::ParameterRegistrationCheck) instead
             *  of directly calling this method.
             */
            void initialize();

            /*! Check if the logging manager has already been initialized.
             */
            [[nodiscard]] bool initialized() const { return m_initialized; }

            /*! Check if the logging manager has already been initialized.
             *  Aborts the program if not.
             */
            void assertInitialized() const;

            /*! Registers a log output stream factory with the manager.
             */
            void registerLogOutputStream(const std::string& name, LogOutputStreamFactory factory);

            /*! Removes a log output stream from the logging manager.
             */
            void removeLogOutputStream(LogOutputStream* log_output_stream, bool remove_from_list = true);

            /*! Registers a log stream factory with the manager.
             */
            void registerLogStream(const std::string& name, LogStreamFactory factory);

            /*! Registers a log stream with the manager.
             */
            void registerLogStream(LogStream* log_stream);

            /*! Removes a log stream from the logging manager.
             */
            void removeLogStream(const std::string& log_stream_name);

            /*! Prints the configuration of log streams and log output streams.
             *
             *  Remark: This is mainly for debugging purposes!
             */
            void printConfiguration() const;

            /*! Changes the log output format of the log output streams. See
             *  LogOutputStream#changeLogFormat for format definition
             */
            void changeLogFormat(const std::string& name, const char* format = "~T ~S(~L)~ C~(O~::D: ~E");

            /*! Shuts down the logging framework. Any log messages that are
             *  pending in log output streams are written out. The log output
             *  stream threads are then stopped so that no further log messages
             *  are processed.
             */
            void shutdown();

            //! Set the log level globally for all existing streams.
            void setLogLevel(icl_core::logging::LogLevel log_level);

        private:
            typedef std::list<std::string> StringList;

            //! Configuration of a LogOutputStream.
            struct LogOutputStreamConfig
            {
                /*! The name of the output stream class as registered by the
                 *  implementation.
                 */
                std::string output_stream_name;
                //! The name of the output stream instance which will be created.
                std::string name;
                //! The log level of this output stream.
                LogLevel log_level;
                //! All associated log streams.
                StringList log_streams;
            };
            typedef std::map<std::string, LogOutputStreamConfig> LogOutputStreamConfigMap;

            //! Configuration of a LogStream.
            struct LogStreamConfig
            {
                //! Name of the log stream.
                std::string name;
                //! Log level of the log stream.
                LogLevel log_level;
            };
            typedef std::map<std::string, LogStreamConfig> LogStreamConfigMap;

            LoggingManager();
            ~LoggingManager();

            bool m_initialized;
            bool m_shutdown_running;

            LogOutputStreamConfigMap m_output_stream_config;
            LogStreamConfigMap m_log_stream_config;

            typedef std::map<std::string, LogStream*> LogStreamMap;
            typedef std::map<std::string, LogOutputStreamFactory> LogOutputStreamFactoryMap;
            typedef std::map<std::string, LogStreamFactory> LogStreamFactoryMap;
            typedef std::map<std::string, LogOutputStream*> LogOutputStreamMap;
            LogStreamMap m_log_streams;
            LogOutputStreamFactoryMap m_log_output_stream_factories;
            LogStreamFactoryMap m_log_stream_factories;
            LogOutputStreamMap m_log_output_streams;

            LogOutputStream* m_default_log_output;
        };

        //! Internal namespace for implementation details.
        namespace hidden {

            /*! Helper class to register a log output stream with the logging
             *  manager.
             *
             *  Remark: Never use this class directly! Use the
             *  REGISTER_LOG_OUTPUT_STREAM() macro instead!
             */
            class ICL_CORE_LOGGING_IMPORT_EXPORT LogOutputStreamRegistrar
            {
            public:
                LogOutputStreamRegistrar(const std::string& name, LogOutputStreamFactory factory);
            };

            /*! Helper class to register a log stream with the logging manager.
             *
             *  Remark: Never use this class directly! Use the
             *  REGISTER_LOG_STREAM() macro instead!
             */
            class ICL_CORE_LOGGING_IMPORT_EXPORT LogStreamRegistrar
            {
            public:
                LogStreamRegistrar(const std::string& name, LogStreamFactory factory);
            };

        }

        /**
         * Convenience class to manage the initialize() shutdown() sequence
         */
        class ICL_CORE_LOGGING_IMPORT_EXPORT LifeCycle
        {
        public:
            //! Convenience shared pointer shorthand.
            typedef std::shared_ptr<LifeCycle> Ptr;
            //! Convenience shared pointer shorthand (const version).
            typedef std::shared_ptr<const LifeCycle> ConstPtr;

            /** Initializes logging and removes known parameters from argc, argv */
            LifeCycle(int& argc, char* argv[]);

            /** Shuts down logging (!) */
            ~LifeCycle();
        };

    }
}

#endif