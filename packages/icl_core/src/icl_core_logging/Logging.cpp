// this is for emacs file handling -*- mode: c++; indent-tabs-mode: nil -*-
//----------------------------------------------------------------------
/*!\file
 *
 * \author  Klaus Uhl <uhl@fzi.de>
 * \date    2007-11-18
 */
//----------------------------------------------------------------------
#include "icl_core_logging/Logging.h"

#include <icl_core_config/Config.h>

namespace icl_core {
    namespace logging {

        ThreadStream& operator << (ThreadStream& stream, const icl_core::TimeStamp& time_stamp)
        {
            stream << time_stamp.formatIso8601();
            return stream;
        }

        REGISTER_LOG_STREAM(Default)
        REGISTER_LOG_STREAM(Nirwana)
        REGISTER_LOG_STREAM(QuickDebug)

        bool initialize(int& argc, char* argv[], bool remove_read_arguments)
        {
            return icl_core::logging::initialize(
                argc, argv,
                remove_read_arguments ? icl_core::config::Getopt::eCLC_Cleanup : icl_core::config::Getopt::eCLC_None,
                icl_core::config::Getopt::ePRC_Strict);
        }

        bool initialize(int& argc, char* argv[],
            icl_core::config::Getopt::CommandLineCleaning cleanup,
            icl_core::config::Getopt::ParameterRegistrationCheck registration_check)
        {
	        const bool result = icl_core::config::initialize(argc, argv, cleanup, registration_check);
            LoggingManager::instance().initialize();
            return result;
        }

        void initialize()
        {
            LoggingManager::instance().initialize();
        }

        void shutdown()
        {
            LoggingManager::instance().shutdown();
        }

        std::shared_ptr<LifeCycle> autoStart(int& argc, char* argv[])
        {
            return std::make_shared<LifeCycle>(argc, argv);
        }

        void setLogLevel(icl_core::logging::LogLevel log_level)
        {
            LoggingManager::instance().setLogLevel(log_level);
        }

    }
}