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
 * \date    2007-11-17
 *
 * \brief   Defines logging macros.
 *
 */
 //----------------------------------------------------------------------
#ifndef ICL_CORE_LOGGING_LOGGING_H_INCLUDED
#define ICL_CORE_LOGGING_LOGGING_H_INCLUDED

#include <cassert>
#include <iostream>
#include <memory>
#include <chrono>

#include "icl_core_logging/Constants.h"
#include "icl_core_logging/ImportExport.h"
#include "icl_core_logging/LoggingManager.h"
#include "icl_core_logging/LogStream.h"
#include "icl_core_logging/ThreadStream.h"
#include "icl_core_config/GetoptParser.h"


#include "icl_core_logging/LoggingMacros_LOGGING.h"
#include "icl_core_logging/LoggingMacros_LOGGING_FMT.h"
#include "icl_core_logging/LoggingMacros_LLOGGING.h"
#include "icl_core_logging/LoggingMacros_LLOGGING_FMT.h"
#include "icl_core_logging/LoggingMacros_MLOGGING.h"
#include "icl_core_logging/LoggingMacros_MLOGGING_FMT.h"
#include "icl_core_logging/LoggingMacros_SLOGGING.h"
#include "icl_core_logging/LoggingMacros_SLOGGING_FMT.h"

#ifdef _IC_BUILDER_DEPRECATED_STYLE_
# include "icl_core/Deprecate.h"
#endif

#define LOG_THREAD_STREAM(name) name::instance().ThreadStream()

#define DECLARE_LOG_STREAM_CLASS_DEFINITION(name)                 \
  name : public ::icl_core::logging::LogStream                     \
  {                                                                \
  public:                                                          \
    static ::icl_core::logging::LogStream& instance();             \
    static ::icl_core::logging::LogStream *create();               \
  private:                                                         \
    name()                                                         \
      : LogStream(#name)                                           \
    { }                                                            \
    ~name() { }                                                    \
    static name *m_instance;                                       \
    friend class ::icl_core::logging::LoggingManager;              \
    friend class ::icl_core::logging::hidden::LogStreamRegistrar;  \
  };

#define DECLARE_LOG_STREAM(name) class DECLARE_LOG_STREAM_CLASS_DEFINITION(name)
#define DECLARE_LOG_STREAM_IMPORT_EXPORT(name, decl) class decl DECLARE_LOG_STREAM_CLASS_DEFINITION(name)

// Remark: The log stream object is created here but will be deleted in the
// destructor of LoggingManager!
#define REGISTER_LOG_STREAM(name)                                       \
  name * name::m_instance = nullptr;                                       \
  ::icl_core::logging::LogStream& name::instance()                      \
  {                                                                     \
    if (m_instance == nullptr)                                             \
    {                                                                   \
      std::cout << "WARNING: Logging Instance is null, did you initialize the logging framework?\nYou should initialize the logging framework at the beginning of your program. This will also enable setting the log level on the command line." << std::endl; \
      ::icl_core::logging::LoggingManager::instance().initialize();     \
      assert(m_instance != nullptr && "Tried to initialize LoggingManager but m_instance still not available."); \
      return *m_instance;                                               \
    }                                                                   \
    else                                                                \
    {                                                                   \
      return *m_instance;                                               \
    }                                                                   \
  }                                                                     \
  ::icl_core::logging::LogStream * name::create()                       \
  {                                                                     \
    if (m_instance == nullptr)                                             \
    {                                                                   \
      m_instance = new name;                                            \
    }                                                                   \
    return m_instance;                                                  \
  }                                                                     \
  ::icl_core::logging::hidden::LogStreamRegistrar registrar##name(#name, &name::create);

#define REGISTER_LOG_OUTPUT_STREAM(name, factory)                       \
  ::icl_core::logging::hidden::LogOutputStreamRegistrar registrar##name(#name, factory);

#define DECLARE_LOG_STREAM_OPERATOR(object_type)                        \
  ::icl_core::logging::ThreadStream & operator << (::icl_core::logging::ThreadStream &str, \
                                                   const object_type &object);

#define REGISTER_LOG_STREAM_OPERATOR(object_type)                       \
  ::icl_core::logging::ThreadStream & operator << (::icl_core::logging::ThreadStream &str, \
                                                   const object_type &object) \
  {                                                                     \
    std::ostringstream stream;                                          \
    stream << object;                                                   \
    str << stream.str();                                                \
    return str;                                                         \
  }

namespace icl_core {
	//! Flexible, powerful, configurable logging framework.
	namespace logging {

		ICL_CORE_LOGGING_IMPORT_EXPORT
			ThreadStream& operator << (ThreadStream& stream, const std::chrono::system_clock::time_point& time_stamp);

		//ICL_CORE_LOGGING_IMPORT_EXPORT
		//ThreadStream& operator << (ThreadStream& stream, const icl_core::TimeSpan& time_span);

		DECLARE_LOG_STREAM_IMPORT_EXPORT(Default, ICL_CORE_LOGGING_IMPORT_EXPORT)
			DECLARE_LOG_STREAM_IMPORT_EXPORT(Nirwana, ICL_CORE_LOGGING_IMPORT_EXPORT)
			DECLARE_LOG_STREAM_IMPORT_EXPORT(QuickDebug, ICL_CORE_LOGGING_IMPORT_EXPORT)

			/*! Convenience function to initialize the logging framework.
			 *
			 *  Also initializes the configuration framework.
			 */
			bool ICL_CORE_LOGGING_IMPORT_EXPORT initialize(int& argc, char* argv[], bool remove_read_arguments);

		/*! Convenience function to initialize the logging framework.
		 *
		 *  Also initializes the configuration framework.
		 */
		bool ICL_CORE_LOGGING_IMPORT_EXPORT
			initialize(int& argc, char* argv[],
				icl_core::config::Getopt::CommandLineCleaning cleanup
				= icl_core::config::Getopt::eCLC_None,
				icl_core::config::Getopt::ParameterRegistrationCheck registration_check
				= icl_core::config::Getopt::ePRC_Strict);

		/*! Convenience function to initialize the logging framework.
		 *
		 *  Use this version if you already have initialized the configuration
		 *  framework.
		 */
		void ICL_CORE_LOGGING_IMPORT_EXPORT initialize();

		/*! Convenience function to shutdown the logging framework.
		 */
		void ICL_CORE_LOGGING_IMPORT_EXPORT shutdown();

		std::shared_ptr<LifeCycle> ICL_CORE_LOGGING_IMPORT_EXPORT autoStart(int& argc, char* argv[]);

		//! Set a global log level for all streams.
		void ICL_CORE_LOGGING_IMPORT_EXPORT setLogLevel(icl_core::logging::LogLevel log_level);

	}
}

#endif