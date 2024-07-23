// this is for emacs file handling -*- mode: c++; indent-tabs-mode: nil -*-
//----------------------------------------------------------------------
/*!\file
 *
 * \author  Klaus Uhl <uhl@fzi.de>
 * \date    2007-10-02
 *
 */
 //----------------------------------------------------------------------
#include "LogLevel.h"

namespace icl_core {
	namespace logging {

		const char* log_level_descriptions[] =
		{
		  "Trace",
		  "Debug",
		  "Info",
		  "Warning",
		  "Error",
		  "Mute",
		  nullptr
		};

		const char* logLevelDescription(LogLevel log_level)
		{
			if (log_level < eLL_TRACE || log_level > eLL_MUTE)
				return "";

			return log_level_descriptions[log_level];
		}

		bool stringToLogLevel(const std::string& log_level_text, LogLevel& log_level)
		{
			for (size_t index = 0; index <= eLL_MUTE; ++index)
			{
				if (log_level_text == log_level_descriptions[index])
				{
					log_level = static_cast<LogLevel>(index);
					return true;
				}
			}
			return false;
		}

	}
}
