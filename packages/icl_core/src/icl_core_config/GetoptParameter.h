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
 * \date    2009-03-12
 *
 * \brief   Contains GetoptParameter
 *
 */
 //----------------------------------------------------------------------
#ifndef ICL_CORE_CONFIG_GETOPT_PARAMETER_H_INCLUDED
#define ICL_CORE_CONFIG_GETOPT_PARAMETER_H_INCLUDED

#include <string>
#include <vector>

#include "icl_core_config/ImportExport.h"

#ifdef _IC_BUILDER_DEPRECATED_STYLE_
# include "icl_core/Deprecate.h"
#endif

namespace icl_core {
	namespace config {

		class ICL_CORE_CONFIG_IMPORT_EXPORT GetoptParameter
		{
		public:
			/*! Create a new commandline parameter.
			 *
			 *  \param option The long option name of this parameter. If \a
			 *         option ends with a colon (":") then the parameter also
			 *         expects a value.
			 *  \param short_option The short option name of this parameter.  If
			 *         this is set to the empty string then no short option is
			 *         used.
			 *  \param help The help text for this parameter.
			 *  \param is_prefix Set to \c true if this is a prefix option.
			 *         Prefix Options are options like "-o/asd/asd".
			 *
			 *  \see GetoptParameter for details about the syntax of the \a
			 *  option parameter.
			 */
			GetoptParameter(const std::string& option, std::string short_option,
				std::string help, bool is_prefix = false);

			//! Get the long option name.
			[[nodiscard]] std::string option() const { return m_option; }
			//! Get the short option name.
			[[nodiscard]] std::string shortOption() const { return m_short_option; }
			//! Check if the option also expects a value.
			[[nodiscard]] bool hasValue() const { return m_has_value; }
			//! Get the help text.
			[[nodiscard]] std::string help() const { return m_help; }

			//! Check if this is a prefix option.
			[[nodiscard]] bool isPrefixOption() const { return m_is_prefix; }

		private:
			std::string m_option;
			std::string m_short_option;
			std::string m_help;
			bool m_has_value;
			bool m_is_prefix;
		};

		typedef std::vector<GetoptParameter> GetoptParameterList;

	}
}

#endif