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
 * \date    2009-03-06
 *
 * \brief   Contains ConfigParameter
 *
 */
 //----------------------------------------------------------------------
#ifndef ICL_CORE_CONFIG_CONFIG_PARAMETER_H_INCLUDED
#define ICL_CORE_CONFIG_CONFIG_PARAMETER_H_INCLUDED

#include <string>
#include <vector>

#include "icl_core_config/ImportExport.h"
#include "icl_core_config/GetoptParameter.h"

#ifdef _IC_BUILDER_DEPRECATED_STYLE_
# include "icl_core/Deprecate.h"
#endif

namespace icl_core {
	namespace config {

		/*! Contains information about how to handle a specific commandline
		 *  parameter and how to map it into a configuration parameter.
		 *
		 *  For key/value parameters the option value is stored in the
		 *  specified config key. For presence parameters (without a value)
		 *  the value "yes" is stored in the config key if the option is
		 *  present on the commandline; otherwise the value "no" is stored in
		 *  the config key.
		 */
		class ICL_CORE_CONFIG_IMPORT_EXPORT ConfigParameter : public GetoptParameter
		{
		public:
			/*! Create a new config parameter.
			 *
			 *  \param option The long option name of this parameter. If \a
			 *         option ends with a colon (":") then the parameter also
			 *         expects a value.
			 *  \param short_option The short option name of this parameter.  If
			 *         this is set to the empty string then no short option is
			 *         used.
			 *  \param config_key The configuration key in which the option
			 *         value should be stored.
			 *  \param help The help text for this parameter.
			 *  \param default_value The default value to be set, if it has
			 *         neither been set in the config file and on the
			 *         commandline.
			 *
			 *  See GetoptParameter for details about the syntax of the \a
			 *  option parameter.
			 */
			ConfigParameter(const std::string& option, const std::string& short_option,
				std::string config_key, const std::string& help,
				const std::string& default_value = "");

			/*! Get the configuration key in which the option should be stored.
			 */
			[[nodiscard]] std::string configKey() const { return m_config_key; }

			/*! Check if a default value has been set.
			 */
			[[nodiscard]] bool hasDefaultValue() const { return !m_default_value.empty(); }

			/*! Get the default value of the configuration parameter.
			 */
			[[nodiscard]] std::string defaultValue() const { return m_default_value; }

		private:
			std::string m_config_key;
			std::string m_default_value;
		};

		typedef std::vector<ConfigParameter> ConfigParameterList;

	}
}

#endif