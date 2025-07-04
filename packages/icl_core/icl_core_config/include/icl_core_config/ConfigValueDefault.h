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
 * \date    2010-04-28
 *
 */
 //----------------------------------------------------------------------
#ifndef ICL_CORE_CONFIG_CONFIG_VALUE_DEFAULT_H_INCLUDED
#define ICL_CORE_CONFIG_CONFIG_VALUE_DEFAULT_H_INCLUDED

#include <string>

#include "icl_core_config/ConfigValue.h"
#include "icl_core_config/Util.h"

namespace icl_core {
	namespace config {

		/*! Typed "container" class for batch reading of configuration
		 *  parameters with a default value.
		 */
		template <typename T>
		class ConfigValueDefault : public ConfigValue<T>
		{
		public:
			/*! Create a placeholder for later batch reading of configuration
			 *  parameters.
			 */
			ConfigValueDefault(const std::string& key,
				T& value,
				const T& default_value)
				: ConfigValue<T>(key, value),
				m_default_value(default_value)
			{ }

			/*! We need a virtual destructor!
			 */
			~ConfigValueDefault() override = default;

			/*! Actually read the configuration parameter.
			 */
			bool get(std::string const& prefix, icl_core::logging::LogStream& log_stream) const override
			{
				if (!ConfigValue<T>::get(prefix, log_stream))
				{
					this->m_value = m_default_value;
					this->m_str_value = impl::hexical_cast<std::string>(this->m_value);
				}
				return true;
			}

		private:
			const T& m_default_value;
		};

	}
}

template<typename V>
auto CONFIG_VALUE_DEFAULT(auto key, V& value, auto default_value)
{
	return new icl_core::config::ConfigValueDefault<V>(key, value, default_value);
}

#endif