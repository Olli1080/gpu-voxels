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
#ifndef ICL_CORE_CONFIG_CONFIG_ENUM_DEFAULT_H_INCLUDED
#define ICL_CORE_CONFIG_CONFIG_ENUM_DEFAULT_H_INCLUDED

#include <string>

#include "icl_core_config/ConfigEnum.h"
#include "icl_core_config/Util.h"

namespace icl_core {
	namespace config {

		/*! Typed "container" class for batch reading of configuration
		 *  parameters with a default value.
		 */
		template <typename T>
		class ConfigEnumDefault : public ConfigEnum<T>
		{
		public:
			/*! Create a placeholder for later batch reading of configuration
			 *  parameters.
			 */
			ConfigEnumDefault(const std::string& key,
				T& value,
				const T& default_value,
				const char* const* descriptions,
				const char* end_marker = nullptr)
				: ConfigEnum<T>(key, value, descriptions, end_marker),
				m_default_value(default_value)
			{ }

			/*! We need a virtual destructor!
			 */
			~ConfigEnumDefault() override {}

			/*! Actually read the configuration parameter.
			 */
			bool get(std::string const& prefix, icl_core::logging::LogStream& log_stream) const override
			{
				if (!ConfigEnum<T>::get(prefix, log_stream))
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
auto CONFIG_ENUM_DEFAULT(auto key, V& value, const V& default_value, auto descriptions)
{
	return new icl_core::config::ConfigEnumDefault<V>(key, value, default_value, descriptions);
}

#endif
