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
 */
 //----------------------------------------------------------------------
#ifndef ICL_CORE_CONFIG_CONFIG_ENUM_H_INCLUDED
#define ICL_CORE_CONFIG_CONFIG_ENUM_H_INCLUDED

#include <string>

#include <icl_core/EnumHelper.h>

#include "icl_core_config/ConfigManager.h"
#include "icl_core_config/ConfigValueIface.h"


namespace icl_core {
	namespace config {

		/*! Typed "container" class for batch reading of configuration parameters.
		 */
		template <typename T>
		class ConfigEnum : public impl::ConfigValueIface
		{
		public:
			/*! Create a placeholder for later batch reading of configuration
			 *  parameters.
			 */
			ConfigEnum(const std::string& key,
				T& value,
				const char* const* descriptions,
				const char* end_marker = nullptr)
				: m_key(key),
				m_value(value),
				m_descriptions(descriptions),
				m_end_marker(end_marker)
			{ }

			/*! We need a virtual destructor!
			 */
			~ConfigEnum() override {}

			/*! Actually read the configuration parameter.
			 */
			bool get(std::string const& prefix, icl_core::logging::LogStream& log_stream) const override
			{
				if (ConfigManager::instance().get(prefix + m_key, m_str_value))
				{
					int32_t raw_value;
					if (icl_core::string2Enum(m_str_value, raw_value, m_descriptions, m_end_marker))
					{
						m_value = T(raw_value);
						return true;
					}
					else
					{
						return false;
					}
				}
				else
				{
					return false;
				}
			}

			/*! Return the configuration key.
			 */
			std::string key() const override
			{
				return m_key;
			}

			/*! Return the value as string.
			 */
			std::string stringValue() const override
			{
				return m_str_value;
			}

		protected:
			std::string m_key;
			mutable std::string m_str_value;
			T& m_value;
			const char* const* m_descriptions;
			const char* m_end_marker;
		};

	}
}

template<typename V>
auto CONFIG_ENUM(auto key, V& value, auto descriptions)
{
	return new icl_core::config::ConfigEnum<V>(key, value, descriptions);
}

#endif