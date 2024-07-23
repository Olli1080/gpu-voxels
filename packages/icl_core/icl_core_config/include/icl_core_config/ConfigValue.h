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
#ifndef ICL_CORE_CONFIG_CONFIG_VALUE_H_INCLUDED
#define ICL_CORE_CONFIG_CONFIG_VALUE_H_INCLUDED

#include <string>

#include "icl_core_config/ConfigManager.h"
#include "icl_core_config/ConfigValueIface.h"
#include "icl_core_config/Util.h"

namespace icl_core {
	namespace config {

		/*! Typed "container" class for batch reading of configuration
		 *  parameters.
		 */
		template <typename T>
		class ConfigValue : public impl::ConfigValueIface
		{
		public:
			/*! Create a placeholder for later batch reading of configuration
			 *  parameters.
			 */
			ConfigValue(std::string key,
			            T& value)
				: m_key(std::move(key)),
				m_value(value)
			{ }

			/*! We need a virtual destructor!
			 */
			~ConfigValue() override = default;

			/*! Actually read the configuration parameter.
			 */
			bool get(std::string const& prefix, icl_core::logging::LogStream& log_stream) const override
			{
				if (ConfigManager::instance().get(prefix + m_key, m_str_value))
				{
					try
					{
						m_value = impl::hexical_cast<T>(m_str_value);
						return true;
					}
					catch (...)
					{}
				}
				return false;
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
		};

		template<>
		inline
			bool ConfigValue<bool>::get(std::string const& prefix, icl_core::logging::LogStream& log_stream) const
		{
			if (ConfigManager::instance().get(prefix + m_key, m_str_value))
			{
				try
				{
					m_value = impl::strict_bool_cast(m_str_value);
					return true;
				}
				catch (...)
				{}
			}
			return false;
		}
	}
}

template<typename V>
auto CONFIG_VALUE(auto key, V& value)
{
	return new icl_core::config::ConfigValue<V>(key, value);
}

#endif