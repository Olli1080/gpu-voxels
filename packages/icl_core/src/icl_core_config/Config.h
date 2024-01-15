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
 * \date    2008-11-01
 *
 * \brief   Base header file for the configuration framework.
 *
 * Contains convenience functions to access the ConfigManager singleton's functionality.
 *
 */
//----------------------------------------------------------------------
#ifndef ICL_CORE_CONFIG_CONFIG_H_INCLUDED
#define ICL_CORE_CONFIG_CONFIG_H_INCLUDED

#include <icl_core/EnumHelper.h>
//#include <icl_core/StringHelper.h>
#include <icl_core_logging/Logging.h>

#include "icl_core_config/ImportExport.h"
#include "icl_core_config/ConfigIterator.h"
#include "icl_core_config/ConfigManager.h"
#include "icl_core_config/ConfigParameter.h"
#include "icl_core_config/ConfigValues.h"
#include "icl_core_config/GetoptParser.h"
#include "icl_core_config/GetoptParameter.h"
#include "icl_core_config/Util.h"

namespace icl_core {
    //! Framework for processing configuration files.
    namespace config {

        ICL_CORE_CONFIG_IMPORT_EXPORT inline const char* CONFIGFILE_CONFIG_KEY = "/configfile";

        ICL_CORE_CONFIG_IMPORT_EXPORT void dump();

        ICL_CORE_CONFIG_IMPORT_EXPORT void debugOutCommandLine(int argc, char* argv[]);

        ICL_CORE_CONFIG_IMPORT_EXPORT ConfigIterator find(const std::string& query);

        //! Gets the value for the specified \a key from the configuration.
        template <typename T>
        bool get(const std::string& key, T& value);

        //! Gets the value for the specified \a key from the configuration.
        template <typename T>
        bool get(const char* key, T& value)
        {
            return get<T>(std::string(key), value);
        }

        //! Gets the value for the specified \a key from the configuration.
        template <typename T>
        bool get(const std::string& key, T& value)
        {
            std::string str_value;
            if (ConfigManager::instance().get(key, str_value))
            {
                try
                {
                    value = impl::hexical_cast<T>(str_value);
                    return true;
                }
                catch (...)
                {}
            }
            return false;
        }

        //! Template specialization for std::string.
        template <>
        inline bool get<std::string>(const std::string& key, std::string& value)
        {
            return ConfigManager::instance().get(key, value);
        }

        //! Template specialization for boolean values.
        template <>
        inline bool get<bool>(const std::string& key, bool& value)
        {
            std::string str_value;
            if (ConfigManager::instance().get(key, str_value))
            {
                str_value = boost::to_lower_copy(str_value);
                if (str_value == "0" || str_value == "no" || str_value == "false")
                {
                    value = false;
                    return true;
                }
                else if (str_value == "1" || str_value == "yes" || str_value == "true")
                {
                    value = true;
                    return true;
                }
            }
            return false;
        }

        template <typename T>
        bool get(const std::string& key, T& value,
            const char* descriptions[], const char* end_marker = nullptr)
        {
            std::string str_value;
            if (ConfigManager::instance().get(key, str_value))
            {
                int32_t raw_value;
                if (icl_core::string2Enum(str_value, raw_value, descriptions, end_marker))
                {
                    value = T(raw_value);
                    return true;
                }
            }
            return false;
        }

        template <typename T>
        bool get(const char* key, T& value,
            const char* descriptions[], const char* end_marker = nullptr)
        {
            return get<T>(std::string(key), value, descriptions, end_marker);
        }

        template <typename T>
        T getDefault(const std::string& key, const T& default_value)
        {
            T value = default_value;
            get<T>(key, value);
            return value;
        }

        template <typename T>
        T getDefault(const char* key, const T& default_value)
        {
            return getDefault<T>(std::string(key), default_value);
        }

        template <>
    	inline std::string getDefault<std::string>(const std::string& key,
                const std::string& default_value)
        {
            std::string value = default_value;
            get<std::string>(key, value);
            return value;
        }

        /*! Get configuration parameters in batch mode. Returns \c true on
         *  success.  If \a report_error is \c true, writes an error message
         *  for each failed configuration parameter.  Returns \c false if any
         *  parameter failed.  Optionally deletes the contents of the \a
         *  config_values array.
         */
        inline bool get(const ConfigValues config_values, icl_core::logging::LogStream& log_stream,
                bool cleanup = true, bool report_error = true)
        {
            // Read the configuration parameters.
            bool result = true;
            const impl::ConfigValueIface* const* config = config_values;
            while (*config != nullptr)
            {
                if ((*config)->get())
                {
                    SLOGGING_TRACE(log_stream, "Read configuration parameter \""
                        << (*config)->key() << "\" = \"" << (*config)->stringValue()
                        << "\"." << icl_core::logging::endl);
                }
                else
                {
                    if (report_error)
                    {
                        SLOGGING_ERROR(log_stream, "Error reading configuration parameter \""
                            << (*config)->key() << "\"!" << icl_core::logging::endl);
                    }
                    else
                    {
                        SLOGGING_TRACE(log_stream, "Could not read configuration parameter \""
                            << (*config)->key() << "\"." << icl_core::logging::endl);
                    }
                    result = false;
                }
                ++config;
            }

            // Cleanup!
            if (cleanup)
            {
                config = config_values;
                while (*config != nullptr)
                {
                    delete* config;
                    ++config;
                }
            }

            return result;
        }

        /*! Get configuration parameters in batch mode. Returns \c true on
         *  success.  If \a report_error is \c true, writes an error message
         *  for each failed configuration parameter.  Returns \c false if any
         *  parameter failed.  Optionally deletes the contents of the \a
         *  config_values array.
         */
        inline bool get(std::string config_prefix,
                ConfigValueList config_values, icl_core::logging::LogStream& log_stream,
                bool cleanup = true, bool report_error = true)
        {
            /* Remark: config_values has to be passed by value, not by reference.
             *         Otherwise boost::assign::list_of() can not work correctly.
             */

             // Add a trailing slash, if necessary.
            if (!config_prefix.empty() && config_prefix[config_prefix.length() - 1] != '/')
            {
                config_prefix = config_prefix + "/";
            }

            // Read the configuration parameters.
            bool result = false;
            bool error = false;

            for (const auto config : config_values)
            {
                if (config->get(config_prefix, log_stream))
                {
                    SLOGGING_TRACE(log_stream, "Read configuration parameter \""
                        << config_prefix << config->key() << "\" = \"" << config->stringValue()
                        << "\"." << icl_core::logging::endl);
                }
                else
                {
                    if (report_error)
                    {
                        SLOGGING_ERROR(log_stream, "Error reading configuration parameter \""
                            << config_prefix << config->key() << "\"!" << icl_core::logging::endl);
                    }
                    else
                    {
                        SLOGGING_TRACE(log_stream, "Could not read configuration parameter \""
                            << config_prefix << config->key() << "\"." << icl_core::logging::endl);
                    }
                    error = true;
                }
                result = true;
            }

            if (error)
            {
                result = false;
            }

            // Cleanup!
            if (cleanup)
            {
                for (const auto config : config_values)
                    delete config;
            }

            return result;
        }

        inline bool get(const ConfigValueList& config_values, icl_core::logging::LogStream& log_stream,
                        bool cleanup = true, bool report_error = true)
        {
            return get("", config_values, log_stream, cleanup, report_error);
        }

        template <typename T>
        void setValue(const std::string& key, const T& value)
        {
            ConfigManager::instance().setValue<T>(key, value);
        }

        template <typename T>
        void setValue(const char* key, const T& value)
        {
            ConfigManager::instance().setValue<T>(std::string(key), value);
        }

        inline
            void setValue(const std::string& key, const std::string& value)
        {
            setValue<std::string>(key, value);
        }

        inline
            bool paramOptPresent(const std::string& name)
        {
            return Getopt::instance().paramOptPresent(name);
        }

        template <typename T>
        bool paramOpt(const std::string& name, T& value)
        {
	        const Getopt& getopt = Getopt::instance();
            if (getopt.paramOptPresent(name))
            {
                try
                {
                    value = impl::hexical_cast<T>(getopt.paramOpt(name));
                    return true;
                }
                catch (...)
                {
                    return false;
                }
            }
            else
            {
                return false;
            }
        }
        template <typename T>
        bool paramOpt(const char* name, T& value)
        {
            return paramOpt<T>(std::string(name), value);
        }

        template <typename T>
        bool paramOpt(const std::string& name, T& value,
            const char* descriptions[], const char* end_marker = nullptr)
        {
	        const Getopt& getopt = Getopt::instance();
            if (getopt.paramOptPresent(name))
            {
	            const std::string str_value = getopt.paramOpt(name);
                int32_t raw_value;
                if (icl_core::string2Enum(str_value, raw_value, descriptions, end_marker))
                {
                    value = T(raw_value);
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

        template <typename T>
        bool paramOpt(const char* name, T& value,
            const char* descriptions[], const char* end_marker = nullptr)
        {
            return paramOpt<T>(std::string(name), value, descriptions, end_marker);
        }

        template <typename T>
        T paramOptDefault(const std::string& name, const T& default_value)
        {
	        if (const Getopt& getopt = Getopt::instance(); getopt.paramOptPresent(name))
            {
                try
                {
                    return impl::hexical_cast<T>(getopt.paramOpt(name));
                }
                catch (...)
                {}
            }
            return default_value;
        }

        template <typename T>
        T paramOptDefault(const char* name, const T& default_value)
        {
            return paramOptDefault<T>(std::string(name), default_value);
        }

        template <typename T>
        bool paramNonOpt(size_t index, T& value)
        {
	        if (const Getopt& getopt = Getopt::instance(); index < getopt.paramNonOptCount())
            {
                try
                {
                    value = impl::hexical_cast<T>(getopt.paramNonOpt(index));
                    return true;
                }
                catch (...)
                {}
            }
            return false;
        }

        template <typename T>
        bool paramNonOpt(size_t index, T& value,
            const char* descriptions[], const char* end_marker = nullptr)
        {
	        if (const Getopt& getopt = Getopt::instance(); index < getopt.paramNonOptCount())
            {
	            const std::string str_value = getopt.paramNonOpt(index);
                int32_t raw_value;
                if (icl_core::string2Enum(str_value, raw_value, descriptions, end_marker))
                {
                    value = T(raw_value);
                    return true;
                }
            }
            return false;
        }

        inline std::string paramNonOpt(size_t index)
        {
            return Getopt::instance().paramNonOpt(index);
        }

        inline size_t extraCmdParamCount()
        {
            return Getopt::instance().extraCmdParamCount();
        }

        inline std::string extraCmdParam(size_t index)
        {
            return Getopt::instance().extraCmdParam(index);
        }

        inline void activateExtraCmdParams(const std::string& delimiter = "--")
        {
            Getopt::instance().activateExtraCmdParams(delimiter);
        }

        inline size_t paramNonOptCount()
        {
            return Getopt::instance().paramNonOptCount();
        }

        inline void addParameter(const ConfigParameter& parameter)
        {
            ConfigManager::instance().addParameter(parameter);
        }

        inline void addParameter(const ConfigParameterList& parameters)
        {
            ConfigManager::instance().addParameter(parameters);
        }

        inline void addParameter(const ConfigPositionalParameter& parameter)
        {
            ConfigManager::instance().addParameter(parameter);
        }

        inline void addParameter(const ConfigPositionalParameterList& parameters)
        {
            ConfigManager::instance().addParameter(parameters);
        }

        inline void addParameter(const GetoptParameter& parameter)
        {
            Getopt::instance().addParameter(parameter);
        }

        inline void addParameter(const GetoptParameterList& parameters)
        {
            Getopt::instance().addParameter(parameters);
        }

        inline void addParameter(const GetoptPositionalParameter& parameter)
        {
            Getopt::instance().addParameter(parameter);
        }

        inline void addParameter(const GetoptPositionalParameterList& parameters)
        {
            Getopt::instance().addParameter(parameters);
        }

        inline void setProgramVersion(std::string const& version)
        {
            Getopt::instance().setProgramVersion(version);
        }

        inline void setProgramDescription(std::string const& description)
        {
            Getopt::instance().setProgramDescription(description);
        }

        inline void printHelp()
        {
            Getopt::instance().printHelp();
        }

        ICL_CORE_CONFIG_IMPORT_EXPORT bool initialize(int& argc, char* argv[], bool remove_read_arguments);

        ICL_CORE_CONFIG_IMPORT_EXPORT
            bool initialize(int& argc, char* argv[],
                Getopt::CommandLineCleaning cleanup = Getopt::eCLC_None,
                Getopt::ParameterRegistrationCheck registration_check = Getopt::ePRC_Strict);

    }
}

#endif
