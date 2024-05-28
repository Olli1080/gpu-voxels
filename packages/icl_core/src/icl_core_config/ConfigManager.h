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
 * \date    2007-12-04
 *
 * \brief   Contains ConfigManager.
 *
 */
//----------------------------------------------------------------------
#ifndef ICL_CORE_CONFIG_CONFIG_MANAGER_H_INCLUDED
#define ICL_CORE_CONFIG_CONFIG_MANAGER_H_INCLUDED

#include <list>
#include <map>

#include "icl_core_config/ConfigIterator.h"
#include "icl_core_config/ConfigParameter.h"
#include "icl_core_config/ConfigPositionalParameter.h"
#include "icl_core_config/ImportExport.h"
#include "icl_core_config/AttributeTree.h"

#ifdef _IC_BUILDER_DEPRECATED_STYLE_
# include "icl_core/Deprecate.h"
#endif

class TiXmlNode;

namespace icl_core {
    namespace config {

        class AttributeTree;
        class ConfigObserver;

        //! Class for handling configuration files.
        /*!
         * ConfigManager is implemented as a singleton so that it
         * can be used from anywhere without the need to pass
         * a config object around.
         *
         * Before the configuration class can be used it has
         * to be initialized through a call to Initialize().
         * It will parse the command line and look for a
         * "-c [filename]" or "--configfile=[filename]" option.
         * ConfigManager will try to read the specified file and extract
         * all configuration attributes from it.
         *
         * Configuration files are XML files. The names of the XML
         * tags are used as the names of the configuration attributes
         * while the content text of the XML tags are used as
         * the attributes' values. Leading and trailing whitespace
         * is automatically removed from the values. Remark: The name
         * of the root element in the configuration file can be
         * chosen arbitrarily. It is not or interpreted from ConfigManager
         * in any way.
         *
         * Configuration attributes are retrieved using an XPath like
         * syntax. Hierarchical attribute names are separated by "/".
         */
        class ICL_CORE_CONFIG_IMPORT_EXPORT ConfigManager : public icl_core::KeyValueDirectory<std::string>
        {
        public:
            /*!
             * Get the singleton ConfigManager instance.
             */
            static ConfigManager& instance();

            /*!
             * Adds a commandline parameter.
             */
            void addParameter(const ConfigParameter& parameter);
            /*!
             * Adds a list of commandline parameters.
             */
            void addParameter(const ConfigParameterList& parameters);

            /*!
             * Adds a positional commandline parameter.
             */
            void addParameter(const ConfigPositionalParameter& parameter);
            /*!
             * Adds a list of positional commandline parameters.
             */
            void addParameter(const ConfigPositionalParameterList& parameters);

            /*!
             * Initializes ConfigManager. Reads the configuration file if
             * --configfile or -c has been specified on the commandline.
             * If no configuration file has been specified, the initialization
             * is treated as successful!
             *
             * \returns \a true if the initialization was successful, \a false
             *          otherwise. If the initialization fails, an error message
             *          will be printed to stderr.
             */
            bool initialize();

            /*!
             * Check if the configuration framework has already been initialized.
             */
            [[nodiscard]] bool isInitialized() const
            {
                return m_initialized;
            }

            /*!
             * Dumps all configuration keys and the corresponding values
             * to stdout.
             */
            void dump() const;

            //! Add a key/value pair or change a value. In contrast to Insert, this method notifies observers
            template <class T>
            bool setValue(const std::string& key, const T& value)
            {
                std::string string_value;
                if constexpr (std::is_same_v<T, std::string>)
                    string_value = value;
                else
                    string_value = std::to_string(value);

                if (key == "/configfile")
                {
                    load(string_value);
                }

	            const bool result = insert(key, string_value);
                notify(key);
                return result;
            }

            /**! Register an observer which gets notified of changed key/value pairs
             *   @param observer The observer to add to the list of registered observers
             *   @param key The key to be notified of, or an empty std::string for all changes
             */
            void registerObserver(ConfigObserver* observer, const std::string& key = "");

            /*! Unregister an observer so it does not get notified of changes anymore.
             *  Normally you shouldn't need to call this as the destructor of config
             *  observers automatically calls it
             */
            void unregisterObserver(ConfigObserver* observer);

        private:
            //! Creates an empty configuration object.
            ConfigManager();

            //! Reads configuration from a file.
            bool load(const std::string& filename);

            //! Notify all observers about a changed key/value pair
            void notify(const std::string& key) const;

            void readXml(const ::std::string& prefix, TiXmlNode* node, FilePath fp, bool extend_prefix = true);
            void readAttributeTree(const std::string& prefix, AttributeTree* at, bool extend_prefix = true);

            //typedef ::icl_core::Map< ::std::string, ::std::string> KeyValueMap;
            //KeyValueMap m_config_items;
            bool m_initialized;

            ConfigParameterList m_parameter_list;
            ConfigPositionalParameterList m_positional_parameter_list;

            typedef std::map<std::string, std::list<ConfigObserver*>> ObserverMap;
            ObserverMap m_observers;
        };
    }
}

#endif