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
 * \date    2007-05-12
 */
//----------------------------------------------------------------------
#include "icl_core_config/ConfigManager.h"

#include <cassert>

#include <iostream>
#include <ranges>

#include <tinyxml.h>

//#include "icl_core/KeyValueDirectory.hpp"
#include "icl_core_config/AttributeTree.h"
#include "icl_core_config/Config.h"
#include "icl_core_config/ConfigObserver.h"
#include "icl_core_config/GetoptParser.h"

namespace icl_core {

    // Explicit template instantiation!
    template class KeyValueDirectory<std::string>;

    namespace config {

        ConfigManager& ConfigManager::instance()
        {
            static ConfigManager instance;
            return instance;
        }

        void ConfigManager::addParameter(const ConfigParameter& parameter)
        {
            // Add to the own parameter list.
            if (!parameter.configKey().empty())
            {
                m_parameter_list.push_back(parameter);
            }

            // Delegate to Getopt.
            Getopt::instance().addParameter(parameter);
        }

        void ConfigManager::addParameter(const ConfigParameterList& parameters)
        {
            for (auto it = parameters.begin(); it != parameters.end(); ++it)
            {
                addParameter(*it);
            }
        }

        void ConfigManager::addParameter(const ConfigPositionalParameter& parameter)
        {
            // Add to the own parameter list.
            if (!parameter.configKey().empty())
            {
                m_positional_parameter_list.push_back(parameter);
            }

            // Delegate to Getopt.
            Getopt::instance().addParameter(parameter);
        }

        void ConfigManager::addParameter(const ConfigPositionalParameterList& parameters)
        {
            for (auto it = parameters.begin(); it != parameters.end(); ++it)
                addParameter(*it);
        }

        bool ConfigManager::initialize()
        {
            if (isInitialized())
            {
                std::cerr << "CONFIG WARNING: The configuration framework is already initialized!" << std::endl;
                return true;
            }

            if (Getopt::instance().paramOptPresent("configfile"))
            {
                // Read the configuration file.
                const std::string filename = Getopt::instance().paramOpt("configfile");
                if (!load(filename))
                {
                    std::cerr << "CONFIG ERROR: The configuration file '" << filename << "' could not be loaded!"
                        << std::endl;
                    return false;
                }
                insert(CONFIGFILE_CONFIG_KEY, filename);
                notify(CONFIGFILE_CONFIG_KEY);
            }

            // Check for registered parameters.
            for (const auto& it : m_parameter_list)
            {
                if (it.configKey().empty())
                    continue;

                // Fill the configuration parameter from the commandline.
                if (Getopt::instance().paramOptPresent(it.option()))
                {
                    insert(it.configKey(), Getopt::instance().paramOpt(it.option()));
                    notify(it.configKey());
                }
                // If the parameter is still not present but has a default value, then set it.
                else if (!hasKey(it.configKey()) && it.hasDefaultValue())
                {
                    insert(it.configKey(), it.defaultValue());
                    notify(it.configKey());
                }
            }

            // Check for registered positional parameters.
            for (const auto& it : m_positional_parameter_list)
            {
                if (it.configKey().empty())
                    continue;

                // Fill the configuration parameter from the commandline.
                if (Getopt::instance().paramOptPresent(it.name()))
                {
                    insert(it.configKey(), Getopt::instance().paramOpt(it.name()));
                    notify(it.configKey());
                }
                // If the parameter is still not present but has a default value, then set it.
                else if (!hasKey(it.configKey()) && it.hasDefaultValue())
                {
                    insert(it.configKey(), it.defaultValue());
                    notify(it.configKey());
                }
            }

            // Check for option parameters.
            const Getopt::KeyValueList option_params = Getopt::instance().paramPrefixOpt("config-option");
            for (const auto& it : option_params)
            {
                insert(it.m_key, it.m_value);
                notify(it.m_key);
            }

            // Optionally dump the configuration.
            if (Getopt::instance().paramOptPresent("dump-config"))
            {
                dump();
            }

            m_initialized = true;
            return true;
        }

        void ConfigManager::dump() const
        {
            std::cout << "--- BEGIN CONFIGURATION DUMP ---" << std::endl;
            ConfigIterator it = find(".*");
            while (it.next())
            {
                std::cout << it.key() << " = '" << it.value() << "'" << std::endl;
            }
            std::cout << "--- END CONFIGURATION DUMP ---" << std::endl;
        }

        ConfigManager::ConfigManager()
            : m_initialized(false)
        {
            addParameter(ConfigParameter("configfile:", "c", CONFIGFILE_CONFIG_KEY,
                "Specifies the path to the configuration file."));
            Getopt::instance().addParameter(GetoptParameter("dump-config", "dc",
                "Dump the configuration read from the configuration file."));
            Getopt::instance().addParameter(GetoptParameter("config-option:", "o",
                "Overwrite a configuration option.", true));
        }

        bool ConfigManager::load(const std::string& filename)
        {
	        const FilePath fp(filename.c_str());

            if (fp.extension() == ".AttributeTree" || fp.extension() == ".tree")
            {
                AttributeTree attribute_tree;
                const int res = attribute_tree.load(filename.c_str());
                if (res != AttributeTree::eFILE_LOAD_ERROR)
                {
                    if (res == AttributeTree::eOK)
                    {
                        readAttributeTree("", attribute_tree.root(), false);
                    }
                    return true;
                }
                else
                {
                    std::cerr << "CONFIG ERROR: Could not load configuration file '" << filename << std::endl;
                    return false;
                }
            }
            else
            {
                TiXmlDocument doc(filename.c_str());
                if (doc.LoadFile())
                {
                    TiXmlElement* root_element = doc.RootElement();
                    if (root_element != nullptr)
                    {
                        readXml("", root_element, fp, false);
                    }
                    return true;
                }
                else
                {
                    std::cerr << "CONFIG ERROR: Could not load configuration file '" << filename << "' (" << doc.ErrorRow()
                        << ", " << doc.ErrorCol() << "): " << doc.ErrorDesc() << std::endl;
                    return false;
                }
            }
        }

        void ConfigManager::readXml(const std::string& prefix, TiXmlNode* node, FilePath fp, bool extend_prefix)
        {
	        const std::string node_name(node->Value());
            std::string fq_node_name = prefix;
            if (extend_prefix)
            {
                fq_node_name = prefix + "/" + node_name;
            }

            TiXmlNode* child = node->IterateChildren(nullptr);
            while (child != nullptr)
            {
                if (child->Type() == TiXmlNode::TINYXML_ELEMENT)
                {
                    if (strcmp(child->Value(), "INCLUDE") == 0)
                    {
	                    const auto* child_element = dynamic_cast<TiXmlElement*>(child);
                        assert(child_element != nullptr);
                        const char* included_file = child_element->GetText();
                        if (included_file != nullptr)
                        {
                            load(fp.path().string() + included_file);
                        }
                    }
                    else
                    {
                        readXml(fq_node_name, child, fp);
                    }
                }
                else if (child->Type() == TiXmlNode::TINYXML_TEXT)
                {
                    insert(fq_node_name, child->Value());
                    notify(fq_node_name);
                }

                child = node->IterateChildren(child);
            }
        }

        void ConfigManager::readAttributeTree(const std::string& prefix, AttributeTree* at, bool extend_prefix)
        {
            std::string node_name;
            if (at->getDescription() != nullptr)
            {
                node_name = at->getDescription();
            }
            std::string fq_node_name = prefix;
            if (extend_prefix)
            {
                fq_node_name = prefix + "/" + node_name;
            }

            if (!at->isComment() && at->attribute() != nullptr)
            {
                insert(fq_node_name, at->attribute());
                notify(fq_node_name);
            }

            AttributeTree* child = at->firstSubTree();
            while (child != nullptr)
            {
                readAttributeTree(fq_node_name, child);
                child = at->nextSubTree(child);
            }
        }

        void ConfigManager::registerObserver(ConfigObserver* observer, const std::string& key)
        {
            assert(observer && "Null must not be passed as config observer");

            m_observers[key].push_back(observer);

            if (key.empty())
            {
                ConfigIterator iter = icl_core::config::ConfigManager::instance().find(".*");
                while (iter.next())
                {
                    observer->valueChanged(iter.key());
                }
            }
            else if (find(key).next())
            {
                observer->valueChanged(key);
            }
        }

        void ConfigManager::unregisterObserver(ConfigObserver* observer)
        {
            assert(observer && "Null must not be passed as config observer");

            for (auto& configList : m_observers | std::views::values)
                configList.remove(observer);
        }

        void ConfigManager::notify(const std::string& key) const
        {
            std::list<ConfigObserver*> observers;
            auto find_it = m_observers.find(key);
            if (find_it != m_observers.end())
            {
                observers.insert(observers.end(), find_it->second.begin(), find_it->second.end());
            }
            find_it = m_observers.find("");
            if (find_it != m_observers.end())
            {
                observers.insert(observers.end(), find_it->second.begin(), find_it->second.end());
            }

            for (const auto& observer : observers)
                observer->valueChanged(key);
        }

    }
}