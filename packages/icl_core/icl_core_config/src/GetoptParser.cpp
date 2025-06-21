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
 */
//----------------------------------------------------------------------
#include "icl_core_config/GetoptParser.h"

#include <iostream>
#include <cstdlib>
#include <regex>
#include <ranges>

namespace icl_core {
    namespace config {

        Getopt& Getopt::instance()
        {
            static Getopt instance;
            return instance;
        }

        void Getopt::activateExtraCmdParams(const std::string& delimiter)
        {
            m_extra_cmd_param_activated = true;
            m_extra_cmd_param_delimiter = delimiter;
        }

        void Getopt::addParameter(const GetoptParameter& parameter)
        {
            if (parameter.isPrefixOption())
            {
                m_prefix_parameters.emplace(parameter.option(), parameter);
                if (!parameter.shortOption().empty())
                    m_short_prefix_parameters.emplace(parameter.shortOption(), parameter);
            }
            else
            {
                m_parameters.emplace(parameter.option(), parameter);
                if (!parameter.shortOption().empty())
                    m_short_parameters.emplace(parameter.shortOption(), parameter);
            }
        }

        void Getopt::addParameter(const GetoptParameterList& parameters)
        {
            for (const auto& opt : parameters)
                addParameter(opt);
        }

        void Getopt::addParameter(const GetoptPositionalParameter& parameter)
        {
            if (parameter.isOptional())
            {
                m_optional_positional_parameters.push_back(parameter);
            }
            else
            {
                m_required_positional_parameters.push_back(parameter);
            }
        }

        void Getopt::addParameter(const GetoptPositionalParameterList& parameters)
        {
            for (auto opt_it = parameters.begin();
                opt_it != parameters.end(); ++opt_it)
            {
                addParameter(*opt_it);
            }
        }

        bool Getopt::initialize(int& argc, char* argv[], CommandLineCleaning cleanup,
            ParameterRegistrationCheck registration_check)
        {
            if (argc == 0)
                return false;

            if (isInitialized())
            {
                std::cerr << "GETOPT WARNING: The commandline option framework is already initialized!" << std::endl;
                return true;
            }

            // Store the full argc and argv
            m_argc = argc;
            m_argv = argv;


            // Store the program name.
            m_program_name = argv[0];

            // Store all parameters in a temporary list.
            std::list<std::string> arguments;
            for (int index = 1; index < argc; ++index)
                arguments.emplace_back(argv[index]);

            // Scan the commandline parameters and check for
            // registered options.
            size_t positional_parameters_counter = 0;
            bool extra_cmd_params_reached = false;
            std::regex long_opt_regex("--([^-][^=]*)(=(.*))?");
            std::regex short_opt_regex("-([^-].*)");
            std::smatch mres;
            for (auto arg_it = arguments.begin();
                arg_it != arguments.end(); ++arg_it)
            {
                if (extra_cmd_params_reached)
                {
                    m_extra_cmd_param.push_back(*arg_it);
                }
                else if (std::regex_match(*arg_it, mres, long_opt_regex))
                {
                    // Found a long option parameter!
                    std::string name = mres[1];
                    ParameterMap::const_iterator find_it = m_parameters.find(name);
                    if (find_it != m_parameters.end())
                    {
                        if (find_it->second.hasValue())
                        {
                            // According to the regular expression the value has to be
                            // the 3rd (and last) match result.
                            if (mres.size() == 4)
                            {
                                m_param_opt[name] = mres[3];
                            }
                            else
                            {
                                std::cerr << "Found option " << *arg_it << " but the value is missing." << std::endl;
                                printHelp();
                                return false;
                            }
                        }
                        else
                        {
                            m_param_opt[name] = "yes";
                        }
                    }
                    else
                    {
                        // Parameter not found in the list of configured parameters.
                        // Check if a matching prefix option has been registered.
                        bool found = false;
                        std::smatch prefix_res;
                        for (ParameterMap::const_iterator prefix_it = m_prefix_parameters.begin();
                            !found && prefix_it != m_prefix_parameters.end(); ++prefix_it)
                        {
                            if (std::regex_match(name, prefix_res, std::regex(prefix_it->first + "(.*)")))
                            {
                                found = true;

                                if (prefix_it->second.hasValue())
                                {
                                    if (mres.size() == 4)
                                    {
                                        m_prefix_param_opt[prefix_it->first].push_back(KeyValue(prefix_res[1], mres[3]));
                                    }
                                    else
                                    {
                                        std::cerr << "Found prefix option " << name << " but the value is missing." << std::endl;
                                        printHelp();
                                        return false;
                                    }
                                }
                                else
                                {
                                    m_prefix_param_opt[prefix_it->first].push_back(KeyValue(prefix_res[1], "yes"));
                                }
                            }
                        }

                        // Also not a prefix option!
                        if (!found)
                        {
                            if (registration_check == ePRC_Strict)
                            {
                                std::cerr << "Found unknown option " << *arg_it << "." << std::endl;
                                printHelp();
                                return false;
                            }
                            else
                            {
                                m_param_non_opt.push_back(*arg_it);
                            }
                        }
                    }
                }
                else if (std::regex_match(*arg_it, mres, short_opt_regex))
                {
                    // Found a short option parameter!
                    std::string name = mres[1];
                    ParameterMap::const_iterator find_it = m_short_parameters.find(name);
                    if (find_it != m_short_parameters.end())
                    {
                        if (find_it->second.hasValue())
                        {
                            // The value is the next commandline argument.
                            ++arg_it;
                            if (arg_it == arguments.end())
                            {
                                std::cerr << "Found option -" << name << " but the value is missing." << std::endl;
                                printHelp();
                                return false;
                            }
                            else
                            {
                                m_param_opt[find_it->second.option()] = *arg_it;
                            }
                        }
                        else
                        {
                            m_param_opt[find_it->second.option()] = "yes";
                        }
                    }
                    else
                    {
                        // Parameter not found in the list of configured parameters.
                        // Check if a matching prefix option has been registered.
                        bool found = false;
                        std::smatch prefix_res;
                        for (ParameterMap::const_iterator prefix_it = m_short_prefix_parameters.begin();
                            !found && prefix_it != m_short_prefix_parameters.end(); ++prefix_it)
                        {
                            if (std::regex_match(name, prefix_res, std::regex(prefix_it->first + "(.*)")))
                            {
                                found = true;

                                if (prefix_it->second.hasValue())
                                {
                                    // The value is the next commandline argument.
                                    ++arg_it;
                                    if (arg_it == arguments.end())
                                    {
                                        std::cerr << "Found prefix option " << name << " but the value is missing." << std::endl;
                                        printHelp();
                                        return false;
                                    }
                                    else
                                    {
                                        m_prefix_param_opt[prefix_it->second.option()].push_back(KeyValue(prefix_res[1], *arg_it));
                                    }
                                }
                                else
                                {
                                    m_prefix_param_opt[prefix_it->second.option()].push_back(KeyValue(prefix_res[1], "yes"));
                                }
                            }
                        }

                        // Also not a prefix option!
                        if (!found)
                        {
                            if (registration_check == ePRC_Strict)
                            {
                                std::cerr << "Found unknown option " << *arg_it << "." << std::endl;
                                printHelp();
                                return false;
                            }
                            else
                            {
                                m_param_non_opt.push_back(*arg_it);
                            }
                        }
                    }
                }
                else if (m_extra_cmd_param_activated && *arg_it == m_extra_cmd_param_delimiter)
                {
                    extra_cmd_params_reached = true;
                }
                else if (positional_parameters_counter < m_required_positional_parameters.size())
                {
                    // Found a required positional parameter
                    const GetoptPositionalParameter& param = m_required_positional_parameters[positional_parameters_counter];
                    m_param_opt[param.name()] = *arg_it;
                    positional_parameters_counter++;
                }
                else if (positional_parameters_counter < m_required_positional_parameters.size() + m_optional_positional_parameters.size())
                {
                    // Found an optional positional parameter
                    const GetoptPositionalParameter& param = m_optional_positional_parameters[positional_parameters_counter - m_required_positional_parameters.size()];
                    m_param_opt[param.name()] = *arg_it;
                    positional_parameters_counter++;
                }
                else if (positional_parameters_counter >= m_required_positional_parameters.size() + m_optional_positional_parameters.size())
                {
                    /*! \note this would be nice but breaks backwards compatibility
                     *  where people use ePRC_Strict but want to use unregistered
                     *  positional parameters.
                     */
                     //      if (registration_check == ePRC_Strict)
                     //      {
                     //        std::cerr << "Found unknown positional parameter \"" << *arg_it << "\" and registration_check is ePRC_Strict. Aborting." << std::endl;
                     //        printHelp();
                     //        return false;
                     //      }
                     //      else
                    {
                        m_param_non_opt.push_back(*arg_it);
                    }
                }
            }

            // Check if all required positional parameters are given
            if (positional_parameters_counter < m_required_positional_parameters.size())
            {
                std::cerr << "Not all required parameters are given. Aborting." << std::endl;
                printHelp();
                exit(0);
            }

            // Check if the help text has to be printed.
            if (m_param_opt.contains("help"))
            {
                printHelp();
                exit(0);
            }

            // Remove all option parameters from the "real" commandline.
            if (cleanup == eCLC_Cleanup)
            {
                int check = 1;
                while (check < argc)
                {
                    const auto find_it =
                        std::ranges::find(m_param_non_opt, std::string(argv[check]));
                    if (find_it == m_param_non_opt.end())
                    {
                        for (int move = check + 1; move < argc; ++move)
                        {
                            argv[move - 1] = argv[move];
                        }
                        --argc;
                    }
                    else
                    {
                        ++check;
                    }
                }
            }

            return true;
        }

        int& Getopt::argc()
        {
            return m_argc;
        }

        char** Getopt::argv() const
        {
            return m_argv;
        }

        std::string Getopt::extraCmdParam(size_t index) const
        {
            return m_extra_cmd_param[index];
        }

        size_t Getopt::extraCmdParamCount() const
        {
            return m_extra_cmd_param.size();
        }

        std::string Getopt::paramOpt(const std::string& name) const
        {
            const auto find_it = m_param_opt.find(name);
            if (find_it != m_param_opt.end())
                return find_it->second;

            return "";
        }

        bool Getopt::paramOptPresent(const std::string& name) const
        {
            return m_param_opt.contains(name);
        }

        Getopt::KeyValueList Getopt::paramPrefixOpt(const std::string& prefix) const
        {
            const auto find_it = m_prefix_param_opt.find(prefix);
            if (find_it != m_prefix_param_opt.end())
                return find_it->second;

            return {};
        }

        bool Getopt::paramPrefixOptPresent(const std::string& prefix) const
        {
            return m_prefix_param_opt.contains(prefix);
        }

        std::string Getopt::paramNonOpt(size_t index) const
        {
            if (index < m_param_non_opt.size())
                return m_param_non_opt.at(index);

            return "";
        }

        size_t Getopt::paramNonOptCount() const
        {
            return m_param_non_opt.size();
        }

        std::string Getopt::programName() const
        {
            return m_program_name;
        }

        std::string Getopt::programVersion() const
        {
            return m_program_version;
        }

        void Getopt::setProgramVersion(std::string const& version)
        {
            m_program_version = version;
        }

        std::string Getopt::programDescription() const
        {
            return m_program_description;
        }

        void Getopt::setProgramDescription(std::string const& description)
        {
            m_program_description = description;
        }

        void Getopt::printHelp() const
        {
            // prepare list of all positional parameters
            GetoptPositionalParameterList positional_parameters = m_required_positional_parameters;
            std::ranges::copy(m_optional_positional_parameters, std::back_inserter(positional_parameters));

            std::cerr << programName();
            if (!programVersion().empty())
            {
                std::cerr << " (version " << programVersion() << ")";
            }
            std::cerr << std::endl << std::endl;

            std::cerr << "Usage: ";
            std::cerr << programName();

            std::cerr << " [OPTIONS...]";

            for (const auto& param : positional_parameters)
            {
                if (param.isOptional())
                {
                    std::cerr << " [<" << param.name() << ">]";
                }
                else
                {
                    std::cerr << " <" << param.name() << ">";
                }
            }

            std::cerr << std::endl << std::endl << programDescription() << std::endl << std::endl;

            if (!positional_parameters.empty())
            {
                std::cerr << "Positional Parameters:" << std::endl;

                for (const auto& param : positional_parameters)
                {
                    std::cerr << "  <" << param.name() << ">" << ":" << std::endl << "\t"
                        << std::regex_replace(param.help(), std::regex("\\n"), "\n\t")
                        << std::endl;
                }
                std::cerr << std::endl;
            }

            for (int i = 0; i < 2; ++i)
            {
                std::cerr << (i == 0 ? "Generic options:" : "Options:") << std::endl;
                for (const auto& val : m_parameters | std::views::values)
                {
                    bool const is_generic =
                        val.option() == "configfile" ||
                        val.option() == "dump-config" ||
                        val.option() == "help" ||
                        val.option() == "log-level" ||
                        val.option() == "quick-debug";
                    if (!i == is_generic)
                    {
                        std::cerr << "  ";
                        // Short option.
                        if (!val.shortOption().empty())
                        {
                            std::cerr << "-" << val.shortOption();
                            if (val.hasValue())
                            {
                                std::cerr << " <value>";
                            }
                            std::cerr << ", ";
                        }

                        // Long option.
                        std::cerr << "--" << val.option();
                        if (val.hasValue())
                        {
                            std::cerr << "=<value>";
                        }

                        // Help text
                        std::cerr << ":" << std::endl << "\t"
                            << std::regex_replace(val.help(), std::regex("\\n"), "\n\t")
                            << std::endl;
                    }
                }
                std::cerr << std::endl;
            }
        }

        Getopt::Getopt()
            : m_extra_cmd_param_activated(false),
            m_initialized(false)
        {
            addParameter(GetoptParameter("help", "h", "Print commandline help."));
        }

    }
}
