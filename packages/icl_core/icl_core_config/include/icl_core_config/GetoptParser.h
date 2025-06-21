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
 * \brief   Contains Getopt.
 *
 */
//----------------------------------------------------------------------
#ifndef ICL_CORE_CONFIG_GETOPT_PARSER_H_INCLUDED
#define ICL_CORE_CONFIG_GETOPT_PARSER_H_INCLUDED

#include <string>
#include <list>
#include <map>

#include "icl_core/Deprecate.h"
#include "icl_core_config/ImportExport.h"
#include "icl_core_config/GetoptParameter.h"
#include "icl_core_config/GetoptPositionalParameter.h"

namespace icl_core {
    namespace config {

        /*! \brief Handles commandline parameters.
         *
         *  Getopt reads all commandline parameters and extracts commandline
         *  options (both key/value and simple ones).  All parameters, which
         *  which were not identified as option parameters can be accessed as
         *  non-option parameters.
         *
         *  Commandline options have to be registered with calls to
         *  AddParameter(). Then parsing the commandline is initialized by a
         *  call to Initialize() with the commandline as arguments.
         *
         *  Getopt is implemented as a singleton so that it can be used from
         *  everywhere after it has been initialized once.
         */
        class ICL_CORE_CONFIG_IMPORT_EXPORT Getopt
        {
        public:
            enum ParameterRegistrationCheck
            {
                ePRC_Strict, //!< all options have to be registered
                ePRC_Relaxed //!< options not registered are ignored
            };

            enum CommandLineCleaning
            {
                eCLC_None,   //!< command line options are left untouched
                eCLC_Cleanup //!< known command line options are removed
            };

            struct KeyValue
            {
                KeyValue(std::string key, std::string value)
                    : m_key(std::move(key)),
                    m_value(std::move(value))
                { }

                std::string m_key;
                std::string m_value;
            };
            typedef std::list<KeyValue> KeyValueList;

            /*! Get the singleton instance.
             */
            static Getopt& instance();

            /*! Active extra command parameters. They are delimited from regular
             *  commandline parameters using the \a delimiter and run from there
             *  to the end of the commandline.
             */
            void activateExtraCmdParams(const std::string& delimiter = "--");

            /*! Adds a parameter to the list of commandline options.
             */
            void addParameter(const GetoptParameter& parameter);

            /*! Adds a list of parameters to the list of commandline options.
             */
            void addParameter(const GetoptParameterList& parameters);

            /*! Adds a positional parameter to the list of commandline options.
             */
            void addParameter(const GetoptPositionalParameter& parameter);

            /*! Adds a list of positional parameters to the list of commandline
             *  options.
             */
            void addParameter(const GetoptPositionalParameterList& parameters);

            /*! Initializes Getopt with a commandline.
             *
             * \param argc Number of command line options in argv
             * \param argv Command line options
             * \param cleanup Can be eCLC_None to leave argc and argv untouched
             *        or eCLC_Cleanup to remove known options from argv and
             *        decrease argc appropriately
             * \param registration_check When encountering a not registered
             *        command line option, the value ePRC_Strict causes the
             *        initialization to fail, while ePRC_Relaxed accepts it
             *        anyway
             */
            bool initialize(int& argc, char* argv[], CommandLineCleaning cleanup = eCLC_None,
                ParameterRegistrationCheck registration_check = ePRC_Strict);

            /*! Returns \c true if Getopt has already been initialized.
             */
            [[nodiscard]] bool isInitialized() const { return m_initialized; }

            //! Get the original argc
            int& argc();

            //! Get the original argv
            [[nodiscard]] char** argv() const;

            /*! Get the extra command parameter at \a index.
             */
            [[nodiscard]] std::string extraCmdParam(size_t index) const;

            /*! Get the number of extra command parameters.
             */
            [[nodiscard]] size_t extraCmdParamCount() const;

            /*! Get the value of the commandline option \a name.
             *
             *  \returns An empty std::string if the option has not been set,
             *           otherwise the value of the option.
             */
            [[nodiscard]] std::string paramOpt(const std::string& name) const;

            /*! Checks if the option \a name is present.
             */
            [[nodiscard]] bool paramOptPresent(const std::string& name) const;

            /*! Get the list of defined suffixes for the specified \a prefix.
             */
            [[nodiscard]] KeyValueList paramPrefixOpt(const std::string& prefix) const;

            /*! Check in a prefix option is present.
             */
            [[nodiscard]] bool paramPrefixOptPresent(const std::string& prefix) const;

            /*! Get the non-option parameter at the specified \a index.
             *
             *  \returns An empty std::string if no such parameter exists.
             */
            [[nodiscard]] std::string paramNonOpt(size_t index) const;

            /*! Get the number of non-option parameters.
             */
            [[nodiscard]] size_t paramNonOptCount() const;

            /*! Get the program name.
             */
            [[nodiscard]] std::string programName() const;

            /*! Get the program version.
             */
            [[nodiscard]] std::string programVersion() const;

            /*! Set the program version.
             */
            void setProgramVersion(std::string const& version);

            /*! Set the program description, a short std::string describing the program's purpose.
             */
            void setProgramDescription(std::string const& description);

            /*! Get the program description.
             */
            [[nodiscard]] std::string programDescription() const;

            /*! Prints the help text.
             */
            void printHelp() const;

        private:

            Getopt();

            typedef std::map<std::string, GetoptParameter> ParameterMap;
            ParameterMap m_parameters;
            ParameterMap m_prefix_parameters;
            ParameterMap m_short_parameters;
            ParameterMap m_short_prefix_parameters;
            GetoptPositionalParameterList m_required_positional_parameters;
            GetoptPositionalParameterList m_optional_positional_parameters;
            bool m_extra_cmd_param_activated;
            std::string m_extra_cmd_param_delimiter;

            bool m_initialized;

            int m_argc;
            char** m_argv;
            std::string m_program_name;
            std::string m_program_version;
            std::string m_program_description;
            std::vector<std::string> m_param_non_opt;
            std::map<std::string, std::string> m_param_opt;
            std::map<std::string, KeyValueList> m_prefix_param_opt;
            std::vector<std::string> m_extra_cmd_param;
        };

    }
}
#endif