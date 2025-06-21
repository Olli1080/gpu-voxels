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
#include "icl_core_config/GetoptParameter.h"

namespace icl_core {
    namespace config {

        GetoptParameter::GetoptParameter(const std::string& option, std::string short_option,
                                         std::string help, bool is_prefix)
            : m_short_option(std::move(short_option)),
            m_help(std::move(help)),
            m_is_prefix(is_prefix)
        {
            if (!option.empty() && *option.rbegin() == ':')
            {
                m_option = option.substr(0, option.length() - 1);
                m_has_value = true;
            }
            else
            {
                m_option = option;
                m_has_value = false;
            }
        }
    }
}