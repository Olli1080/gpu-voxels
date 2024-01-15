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
#include "icl_core_config/ConfigParameter.h"

namespace icl_core {
    namespace config {

        ConfigParameter::ConfigParameter(const std::string& option, const std::string& short_option,
                                         std::string config_key, const std::string& help,
            const std::string& default_value)
            : GetoptParameter(option, short_option,
                default_value.empty() ? help : help + "\n(defaults to " + default_value + ")"),
            m_config_key(std::move(config_key)),
            m_default_value(default_value)
        { }

    }
}