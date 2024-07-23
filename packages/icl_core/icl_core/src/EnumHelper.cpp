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
 * \date    2009-03-11
 */
//----------------------------------------------------------------------

#include "EnumHelper.h"

namespace icl_core {

    bool string2Enum(const std::string& str, int32_t& value,
        const char* const* descriptions, const char* end_marker)
    {
        bool result = false;

        for (int32_t index = 0;
            (end_marker == nullptr && descriptions[index] != nullptr)
            || (end_marker != nullptr && strcmp(descriptions[index], end_marker) != 0);
            ++index)
        {
            // Return success if a matching description has been found.
            if (strcmp(str.c_str(), descriptions[index]) == 0)
            {
                value = index;
                result = true;
            }
        }

        return result;
    }

    namespace impl {
        template<typename T>
        bool string2Enum(const std::string& str, T& value,
            const std::vector<std::string>& descriptions)
        {
            bool result = false;

            for (T index = 0; index < T(descriptions.size()); ++index)
            {
                // Return success if a matching description has been found.
                if (str == descriptions[static_cast<std::size_t>(index)])
                {
                    value = index;
                    result = true;
                }
            }

            return result;
        }
    }

    bool string2Enum(const std::string& str, int32_t& value,
        const std::vector<std::string>& descriptions)
    {
        return impl::string2Enum<int32_t>(str, value, descriptions);
    }

    bool string2Enum(const std::string& str, int64_t& value,
        const std::vector<std::string>& descriptions)
    {
        return impl::string2Enum<int64_t>(str, value, descriptions);
    }
}