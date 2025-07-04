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
 * \date    2012-01-24
 *
 */
//----------------------------------------------------------------------
#ifndef ICL_CORE_CONFIG_MEMBER_VALUE_H_INCLUDED
#define ICL_CORE_CONFIG_MEMBER_VALUE_H_INCLUDED

#include "icl_core/RemoveMemberPointer.h"
#include "icl_core_config/ConfigManager.h"
#include "icl_core_config/MemberValueIface.h"
#include "icl_core_config/Util.h"

#include <string>
#include <functional>


#define MEMBER_VALUE_1(suffix, cls, member1)                                                   \
  (new icl_core::config::MemberValue<                                                          \
     icl_core::RemoveMemberPointer<decltype(&cls::member1)>::Type, cls>(         \
     suffix,                                                                                   \
     std::bind(&cls::member1, std::placeholders::_1)))

#define MEMBER_VALUE_2(suffix, cls, member1, member2)                                          \
  (new icl_core::config::MemberValue<                                                          \
     icl_core::RemoveMemberPointer<decltype(                                     \
       &icl_core::RemoveMemberPointer<decltype(                                  \
         &cls::member1)>::Type::member2)>::Type, cls>(                                         \
     suffix,                                                                                   \
     std::bind(                                                                      \
       &icl_core::RemoveMemberPointer<decltype(&cls::member1)>::Type::member2,   \
       std::bind(&cls::member1, std::placeholders::_1))))

#define MEMBER_VALUE_3(suffix, cls, member1, member2, member3)                                 \
  (new icl_core::config::MemberValue<                                                          \
     icl_core::RemoveMemberPointer<decltype(                                     \
       &icl_core::RemoveMemberPointer<decltype(                                  \
         &icl_core::RemoveMemberPointer<decltype(                                \
           &cls::member1)>::Type::member2)>::Type::member3)>::Type, cls>(                      \
     suffix,                                                                                   \
     std::bind(                                                                      \
       &icl_core::RemoveMemberPointer<decltype(                                  \
         &icl_core::RemoveMemberPointer<decltype(                                \
           &cls::member1)>::Type::member2)>::Type::member3,                                    \
       std::bind(                                                                    \
         &icl_core::RemoveMemberPointer<decltype(&cls::member1)>::Type::member2, \
         std::bind(&cls::member1, std::placeholders::_1)))))


namespace icl_core {
    namespace config {

        template<typename T, typename Q>
        class MemberValue : public impl::MemberValueIface<Q>
        {
        public:
            MemberValue(std::string config_suffix,
                        std::function<T& (Q&)> accessor)
                : m_config_suffix(std::move(config_suffix)),
                m_accessor(accessor)
            {
            }

            ~MemberValue() override = default;

            bool get(std::string const& key, Q& value) const override
            {
                bool result = false;
                if (ConfigManager::instance().get(key, m_str_value))
                {
                    try
                    {
                        m_accessor(value) = impl::hexical_cast<T>(m_str_value);
                        result = true;
                    }
                    catch (...)
                    {
                        result = false;
                    }
                }
                else
                {
                    result = false;
                }
                return result;
            }

            std::string getSuffix() const override { return m_config_suffix; }
            std::string getStringValue() const override { return m_str_value; }

        private:
            std::string m_config_suffix;
            boost::function<T& (Q&)> m_accessor;
            mutable std::string m_str_value;
        };

        //! Template specialization for boolean MemberValues.
        template<typename Q>
        class MemberValue<bool, Q> : public impl::MemberValueIface<Q>
        {
        public:
            MemberValue(std::string const& config_suffix,
                boost::function<bool& (Q&)> accessor)
                : m_config_suffix(config_suffix),
                m_accessor(accessor)
            {
            }

            ~MemberValue() override {}

            bool get(std::string const& key, Q& value) const override
            {
                bool result = false;
                if (ConfigManager::instance().get(key, m_str_value))
                {
                    try
                    {
                        m_accessor(value) = impl::strict_bool_cast(m_str_value);
                        result = true;
                    }
                    catch (...)
                    {
                        result = false;
                    }
                }
                else
                {
                    result = false;
                }
                return result;
            }

            std::string getSuffix() const override { return m_config_suffix; }
            std::string getStringValue() const override { return m_str_value; }

        private:

            std::string m_config_suffix;
            std::function<bool& (Q&)> m_accessor;
            mutable std::string m_str_value;
        };

    }
}

/*
template<typename cls, typename member1>
auto MEMBER_VALUE_1(auto suffix)
{
    return new icl_core::config::MemberValue<
        typename icl_core::RemoveMemberPointer<decltype(&cls::member1)>::Type, cls>(
            suffix,
            std::bind(&cls::member1, std::placeholders::_1));
}

template<typename cls>
auto MEMBER_VALUE_2(auto suffix, auto member1, auto member2)
{
    return new icl_core::config::MemberValue<
        typename icl_core::RemoveMemberPointer<decltype(
            &icl_core::RemoveMemberPointer<decltype(
                &cls::member1)>::Type::member2)>::Type, cls>(
                    suffix,
                    std::bind(
                        &icl_core::RemoveMemberPointer<decltype(&cls::member1)>::Type::member2,
                        std::bind(&cls::member1, std::placeholders::_1)));
}

template<typename cls>
auto MEMBER_VALUE_3(auto suffix, auto member1, auto member2, auto member3)
{
    return new icl_core::config::MemberValue<
        typename icl_core::RemoveMemberPointer<decltype(
            &icl_core::RemoveMemberPointer<decltype(
                &icl_core::RemoveMemberPointer<decltype(
                    &cls::member1)>::Type::member2)>::Type::member3)>::Type, cls>(
                        suffix,
                        std::bind(
                            &icl_core::RemoveMemberPointer<decltype(
                                &icl_core::RemoveMemberPointer<decltype(
                                    &cls::member1)>::Type::member2)>::Type::member3,
                            std::bind(
                                &icl_core::RemoveMemberPointer<decltype(&cls::member1)>::Type::member2,
                                std::bind(&cls::member1, std::placeholders::_1))));
                                */

#endif
