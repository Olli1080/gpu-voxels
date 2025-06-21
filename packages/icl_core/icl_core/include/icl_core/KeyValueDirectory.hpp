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
 * \date    2008-11-04
 *
 * \brief   Contains KeyValueDirectory
 *
 * Implements a lightweight key/value directory.
 *
 */
//----------------------------------------------------------------------

#ifndef ICL_CORE_KEY_VALUE_DIRECTORY_HPP_INCLUDED
#define ICL_CORE_KEY_VALUE_DIRECTORY_HPP_INCLUDED

#include <icl_core/KeyValueDirectory.h>

namespace icl_core {

    template <typename T>
    KeyValueDirectoryIterator<T> KeyValueDirectory<T>::find(const std::string& query) const
    {
        return KeyValueDirectoryIterator<T>(query, this);
    }

    template <typename T>
    bool KeyValueDirectory<T>::get(const std::string& key, T& value) const
    {
        typename KeyValueMap::const_iterator find_it = m_items.find(key);
        if (find_it != m_items.end())
        {
            value = find_it->second;
            return true;
        }
        return false;
    }

    template <typename T>
    bool KeyValueDirectory<T>::hasKey(const std::string& key) const
    {
        typename KeyValueMap::const_iterator find_it = m_items.find(key);
        return find_it != m_items.end();
    }

    template <typename T>
    bool KeyValueDirectory<T>::insert(const std::string& key, const T& value)
    {
        typename KeyValueMap::const_iterator find_it = m_items.find(key);
        m_items[key] = value;
        return find_it == m_items.end();
    }

    template <typename T>
    KeyValueDirectoryIterator<T>::KeyValueDirectoryIterator(const std::string& query,
        const KeyValueDirectory<T>* directory)
        : m_directory(directory),
        m_query(query)
    {
        reset();
    }

    template <typename T>
    std::string KeyValueDirectoryIterator<T>::key() const
    {
        return m_current_entry->first;
    }

    template <typename T>
    std::string KeyValueDirectoryIterator<T>::matchGroup(size_t index) const
    {
        if (index < m_current_results.size())
            return m_current_results[index];

        return "";
    }

    template <typename T>
    bool KeyValueDirectoryIterator<T>::next()
    {
        // If the iterator has been reset (or has just been initialized)
        // we move to the first element.
        if (m_reset == true)
        {
            m_reset = false;
            m_current_entry = m_directory->m_items.begin();
        }
        // Otherwise move to the next iterator position.
        else
        {
            ++m_current_entry;
        }

        // Check if the current iterator position matches the query.
        while (m_current_entry != m_directory->m_items.end() &&
            !std::regex_match(m_current_entry->first, m_current_results, m_query))
        {
            ++m_current_entry;
        }

        // Check if there is an element left.
        return m_current_entry != m_directory->m_items.end();
    }

    template <typename T>
    void KeyValueDirectoryIterator<T>::reset()
    {
        m_reset = true;
    }

    template <typename T>
    const T& KeyValueDirectoryIterator<T>::value() const
    {
        return m_current_entry->second;
    }
}

#endif