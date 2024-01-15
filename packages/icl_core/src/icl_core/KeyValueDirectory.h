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

#ifndef ICL_CORE_KEY_VALUE_DIRECTORY_H_INCLUDED
#define ICL_CORE_KEY_VALUE_DIRECTORY_H_INCLUDED

#include <map>
#include <regex>

#ifdef _IC_BUILDER_DEPRECATED_STYLE_
# include "icl_core/Deprecate.h"
#endif

namespace icl_core {

	template <typename T>
	class KeyValueDirectoryIterator;

	/*! Implements a lightweight key/value directory.
	 */
	template <typename T>
	class KeyValueDirectory
	{
		friend class KeyValueDirectoryIterator<T>;
	public:

		/*! Finds all entries which match the specified \a query.  Boost
		 *  regular expressions are allowed for the query.
		 *
		 *  \returns An iterator which iterates over all entries, which
		 *           match the specified \a query.
		 */
		KeyValueDirectoryIterator<T> find(const std::string& query) const;

		/*! Get a \a value for the specified \a key.
		 *
		 *  \returns \c true if a configuration value for the key exists, \c
		 *           false otherwise.
		 */
		bool get(const std::string& key, T& value) const;

		/*! Check if the \a key exists.
		 */
		[[nodiscard]] bool hasKey(const std::string& key) const;

		/*! Insert a new \a key / \a value pair.
		 *
		 *  \returns \c true if a new element was inserted, \c false if an
		 *           existing element was replaced.
		 */
		bool insert(const std::string& key, const T& value);

	private:

		typedef std::map<std::string, T> KeyValueMap;
		KeyValueMap m_items;
	};

	/*!
	 * Implements an iterator for regular expression querys to
	 * a key/value directory.
	 */
	template <typename T>
	class KeyValueDirectoryIterator
	{
	public:
		/*!
		 * Create a new iterator for the \a query on the \a directory.
		 */
		KeyValueDirectoryIterator(const std::string& query, const KeyValueDirectory<T>* directory);

		/*!
		 * Get the key of the current match result.
		 */
		[[nodiscard]] std::string key() const;

		/*!
		 * Get the match group at the specified \a index.
		 * \n
		 * Remark: Match groups are the equivalent of Perl's (or sed's)
		 * $n references.
		 */
		[[nodiscard]] std::string matchGroup(size_t index) const;

		/*!
		 * Move to the next query result.
		 *
		 * \returns \a false if no next query result exists.
		 */
		bool next();

		/*!
		 * Resets the iterator. You have to call Next() to move it
		 * to the first matching configuration entry.
		 */
		void reset();

		/*!
		 * Get the value of the current match result.
		 */
		const T& value() const;

	private:

		const KeyValueDirectory<T>* m_directory;
		std::regex m_query;
		std::match_results<std::string::const_iterator> m_current_results;

		typename KeyValueDirectory<T>::KeyValueMap::const_iterator m_current_entry;
		bool m_reset;
	};
}

#include "icl_core/KeyValueDirectory.hpp"

#endif