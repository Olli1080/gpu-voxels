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
 * \date    2007-12-07
 *
 * \brief   Contains ConfigIterator.
 *
 */
 //----------------------------------------------------------------------
#ifndef ICL_CORE_CONFIG_CONFIG_ITERATOR_H_INCLUDED
#define ICL_CORE_CONFIG_CONFIG_ITERATOR_H_INCLUDED

#include "icl_core/KeyValueDirectory.h"
#include "icl_core_config/ImportExport.h"

namespace icl_core {
	namespace config {

		typedef icl_core::KeyValueDirectoryIterator<std::string> ConfigIterator;

	}
}

#endif