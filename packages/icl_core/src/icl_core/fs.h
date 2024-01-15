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
 * \date    2008-01-28
 *
 * \brief   Contains global filesystem related functions,
 *          encapsulated into the icl_core::os namespace
 */
 //----------------------------------------------------------------------
#ifndef ICL_CORE_OS_FS_H_INCLUDED
#define ICL_CORE_OS_FS_H_INCLUDED

#include <memory>

#include "icl_core/ImportExport.h"

#ifdef _IC_BUILDER_DEPRECATED_STYLE_
# include "icl_core/Deprecate.h"
#endif

#ifdef _IC_BUILDER_ZLIB_
#include <zlib.h>
#endif

namespace icl_core {
	//! Namespace for operating system specific implementations.
	namespace os {

#ifdef _IC_BUILDER_ZLIB_
		/*!
		 * Zip the specified file using the gzip algorithm.
		 * Append the \a additional_extension to the original filename.
		 */
		bool ICL_CORE_IMPORT_EXPORT zipFile(const char* filename, const char* additional_extension = "");

		struct ZipFileDeleter
		{
			void operator()(gzFile file);
		};

		typedef std::unique_ptr<std::remove_pointer_t<gzFile>, ZipFileDeleter> ZipFilePtr;

		ZipFilePtr openZipFile(const char* path, const char* mode);

#endif

	}
}

#endif