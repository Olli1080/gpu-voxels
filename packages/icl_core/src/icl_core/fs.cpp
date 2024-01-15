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
 * \date    2009-09-08
 *
 * \brief   Contains global filesystem functions
 *
 */
//----------------------------------------------------------------------
#include <string>
#include "icl_core/fs.h"

#ifdef _IC_BUILDER_ZLIB_
#include <iostream>
#endif

namespace icl_core {
    namespace os {

#ifdef _IC_BUILDER_ZLIB_
        bool zipFile(const char* filename, const char* additional_extension)
        {
            bool ret = true;
            const std::string gzip_file_name = std::string(filename) + additional_extension + ".gz";
            const gzFile unzipped_file = gzopen(filename, "rb");
            const gzFile zipped_file = gzopen(gzip_file_name.c_str(), "wb");

            if (unzipped_file != nullptr && zipped_file != nullptr)
            {
	            char big_buffer[0x1000];
	            int bytes_read = gzread(unzipped_file, big_buffer, 0x1000);
                while (bytes_read > 0)
                {
                    if (gzwrite(zipped_file, big_buffer, bytes_read) != bytes_read)
                    {
                        std::cerr << "ZipFile(" << filename << "->" << gzip_file_name << ") Error on writing." << std::endl;
                        ret = false;
                        break;
                    }

                    bytes_read = gzread(unzipped_file, big_buffer, 0x1000);
                }
            }

            if (unzipped_file != nullptr)
                gzclose(unzipped_file);

            if (zipped_file != nullptr)
                gzclose(zipped_file);

            return ret;
        }

        ZipFilePtr openZipFile(const char* path, const char* mode)
        {
            return ZipFilePtr(gzopen(path, mode), {});
        }

        void ZipFileDeleter::operator()(gzFile file)
        {
            gzclose(file);
            file = nullptr;
        }

#endif
    }
}