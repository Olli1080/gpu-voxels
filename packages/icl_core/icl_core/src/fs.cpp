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
#include "fs.h"

#ifdef USE_ZLIB
#include <string>
#include <iostream>
#endif

namespace icl_core {

#ifdef USE_ZLIB
        bool zipFile(const std::filesystem::path& filename, const std::string& additional_extension)
        {
            bool ret = true;
            std::filesystem::path gzip_filename = filename;
            gzip_filename.replace_filename(filename.filename().string() + additional_extension + ".gz");

            const auto unzipped_file = openZipFile(filename, "rb");
            const auto zipped_file = openZipFile(gzip_filename, "wb");

            if (unzipped_file != nullptr && zipped_file != nullptr)
            {
	            char big_buffer[0x1000];
	            int bytes_read = gzread(unzipped_file.get(), big_buffer, 0x1000);
                while (bytes_read > 0)
                {
                    if (gzwrite(zipped_file.get(), big_buffer, bytes_read) != bytes_read)
                    {
                        std::cerr << "ZipFile(" << filename << "->" << gzip_filename.string() << ") Error on writing.\n";
                        ret = false;
                        break;
                    }

                    bytes_read = gzread(unzipped_file.get(), big_buffer, 0x1000);
                }
            }
            return ret;
        }

        ZipFilePtr openZipFile(const std::filesystem::path& path, const char* mode)
        {
            return ZipFilePtr(gzopen(path.string().c_str(), mode), {});
        }

        void ZipFileDeleter::operator()(gzFile file)
        {
            gzclose(file);
            file = nullptr;
        }

#endif
}