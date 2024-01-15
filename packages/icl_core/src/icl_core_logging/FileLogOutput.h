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
 * \date    2007-10-03
 *
 * \brief   Contains icl_logging::FileLogOutput
 *
 * \b icl_logging::FileLogOutput
 *
 */
//----------------------------------------------------------------------
#ifndef ICL_CORE_LOGGING_FILE_LOG_OUTPUT_H_INCLUDED
#define ICL_CORE_LOGGING_FILE_LOG_OUTPUT_H_INCLUDED

#include <fstream>

#include "icl_core/fs.h"
#include "icl_core/TimeStamp.h"
#include "icl_core_logging/ImportExport.h"
#include "icl_core_logging/LogOutputStream.h"

namespace icl_core {
    namespace logging {

        /*! An output stream which streams to a file.
         *
         *  This class is implemented as a singleton so that only one instance
         *  can exist in any process.
         */
        class ICL_CORE_LOGGING_IMPORT_EXPORT FileLogOutput : public LogOutputStream,
            protected virtual icl_core::Noncopyable
        {
            friend class LoggingManager;

        public:
            /*! Creates a new file log output stream object.
             */
            static LogOutputStream* create(const std::string& name, const std::string& config_prefix,
                icl_core::logging::LogLevel log_level = cDEFAULT_LOG_LEVEL);

        private:

            FileLogOutput(const std::string& name, const std::string& config_prefix,
                icl_core::logging::LogLevel log_level);
            FileLogOutput(const std::string& name, std::string config_prefix,
                icl_core::logging::LogLevel log_level, bool flush);

            ~FileLogOutput() override;

            void pushImpl(const std::string& log_line) override;

            void expandFilename();

            [[nodiscard]] bool isOpen() const;
            void flush();

            void closeLogFile();
            void openLogFile();
            void rotateLogFile();

            std::string m_filename;
            std::ofstream m_log_file;

            bool m_rotate;
            std::chrono::days m_last_rotation;

            bool m_delete_old_files;
            std::chrono::days m_delete_older_than_days;

            bool m_flush;

#if defined(_IC_BUILDER_ZLIB_)
            bool m_online_zip;
            os::ZipFilePtr m_zipped_log_file;
#endif
        };

    }
}

#endif