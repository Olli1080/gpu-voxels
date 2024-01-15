// this is for emacs file handling -*- mode: c++; indent-tabs-mode: nil -*-
//----------------------------------------------------------------------
/*!\file
 *
 * \author  Klaus Uhl <uhl@fzi.de>
 * \date    2007-10-03
 *
 */
//----------------------------------------------------------------------
#include "icl_core_logging/FileLogOutput.h"

#include <filesystem>
#include <iostream>
#include <cstdio>

#ifdef _IC_BUILDER_ZLIB_
#include <zlib.h>
#endif

#include "icl_core/fs.h"
#include "icl_core_config/Config.h"
#include "icl_core_logging/Logging.h"

namespace icl_core {
    namespace logging {

        REGISTER_LOG_OUTPUT_STREAM(File, &FileLogOutput::create)

        LogOutputStream* FileLogOutput::create(const std::string& name, const std::string& config_prefix,
            icl_core::logging::LogLevel log_level)
        {
            return new FileLogOutput(name, config_prefix, log_level);
        }

        FileLogOutput::FileLogOutput(const std::string& name, const std::string& config_prefix,
            icl_core::logging::LogLevel log_level)
            : LogOutputStream(name, config_prefix, log_level),
            m_rotate(false),
            m_last_rotation(0),
            m_delete_old_files(false),
            m_delete_older_than_days(0)
        {
            icl_core::config::get<bool>(config_prefix + "/Rotate", m_rotate);

            if (m_rotate)
                m_last_rotation = icl_core::TimeStamp::now().tsNDays();

            uint32_t temp;
            if (icl_core::config::get<uint32_t>(config_prefix + "/DeleteOlderThan", temp))
            {
                m_delete_older_than_days = std::chrono::days{ temp };
                m_delete_old_files = true;
            }

#if defined(_IC_BUILDER_ZLIB_)
            m_online_zip = icl_core::config::getDefault<bool>(config_prefix + "/Zip", false);
#endif

            m_flush = icl_core::config::getDefault<bool>(config_prefix + "/Flush", true);

            if (icl_core::config::get<std::string>(config_prefix + "/FileName", m_filename))
            {
                expandFilename();

                // Determine the last write time of the log file, if it already
                // exists.
                const std::filesystem::path log_file_path(m_filename);
                if (std::filesystem::exists(log_file_path))
                {
                    if (std::filesystem::is_directory(log_file_path))
                    {
                        std::cerr << "The filename specified for log output stream "
                            << config_prefix << " is a directory." << std::endl;
                    }
                    else
                    {
                        m_last_rotation = icl_core::TimeStamp(std::chrono::clock_cast<std::chrono::system_clock>(std::filesystem::last_write_time(log_file_path))).tsNDays();
                        rotateLogFile();
                    }
                }

                openLogFile();
            }
            else
            {
                std::cerr << "No filename specified for file log output stream " << config_prefix << std::endl;
            }
        }

        FileLogOutput::FileLogOutput(const std::string& name, std::string filename,
            icl_core::logging::LogLevel log_level, bool flush)
            : LogOutputStream(name, log_level),
            m_filename(std::move(filename)),
            m_rotate(false),
            m_last_rotation(0),
            m_delete_old_files(false),
            m_delete_older_than_days(0),
            m_flush(flush)
#if defined(_IC_BUILDER_ZLIB_)
            ,m_online_zip(false)
#endif
        {
            expandFilename();
            openLogFile();
        }

        FileLogOutput::~FileLogOutput()
        {
            closeLogFile();
        }

        void FileLogOutput::pushImpl(const std::string& log_line)
        {
            rotateLogFile();

            if (!isOpen())
                openLogFile();

            if (isOpen())
            {
#ifdef _IC_BUILDER_ZLIB_
                if (m_online_zip)
                {
                    gzwrite(m_zipped_log_file.get(), log_line.c_str(), static_cast<unsigned int>(log_line.length()));
                }
                else
#endif
                {
                    m_log_file << log_line;
                }

                if (m_flush)
                    flush();
            }
        }

        bool FileLogOutput::isOpen() const
        {
#ifdef _IC_BUILDER_ZLIB_
            if (m_online_zip)
            {
                return m_zipped_log_file != nullptr;
            }
            else
#endif
            {
                return m_log_file.is_open();
            }
        }

        void FileLogOutput::flush()
        {
#ifdef _IC_BUILDER_ZLIB_
            if (m_online_zip)
            {
                gzflush(m_zipped_log_file.get(), Z_SYNC_FLUSH);
            }
            else
#endif
            {
                m_log_file.flush();
            }
        }

        void FileLogOutput::closeLogFile()
        {
#ifdef _IC_BUILDER_ZLIB_
            if (m_online_zip)
            {
                m_zipped_log_file.reset();
            }
            else
#endif
            {
                if (m_log_file.is_open())
                {
                    m_log_file.close();
                }
            }
        }

        void FileLogOutput::openLogFile()
        {
#if defined(_IC_BUILDER_ZLIB_)
            if (m_online_zip)
            {
                m_zipped_log_file = os::openZipFile(m_filename.c_str(), "a+b");
                if (m_zipped_log_file == nullptr)
                {
                    std::cerr << "Could not open log file " << m_filename << std::endl;
                }
                else
                {
                    const char* buffer = "\n\n-------------FILE (RE-)OPENED------------------\n";
                    gzwrite(m_zipped_log_file.get(), buffer, static_cast<unsigned int>(strlen(buffer)));
                }
            }
            else
#endif
                if (!m_log_file.is_open())
                {
                    m_log_file.open(m_filename.c_str(), std::ios::out | std::ios::app);
                    if (m_log_file.is_open())
                    {
                        m_log_file << "\n\n-------------FILE (RE-)OPENED------------------\n";
                        m_log_file.flush();
                    }
                    else
                    {
                        std::cerr << "Could not open log file " << m_filename << std::endl;
                    }
                }
        }

        void FileLogOutput::rotateLogFile()
        {
            if (m_rotate)
            {
	            const auto current_day = icl_core::TimeStamp::now().tsNDays();
                if (m_last_rotation != current_day)
                {
                    // First, close the log file if it's open.
                    closeLogFile();

                    // Move the file. ZIP it, if libz is available.
                    char time_str[12];
                    icl_core::TimeStamp(std::chrono::system_clock::time_point{m_last_rotation}).strfTime(time_str, 12, ".%Y-%m-%d");
#ifdef _IC_BUILDER_ZLIB_
                    if (!m_online_zip)
                    {
                        icl_core::os::zipFile(m_filename.c_str(), time_str);
                        _unlink(m_filename.c_str());
                    }
                    else
#endif
                    {
                        rename(m_filename.c_str(), (m_filename + time_str).c_str());
                    }

                    // Delete old log files.
                    if (m_delete_old_files)
                    {
	                    const std::filesystem::path log_file_path = std::filesystem::path(m_filename).parent_path();
	                    const std::string log_file_name = std::filesystem::path(m_filename).filename().string();
                        if (std::filesystem::exists(log_file_path) && std::filesystem::is_directory(log_file_path))
                        {
	                        const icl_core::TimeStamp delete_older_than(std::chrono::system_clock::time_point{current_day - m_delete_older_than_days});
                            for (std::filesystem::directory_iterator it(log_file_path), end; it != end; ++it)
                            {
                                // If the found file starts with the name of the log file the check its last write time.
                                if (!is_directory(*it)
                                    && icl_core::TimeStamp(std::chrono::clock_cast<std::chrono::system_clock>(std::filesystem::last_write_time(*it))) < delete_older_than
                                    && it->path().filename().string().find(log_file_name) == 0
                                    )
                                {
                                    std::filesystem::remove(*it);
                                }
                            }
                        }
                    }

                    // Store the rotation time.
                    m_last_rotation = current_day;

                    // Re-open the log file.
                    openLogFile();
                }
            }
        }

        void FileLogOutput::expandFilename()
        {
            //https://stackoverflow.com/questions/1902681/expand-file-names-that-have-environment-variables-in-their-path

            static std::regex env(R"(\$\{([^}]+)\})");
            std::smatch match;
            while (std::regex_search(m_filename, match, env)) {
                const char* s = getenv(match[1].str().c_str());
                const std::string var(s == nullptr ? "" : s);
                m_filename.replace(match[0].first, match[0].second, var);
            }
        }

    }
}