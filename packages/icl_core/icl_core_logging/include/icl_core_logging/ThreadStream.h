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
 * \author  Kay-Ulrich Scholl <scholl@fzi.de>
 * \author  Klaus Uhl <uhl@fzi.de>
 * \date    2006-10-08
 *
 * \brief   Contains icl_logging::ThreadStream
 *
 * \b icl_logging::ThreadStream
 *
 */
 //----------------------------------------------------------------------
#ifndef ICL_CORE_LOGGING_THREAD_STREAM_H_INCLUDED
#define ICL_CORE_LOGGING_THREAD_STREAM_H_INCLUDED

#include <list>
//#include <set>
#include <thread>

#include "icl_core_logging/Constants.h"
#include "icl_core_logging/ImportExport.h"
#include "icl_core_logging/LogLevel.h"

#ifdef _IC_BUILDER_DEPRECATED_STYLE_
# include "icl_core/Deprecate.h"
#endif

#if defined ICL_CORE_QT_SUPPORT
class QString;
#endif

#ifdef _IC_BUILDER_EIGEN_
#include <Eigen/Core>
#endif

namespace icl_core {
	namespace logging {

		class LogStream;

		//! Implements the actual logging for an individual thread.
		class ICL_CORE_LOGGING_IMPORT_EXPORT ThreadStream
		{
			friend class LogStream;
		public:
			/*! Change the thread stream's log level to \a level.
			 */
			void changeLevel(icl_core::logging::LogLevel level) { m_level = level; }

			/*! Return the current log level of this thread stream.
			 */
			[[nodiscard]] icl_core::logging::LogLevel getLogLevel() const { return m_level; }

			/*! Set the \a classname of the current log entry.
			 *
			 *  This value is automatically reset after a flush().
			 */
			void setClassname(const char* classname);

			/*! Set the \a objectname of the current log entry.
			 *
			 *  This value is automatically reset after a flush().
			 */
			void setObjectname(const char* objectname);

			/*! Set the \a filename of the current log entry.
			 *
			 *  This value is automatically reset after a flush().
			 */
			void setFilename(const char* filename);

			/*! Set the \a function of the current log entry.
			 *
			 *  This value is automatically reset after a flush().
			 */
			void setFunction(const char* function);

			/*! Set the \a line of the current log entry.
			 *
			 *  This value is automatically reset after a flush().
			 */
			void setLine(size_t line);

			/*! Set the log level of the current log line.
			 */
			void setLineLogLevel(icl_core::logging::LogLevel line_log_level);

			/*! Writes \a number_of_bytes characters from \a buffer to the
			 *  thread stream.  \a protected_buffer_size is the number of bytes
			 *  to keep free at the end of the buffer.  Normal Write operations
			 *  should leave this at 1 so that there is always room for a
			 *  Newline character.  Only the endl() function sets this to 0 to
			 *  be able to add the Newline character and thus make sure that
			 *  even when the buffer is filled completely, a line break is still
			 *  inserted correctly.
			 */
			void write(const char* buffer, size_t number_of_bytes, size_t protected_buffer_size = 1);

			/*! Writes text to the thread stream. Uses a printf-style format
			 *  string and variable number of additional arguments.
			 *
			 *  Implicitly flushes the thread stream.
			 */
			void printf(char const* const fmt, ...);

			/*! Flushes the internal buffer of the thread stream to the parent's
			 *  registered log output streams.
			 */
			void flush();

			/*! Type definition for flush() and endl() functions.
			 */
			typedef ThreadStream& (*ThreadStreamFunc)(ThreadStream&);

			/*! This operator allows us to stream the flush() and endl()
			 *  functions into a ThreadStream just like any other datatype.
			 */
			ThreadStream& operator << (ThreadStreamFunc func)
			{
				return (*func)(*this);
			}

		private:
			/*! Thread stream objects are always created by LogStream so the
			 *  constructor is private.
			 */
			ThreadStream(LogStream* parent);
			~ThreadStream() = default;

			LogStream* m_parent;
			icl_core::logging::LogLevel m_level;

			icl_core::logging::LogLevel m_line_log_level;
			char m_filename[cDEFAULT_LOG_SIZE + 1];
			size_t m_line;
			char m_classname[cDEFAULT_LOG_SIZE + 1];
			char m_objectname[cDEFAULT_LOG_SIZE + 1];
			char m_function[cDEFAULT_LOG_SIZE + 1];
			char m_data[cDEFAULT_LOG_SIZE + 1];
			size_t m_write_index;
			//char *m_write_pointer;
		};

		/*! This function (or better a pointer to this function) can be
		 *  streamed into a ThreadStream in order to force the stream to write
		 *  out its current buffer.
		 */
		inline ThreadStream& flush(ThreadStream& stream)
		{
			stream.flush();
			return stream;
		}

		/*! Does the same as flush() but writes a newline first.
		 */
		inline ThreadStream& endl(ThreadStream& stream)
		{
			stream.write("\n", 1, 0);
			return stream << flush;
		}

		ICL_CORE_LOGGING_IMPORT_EXPORT ThreadStream& operator << (ThreadStream& stream, uint8_t value);
		ICL_CORE_LOGGING_IMPORT_EXPORT ThreadStream& operator << (ThreadStream& stream, uint16_t value);
		ICL_CORE_LOGGING_IMPORT_EXPORT ThreadStream& operator << (ThreadStream& stream, uint32_t value);
#if __WORDSIZE != 64
		ICL_CORE_LOGGING_IMPORT_EXPORT ThreadStream& operator << (ThreadStream& stream, unsigned long value);
#endif
		ICL_CORE_LOGGING_IMPORT_EXPORT ThreadStream& operator << (ThreadStream& stream, uint64_t value);
		ICL_CORE_LOGGING_IMPORT_EXPORT ThreadStream& operator << (ThreadStream& stream, int8_t value);
		ICL_CORE_LOGGING_IMPORT_EXPORT ThreadStream& operator << (ThreadStream& stream, int16_t value);
		ICL_CORE_LOGGING_IMPORT_EXPORT ThreadStream& operator << (ThreadStream& stream, int32_t value);
#if __WORDSIZE != 64
		ICL_CORE_LOGGING_IMPORT_EXPORT ThreadStream& operator << (ThreadStream& stream, long value);
#endif
		ICL_CORE_LOGGING_IMPORT_EXPORT ThreadStream& operator << (ThreadStream& stream, int64_t value);

		ICL_CORE_LOGGING_IMPORT_EXPORT ThreadStream& operator << (ThreadStream& stream, const char* text);
		ICL_CORE_LOGGING_IMPORT_EXPORT ThreadStream& operator << (ThreadStream& stream, const std::string& text);
		ICL_CORE_LOGGING_IMPORT_EXPORT ThreadStream& operator << (ThreadStream& stream, double value);
		ICL_CORE_LOGGING_IMPORT_EXPORT ThreadStream& operator << (ThreadStream& stream, bool value);
		ICL_CORE_LOGGING_IMPORT_EXPORT ThreadStream& operator << (ThreadStream& stream, void* value);

		ICL_CORE_LOGGING_IMPORT_EXPORT ThreadStream& operator << (ThreadStream& stream, const std::thread::id& value);

#if defined ICL_CORE_QT_SUPPORT
		ICL_CORE_LOGGING_IMPORT_EXPORT ThreadStream& operator << (ThreadStream& stream, const QString& value);
#endif

#ifdef _IC_BUILDER_EIGEN_
		template <typename TScalar, int _Rows, int _Cols, int _Options, int _MaxRows, int _MaxCols>
		icl_core::logging::ThreadStream&
			operator << (icl_core::logging::ThreadStream& stream,
				const Eigen::Matrix<TScalar, _Rows, _Cols, _Options, _MaxRows, _MaxCols>& matrix)
		{
			stream << "[";
			for (int col = 0; col < matrix.cols(); ++col)
			{
				for (int row = 0; row < matrix.rows(); ++row)
				{
					stream << " " << matrix(row, col);
				}
				if (col < matrix.cols() - 1)
				{
					stream << " ;";
				}
			}
			stream << " ]";
			return stream;
		}
#endif

	}
}

#endif