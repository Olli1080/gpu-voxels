project(icl_core_logging
  LANGUAGES CXX
)

set(ICL_INCLUDE_DIR include/${PROJECT_NAME})

target_include_directories(${PROJECT_NAME} PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR}/include>
    $<INSTALL_INTERFACE:include>
)

target_sources(${PROJECT_NAME}
  PRIVATE    
    src/FileLogOutput.cpp
    src/Logging.cpp
    src/LoggingManager.cpp
    src/LogLevel.cpp
    src/LogOutputStream.cpp
    src/LogStream.cpp
    src/StdErrorLogOutput.cpp
    src/StdLogOutput.cpp
    src/ThreadStream.cpp
  PUBLIC FILE_SET HEADERS 
  BASE_DIRS ${ICL_INCLUDE_DIR} "${CMAKE_CURRENT_BINARY_DIR}/include"
  FILES
    ${ICL_INCLUDE_DIR}/icl_core_logging.h
    ${ICL_INCLUDE_DIR}/Constants.h
    ${ICL_INCLUDE_DIR}/FileLogOutput.h
    ${ICL_INCLUDE_DIR}/Logging.h
    ${ICL_INCLUDE_DIR}/LoggingMacros_LLOGGING.h
    ${ICL_INCLUDE_DIR}/LoggingMacros_LLOGGING_FMT.h
    ${ICL_INCLUDE_DIR}/LoggingMacros_LOGGING.h
    ${ICL_INCLUDE_DIR}/LoggingMacros_LOGGING_FMT.h
    ${ICL_INCLUDE_DIR}/LoggingMacros_MLOGGING.h
    ${ICL_INCLUDE_DIR}/LoggingMacros_MLOGGING_FMT.h
    ${ICL_INCLUDE_DIR}/LoggingMacros_SLOGGING.h
    ${ICL_INCLUDE_DIR}/LoggingMacros_SLOGGING_FMT.h
    ${ICL_INCLUDE_DIR}/LoggingManager.h
    ${ICL_INCLUDE_DIR}/LogLevel.h
    ${ICL_INCLUDE_DIR}/LogOutputStream.h
    ${ICL_INCLUDE_DIR}/LogStream.h
    ${ICL_INCLUDE_DIR}/StdErrorLogOutput.h
    ${ICL_INCLUDE_DIR}/StdLogOutput.h
    ${ICL_INCLUDE_DIR}/ThreadStream.h
    "${CMAKE_CURRENT_BINARY_DIR}/${ICL_INCLUDE_DIR}/ImportExport.h"
)

#target_include_directories(icl_core_config PUBLIC ${ZLIB_INCLUDE_DIRS})
target_link_libraries(${PROJECT_NAME}
 PUBLIC 
  icl_core icl_core_config
)

target_compile_features(${PROJECT_NAME} PUBLIC cxx_std_20)

install(
  TARGETS 
    ${PROJECT_NAME}
  EXPORT
    icl_core-targets
  LIBRARY
    DESTINATION lib
  FILE_SET HEADERS
    DESTINATION ${ICL_INCLUDE_DIR}
)

# this is for emacs file handling -*- mode: cmake; indent-tabs-mode: nil -*-
#ICMAKER_SET("icl_core_logging" IDE_FOLDER ${ICL_CORE_IDE_FOLDER})

#IF(SQLite3_FOUND)
#  ICMAKER_ADD_SOURCES(
#    SQLiteLogDb.cpp
#    SQLiteLogOutput.cpp
#    )
#  ICMAKER_ADD_HEADERS(
#    SQLiteLogDb.h
#    SQLiteLogOutput.h
#    )
#  ICMAKER_EXTERNAL_DEPENDENCIES(EXPORT
#    SQLite3
#    )
#ENDIF(SQLite3_FOUND)

# @TODO:
# - pthread dependency on posix
# - lxrt
# - #ifdef ICL_CORE_QT_SUPPORT => QtCore dependency

#ICMAKER_LOCAL_CPPDEFINES(-DICL_CORE_LOGGING_EXPORT_SYMBOLS)
#ICMAKER_GLOBAL_CPPDEFINES(-D_IC_BUILDER_ICL_CORE_LOGGING_)

#ICMAKER_INCLUDE_DIRECTORIES(${ICL_CORE_INCLUDE_DIRS})

#ICMAKER_EXTERNAL_DEPENDENCIES(EXPORT
#  Boost_REGEX
#)

#ICMAKER_EXTERNAL_DEPENDENCIES(
#  Boost_FILESYSTEM
#  Boost_SYSTEM
#)
#ICMAKER_EXTERNAL_DEPENDENCIES(OPTIONAL
#  Zlib
#  Eigen2
#  Eigen3
#  )

#ICMAKER_INTERNAL_DEPENDENCIES(
#  icl_core
#  icl_core_config
#)

#IF(ANDROID)
#  ICMAKER_SYSTEM_DEPENDENCIES(-llog)
#ENDIF()

#ICMAKER_BUILD_LIBRARY()
#ICMAKER_INSTALL_HEADERS(${icmaker_target})
