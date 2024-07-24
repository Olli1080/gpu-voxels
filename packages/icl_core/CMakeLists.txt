# this is for emacs file handling -*- mode: cmake; indent-tabs-mode: nil -*-

# Include icmaker if not already available.
#
# When building in an ic_workspace based on icmaker, icmaker is usually globally
# available.
#
# When using catkin (for example in ROS workspaces), icmaker is not globally
# available and thus has to be found and included in every package. Also catkin
# packages usually define a project per package.
#
#

# To be used by other modules
#set(ICL_CORE_INCLUDE_DIRS "${CMAKE_CURRENT_SOURCE_DIR}/src" "${CMAKE_CURRENT_BINARY_DIR}/src" CACHE INTERNAL "")
#set(ICL_CORE_IDL_DIRS "${CMAKE_CURRENT_SOURCE_DIR}/src" CACHE INTERNAL "")
#set(ICL_CORE_IDE_FOLDER "icl_core")

#ICMAKER_REGISTER_PACKAGE(icl_core)

add_subdirectory(icl_core)

#need to generate this header here because of circular dependencies
add_library(icl_core_logging)
generate_export_header(icl_core_logging 
  EXPORT_MACRO_NAME
    ICL_CORE_LOGGING_IMPORT_EXPORT
  EXPORT_FILE_NAME
    "${CMAKE_BINARY_DIR}/include/icl_core_logging/ImportExport.h"
)

add_subdirectory(icl_core_config)
add_subdirectory(icl_core_logging)
add_subdirectory(icl_core_testsuite)
add_subdirectory(icl_core_performance_monitor)