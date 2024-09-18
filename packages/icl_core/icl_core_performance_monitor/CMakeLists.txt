project(icl_core_performance_monitor
  LANGUAGES CXX
)
add_library(${PROJECT_NAME})

set(ICL_INCLUDE_DIR include/${PROJECT_NAME})

generate_export_header(${PROJECT_NAME} 
  EXPORT_MACRO_NAME
    ICL_CORE_PERFORMANCE_MONITOR_IMPORT_EXPORT
  EXPORT_FILE_NAME
    "${CMAKE_CURRENT_BINARY_DIR}/${ICL_INCLUDE_DIR}/ImportExport.h"
)

target_include_directories(${PROJECT_NAME} PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR}/include>
    $<INSTALL_INTERFACE:include>
)

target_sources(${PROJECT_NAME}
  PRIVATE    
    src/PerformanceMonitor.cpp
    src/logging_performance_monitor.cpp
  PUBLIC FILE_SET HEADERS 
  BASE_DIRS ${ICL_INCLUDE_DIR} "${CMAKE_CURRENT_BINARY_DIR}/include"
  FILES
    ${ICL_INCLUDE_DIR}/PerformanceMonitor.h
    ${ICL_INCLUDE_DIR}/PerformanceMonitorMacros.h
    ${ICL_INCLUDE_DIR}/logging_performance_monitor.h
    "${CMAKE_CURRENT_BINARY_DIR}/${ICL_INCLUDE_DIR}/ImportExport.h"
)

target_link_libraries(${PROJECT_NAME} 
  PUBLIC
    icl_core 
    icl_core_logging
)

target_compile_features(${PROJECT_NAME} PUBLIC cxx_std_20)

option(BUILD_EXAMPLES "Building example applications" ON)

add_executable(perf_monitor_example src/example_usage.cpp)
target_link_libraries(perf_monitor_example ${PROJECT_NAME})

install(
  TARGETS 
    ${PROJECT_NAME} perf_monitor_example
  EXPORT
    icl_core-targets
  LIBRARY
    DESTINATION lib
  FILE_SET HEADERS
    DESTINATION ${ICL_INCLUDE_DIR}
)