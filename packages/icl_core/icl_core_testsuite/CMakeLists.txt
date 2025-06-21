project(icl_core_testsuite
  LANGUAGES CXX
)
add_library(${PROJECT_NAME})

set(ICL_INCLUDE_DIR include/${PROJECT_NAME})

generate_export_header(${PROJECT_NAME} 
  EXPORT_MACRO_NAME
    ICL_CORE_TESTSUITE_IMPORT_EXPORT
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
    src/test_suite.cpp
    src/TestListener.cpp
  PUBLIC FILE_SET HEADERS 
  BASE_DIRS ${ICL_INCLUDE_DIR} "${CMAKE_CURRENT_BINARY_DIR}/${ICL_INCLUDE_DIR}"
  FILES
    ${ICL_INCLUDE_DIR}/test_suite.h
    ${ICL_INCLUDE_DIR}/TestListener.h
    "${CMAKE_CURRENT_BINARY_DIR}/${ICL_INCLUDE_DIR}/ImportExport.h"
)

find_package(CppUnit REQUIRED)
target_include_directories(${PROJECT_NAME} PUBLIC ${CPPUNIT_INCLUDE_DIRS})
target_link_libraries(${PROJECT_NAME} 
  PUBLIC
    icl_core 
    icl_core_logging
  PRIVATE 
    ${CPPUNIT_LIBRARIES}
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