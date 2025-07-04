project(icl_core
  LANGUAGES CXX
)
add_library(${PROJECT_NAME})

set(ICL_INCLUDE_DIR include/${PROJECT_NAME})

generate_export_header(${PROJECT_NAME} 
  EXPORT_MACRO_NAME
    ICL_CORE_IMPORT_EXPORT
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
    src/EnumHelper.cpp
    src/fs.cpp
  PUBLIC FILE_SET HEADERS 
  BASE_DIRS ${ICL_INCLUDE_DIR} "${CMAKE_CURRENT_BINARY_DIR}/${ICL_INCLUDE_DIR}"
  FILES
    ${ICL_INCLUDE_DIR}/Deprecate.h
    ${ICL_INCLUDE_DIR}/EnumHelper.h
    ${ICL_INCLUDE_DIR}/fs.h
    ${ICL_INCLUDE_DIR}/icl_core.h
    ${ICL_INCLUDE_DIR}/KeyValueDirectory.h
    ${ICL_INCLUDE_DIR}/KeyValueDirectory.hpp
    ${ICL_INCLUDE_DIR}/Noncopyable.hpp
    ${ICL_INCLUDE_DIR}/RemoveMemberPointer.h
    "${CMAKE_CURRENT_BINARY_DIR}/${ICL_INCLUDE_DIR}/ImportExport.h"
)

option(USE_ZLIB "Build with zlib support" ON)

if(USE_ZLIB)
  find_package(ZLIB REQUIRED)
  target_link_libraries(${PROJECT_NAME} PUBLIC ZLIB::ZLIB)
  target_compile_definitions(${PROJECT_NAME} PUBLIC USE_ZLIB)
endif()

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

add_subdirectory(tests)