project(icl_core_config
  LANGUAGES CXX
)
add_library(${PROJECT_NAME})

set(ICL_INCLUDE_DIR include/${PROJECT_NAME})

generate_export_header(${PROJECT_NAME} 
  EXPORT_MACRO_NAME
    ICL_CORE_CONFIG_IMPORT_EXPORT
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
    src/AttributeTree.cpp
    src/Config.cpp
    src/ConfigManager.cpp
    src/ConfigIterator.cpp
    src/ConfigParameter.cpp
    src/ConfigPositionalParameter.cpp
    src/GetoptParameter.cpp
    src/GetoptPositionalParameter.cpp
    src/GetoptParser.cpp
  PUBLIC FILE_SET HEADERS 
  BASE_DIRS ${ICL_INCLUDE_DIR} "${CMAKE_CURRENT_BINARY_DIR}/${ICL_INCLUDE_DIR}"
  FILES
    ${ICL_INCLUDE_DIR}/AttributeTree.h
    ${ICL_INCLUDE_DIR}/Config.h
    ${ICL_INCLUDE_DIR}/ConfigEnum.h
    ${ICL_INCLUDE_DIR}/ConfigEnumDefault.h
    ${ICL_INCLUDE_DIR}/ConfigIterator.h
    ${ICL_INCLUDE_DIR}/ConfigList.h
    ${ICL_INCLUDE_DIR}/ConfigManager.h
    ${ICL_INCLUDE_DIR}/ConfigObserver.h
    ${ICL_INCLUDE_DIR}/ConfigParameter.h
    ${ICL_INCLUDE_DIR}/ConfigPositionalParameter.h
    ${ICL_INCLUDE_DIR}/ConfigValue.h
    ${ICL_INCLUDE_DIR}/ConfigValueDefault.h
    ${ICL_INCLUDE_DIR}/ConfigValueIface.h
    ${ICL_INCLUDE_DIR}/ConfigValues.h
    ${ICL_INCLUDE_DIR}/GetoptParameter.h
    ${ICL_INCLUDE_DIR}/GetoptPositionalParameter.h
    ${ICL_INCLUDE_DIR}/GetoptParser.h
    ${ICL_INCLUDE_DIR}/icl_core_config.h
    ${ICL_INCLUDE_DIR}/MemberEnum.h
    ${ICL_INCLUDE_DIR}/MemberValue.h
    ${ICL_INCLUDE_DIR}/MemberValueIface.h
    ${ICL_INCLUDE_DIR}/Util.h
    "${CMAKE_CURRENT_BINARY_DIR}/${ICL_INCLUDE_DIR}/ImportExport.h"
)

find_package(tinyxml2 CONFIG REQUIRED)
find_package(Boost COMPONENTS Algorithm)

target_link_libraries(${PROJECT_NAME}
 PUBLIC 
  icl_core icl_core_logging
 PRIVATE
  tinyxml2::tinyxml2
  Boost::boost
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

#TODO:: add tests
#add_subdirectory(tests)
add_subdirectory(ts)