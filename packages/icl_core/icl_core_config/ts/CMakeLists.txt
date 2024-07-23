add_executable(ts_icl_core_config 
  ts_main.cpp
  ts_BatchGet.cpp
)

find_package(Boost CONFIG COMPONENTS unit_test_framework REQUIRED)

target_link_libraries(ts_icl_core_config 
  PRIVATE 
    icl_core
    icl_core_config
    icl_core_logging
    Boost::unit_test_framework
)

add_test(ts_icl_core_config ts_icl_core_config)

#IF(BUILD_SHARED_LIBS)
  #target_compile_definitions(test_unit PUBLIC BOOST_TEST_DYN_LINK)
#ENDIF(BUILD_SHARED_LIBS)
