add_executable(ts_icl_core 
  ts_main.cpp
)

find_package(Boost CONFIG COMPONENTS unit_test_framework REQUIRED)

target_link_libraries(ts_icl_core 
  PRIVATE 
    icl_core
    Boost::unit_test_framework
)

add_test(ts_icl_core ts_icl_core)

#IF(BUILD_SHARED_LIBS)
  #target_compile_definitions(test_unit PUBLIC BOOST_TEST_DYN_LINK)
#ENDIF(BUILD_SHARED_LIBS)