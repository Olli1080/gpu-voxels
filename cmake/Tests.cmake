# Test helpers. Included from the root CMakeLists.txt when BUILD_TESTS is ON.
#
#   gvl_add_test(<name> SOURCES <src>... [LIBS <target>...] [DEFINITIONS <def>...])
#
# Creates a GoogleTest executable linked with GTest::gtest_main and registers its cases with CTest.
# Always use this wrapper instead of a raw add_executable() for tests.
# Categories are declared in tests/test_categories.cmake: performance tests are registered with the
# CTest name prefix "perf." and label "performance"; the default run skips them
# (ctest -LE performance, or the 'default' test preset).

include(GoogleTest)

function(gvl_add_test name)
  cmake_parse_arguments(ARG "" "" "SOURCES;LIBS;DEFINITIONS" ${ARGN})
  if(NOT ARG_SOURCES)
    message(FATAL_ERROR "gvl_add_test(${name}): SOURCES is required")
  endif()

  add_executable(${name} ${ARG_SOURCES})
  target_compile_features(${name} PRIVATE cxx_std_20)
  target_link_libraries(${name} PRIVATE GTest::gtest_main ${ARG_LIBS})
  if(ARG_DEFINITIONS)
    target_compile_definitions(${name} PRIVATE ${ARG_DEFINITIONS})
  endif()

  if(GVL_TEST_PERFORMANCE_FILTERS)
    list(JOIN GVL_TEST_PERFORMANCE_FILTERS ":" perf_patterns)
    # Default group: everything except the performance patterns.
    gtest_discover_tests(${name}
      TEST_FILTER "-${perf_patterns}"
      DISCOVERY_MODE PRE_TEST)
    gtest_discover_tests(${name}
      TEST_FILTER "${perf_patterns}"
      TEST_PREFIX "perf."
      PROPERTIES LABELS performance
      DISCOVERY_MODE PRE_TEST)
  else()
    gtest_discover_tests(${name} DISCOVERY_MODE PRE_TEST)
  endif()
endfunction()
