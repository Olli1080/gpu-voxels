# Single registry of test categories (see cmake/Tests.cmake).
#
# PERFORMANCE: GoogleTest filter patterns ("Suite.Test", wildcards allowed) of benchmarks / long running
# tests. They are registered as CTest tests named "perf.<Suite>.<Test>" with label "performance" and are
# skipped by the default run (test presets exclude the label; plain ctest: use -LE performance).
# Add a pattern here instead of marking tests individually.
set(GVL_TEST_PERFORMANCE_FILTERS
  # e.g. "Benchmark*"
)
