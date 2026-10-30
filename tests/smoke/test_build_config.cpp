// Verifies that the CMake backend options reach the compiler (cmake/Options.cmake, cmake/Backends.cmake).
#include <gtest/gtest.h>

#ifndef GVL_TEST_EXPECT_CUDA
#error "GVL_TEST_EXPECT_CUDA must be defined by tests/CMakeLists.txt"
#endif
#ifndef GVL_TEST_EXPECT_VULKAN
#error "GVL_TEST_EXPECT_VULKAN must be defined by tests/CMakeLists.txt"
#endif

TEST(BuildConfig, CudaBackendDefineMatchesOption)
{
#ifdef GVL_BACKEND_CUDA
	constexpr bool defined = true;
#else
	constexpr bool defined = false;
#endif
	EXPECT_EQ(defined, GVL_TEST_EXPECT_CUDA == 1);
}

TEST(BuildConfig, VulkanBackendDefineMatchesOption)
{
#ifdef GVL_BACKEND_VULKAN
	constexpr bool defined = true;
#else
	constexpr bool defined = false;
#endif
	EXPECT_EQ(defined, GVL_TEST_EXPECT_VULKAN == 1);
}
