// Tests of the vendored icl_core enum helpers (packages/icl_core/icl_core/src/EnumHelper.cpp).
#include <gtest/gtest.h>

#include <cstdint>
#include <string>
#include <vector>

#include <icl_core/EnumHelper.h>

TEST(EnumHelper, VectorOverloadFindsIndex32)
{
	const std::vector<std::string> names{"free", "occupied", "collision"};
	int32_t value = -1;
	EXPECT_TRUE(icl_core::string2Enum("occupied", value, names));
	EXPECT_EQ(value, 1);
}

TEST(EnumHelper, VectorOverloadFindsIndex64)
{
	const std::vector<std::string> names{"free", "occupied", "collision"};
	int64_t value = -1;
	EXPECT_TRUE(icl_core::string2Enum("collision", value, names));
	EXPECT_EQ(value, 2);
}

TEST(EnumHelper, VectorOverloadLeavesValueOnMiss)
{
	const std::vector<std::string> names{"free", "occupied"};
	int32_t value = 42;
	EXPECT_FALSE(icl_core::string2Enum("unknown", value, names));
	EXPECT_EQ(value, 42);
}

TEST(EnumHelper, CStringArrayWithEndMarker)
{
	const char* const names[] = {"a", "b", "c", "END"};
	int32_t value = -1;
	EXPECT_TRUE(icl_core::string2Enum("c", value, names, "END"));
	EXPECT_EQ(value, 2);
	EXPECT_FALSE(icl_core::string2Enum("END", value, names, "END"));
}

TEST(EnumHelper, CStringArrayWithNullTerminator)
{
	const char* const names[] = {"x", "y", nullptr};
	int32_t value = -1;
	EXPECT_TRUE(icl_core::string2Enum("y", value, names));
	EXPECT_EQ(value, 1);
	EXPECT_FALSE(icl_core::string2Enum("z", value, names));
}
