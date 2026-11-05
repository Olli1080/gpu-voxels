// The .pcd reader of the CUDA helpers (gpu_voxels_helpers, PCL): the model files of the repository, the same files and counts as the Vulkan
// reader test (tests/vulkan/test_point_cloud_file.cpp, PcdModelFiles), and the failure of a missing file.
#include <gtest/gtest.h>

#include <filesystem>
#include <vector>

#include <gpu_voxels/helpers/PcdFileReader.h>

namespace fs = std::filesystem;

namespace
{
	const fs::path kModels = GVL_TEST_MODEL_DIR;
}

TEST(CudaPcdReader, ModelFiles)
{
	const fs::path binary = kModels / "hollie_from_pointcloud2.pcd", ascii = kModels / "robot4cmRes.pcd", intensity = kModels / "pointcloud_0002.pcd";
	if (!fs::exists(binary) || !fs::exists(ascii) || !fs::exists(intensity))
		GTEST_SKIP() << "model files missing";
	gpu_voxels::file_handling::PcdFileReader reader;

	std::vector<gpu_voxels::Vector3f> points;
	ASSERT_TRUE(reader.readPointCloud(binary, points));
	ASSERT_EQ(points.size(), 3151u);
	EXPECT_NEAR(points.front().x(), 5.1761179f, 1e-5);
	EXPECT_NEAR(points.front().y(), -2.8084764f, 1e-5);
	EXPECT_NEAR(points.front().z(), 0.1338043f, 1e-5);

	points.clear();
	ASSERT_TRUE(reader.readPointCloud(ascii, points));
	ASSERT_EQ(points.size(), 139995u);

	points.clear();
	ASSERT_TRUE(reader.readPointCloud(intensity, points));
	ASSERT_EQ(points.size(), 262897u);
	EXPECT_NEAR(points.front().x(), -0.45265159f, 1e-6);
	EXPECT_NEAR(points.front().y(), 0.59387434f, 1e-6);
	EXPECT_NEAR(points.front().z(), -0.088303901f, 1e-6);
}

TEST(CudaPcdReader, MissingFileFails)
{
	gpu_voxels::file_handling::PcdFileReader reader;
	std::vector<gpu_voxels::Vector3f> points;
	EXPECT_FALSE(reader.readPointCloud(kModels / "does_not_exist.pcd", points));
	EXPECT_TRUE(points.empty());
}
