// Slice S2a: Vulkan MetaPointCloud (GPU transformation), meta cloud insertion and self collision detection,
// verified against independent CPU references (tests/reference).
#include "vulkan_fixture.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <iostream>
#include <numbers>
#include <random>
#include <utility>
#include <vector>

#include <gpu_voxels/vulkan/BitVoxelMap.h>
#include <gpu_voxels/vulkan/MetaPointCloud.h>

#include "../reference/bit_voxel_map_reference.h"
#include "../reference/kinematics_reference.h"

using namespace gpu_voxels::vulkan;
using gvl_reference::BitVoxelMapReference;
using gvl_test::VulkanTest;

namespace
{
	using Clouds = std::vector<std::vector<Point3f>>;

	Clouds random_clouds(std::mt19937& rng, const std::vector<uint32_t>& sizes, float range)
	{
		std::uniform_real_distribution<float> coord(-range, range);
		Clouds clouds;
		for (uint32_t n : sizes)
		{
			std::vector<Point3f> cloud(n);
			for (auto& p : cloud)
				p = { coord(rng), coord(rng), coord(rng) };
			clouds.push_back(std::move(cloud));
		}
		return clouds;
	}

	gvl_reference::Mat4 random_rigid(std::mt19937& rng)
	{
		std::uniform_real_distribution<double> angle(-std::numbers::pi, std::numbers::pi), shift(-3.0, 3.0);
		return gvl_reference::translation(shift(rng), shift(rng), shift(rng)) * gvl_reference::rotation_z(angle(rng)) *
			gvl_reference::rotation_y(angle(rng)) * gvl_reference::rotation_x(angle(rng));
	}

	Matrix4f to_float(const gvl_reference::Mat4& m)
	{
		Matrix4f f;
		for (int r = 0; r < 4; ++r)
			for (int c = 0; c < 4; ++c)
				f(r, c) = static_cast<float>(m.at(r, c));
		return f;
	}

	gvl_reference::Mat4 to_reference(const Matrix4f& f)
	{
		gvl_reference::Mat4 m;
		for (int r = 0; r < 4; ++r)
			for (int c = 0; c < 4; ++c)
				m.at(r, c) = f(r, c);
		return m;
	}

	::testing::AssertionResult near_points(const std::vector<Point3f>& got, const Clouds& source, const std::vector<gvl_reference::Mat4>& transforms,
		double tolerance, const std::vector<bool>& transformed_mask = {})
	{
		std::size_t index = 0;
		for (std::size_t k = 0; k < source.size(); ++k)
			for (const auto& p : source[k])
			{
				std::array<double, 3> expected{ 0, 0, 0 };
				if (transformed_mask.empty() || transformed_mask[k])
					expected = transforms[k].apply(p);
				const Point3f& g = got.at(index);
				if (std::abs(g.x - expected[0]) > tolerance || std::abs(g.y - expected[1]) > tolerance ||
					std::abs(g.z - expected[2]) > tolerance)
					return ::testing::AssertionFailure() << "cloud " << k << " point " << index << ": got (" << g.x << ", " << g.y << ", "
						<< g.z << ") expected (" << expected[0] << ", " << expected[1] << ", " << expected[2] << ")";
				++index;
			}
		return ::testing::AssertionSuccess();
	}

	// ---- map helpers (same idea as in test_bit_voxel_map.cpp) ----
	constexpr Dim3ui kDims{ 24, 20, 22 };
	constexpr float kSide = 0.1f;

	Point3f random_inside_point(std::mt19937& rng, Dim3ui d, float side, uint32_t max_cell = 1u << 30)
	{
		std::uniform_int_distribution<uint32_t> vx(0, std::min(d.x, max_cell) - 1), vy(0, std::min(d.y, max_cell) - 1),
			vz(0, std::min(d.z, max_cell) - 1);
		std::uniform_real_distribution<float> frac(0.1f, 0.9f);
		return { (static_cast<float>(vx(rng)) + frac(rng)) * side, (static_cast<float>(vy(rng)) + frac(rng)) * side,
			(static_cast<float>(vz(rng)) + frac(rng)) * side };
	}

	Clouds random_map_clouds(std::mt19937& rng, const std::vector<uint32_t>& sizes, uint32_t max_cell = 1u << 30)
	{
		Clouds clouds;
		for (uint32_t n : sizes)
		{
			std::vector<Point3f> cloud(n);
			for (auto& p : cloud)
				p = random_inside_point(rng, kDims, kSide, max_cell);
			clouds.push_back(std::move(cloud));
		}
		return clouds;
	}

	::testing::AssertionResult map_equals(const BitVoxelMap& map, const BitVoxelMapReference& ref)
	{
		const auto gpu = map.download_words();
		const auto cpu = ref.to_words(0, map.voxel_count());
		for (std::size_t i = 0; i < gpu.size(); ++i)
			if (gpu[i] != cpu[i])
				return ::testing::AssertionFailure() << "voxel " << i / 8 << " word " << i % 8 << ": vulkan 0x" << std::hex << gpu[i]
					<< " reference 0x" << cpu[i];
		return ::testing::AssertionSuccess();
	}

	std::vector<uint32_t> words_of(const std::bitset<256>& bits)
	{
		std::vector<uint32_t> w(8, 0);
		for (uint32_t b = 0; b < 256; ++b)
			if (bits.test(b))
				w[b / 32] |= 1u << (b % 32);
		return w;
	}
}

// ------------------------------------------------------------------ MetaPointCloud

TEST_F(VulkanTest, MetaCloudGeometryAndRoundTrip)
{
	std::mt19937 rng(1);
	const Clouds clouds = random_clouds(rng, { 3, 0, 5, 0, 1 }, 2.0f);
	MetaPointCloud cloud(ctx(), clouds);
	EXPECT_EQ(cloud.num_clouds(), 5u);
	EXPECT_EQ(cloud.accumulated_size(), 9u);
	EXPECT_EQ(cloud.offsets(), (std::vector<uint32_t>{ 0, 3, 3, 8, 8, 9 }));
	EXPECT_EQ(cloud.cloud_size(2), 5u);
	EXPECT_EQ(cloud.cloud_offset(2), 3u);
	for (uint32_t k = 0; k < clouds.size(); ++k)
	{
		const auto got = cloud.download_cloud(k);
		ASSERT_EQ(got.size(), clouds[k].size());
		for (std::size_t i = 0; i < got.size(); ++i)
			EXPECT_TRUE(got[i].x == clouds[k][i].x && got[i].y == clouds[k][i].y && got[i].z == clouds[k][i].z);
	}

	const auto replacement = random_clouds(rng, { 5 }, 1.0f)[0];
	cloud.update_cloud(2, replacement);
	const auto updated = cloud.download_cloud(2);
	EXPECT_EQ(updated[4].z, replacement[4].z);
	EXPECT_EQ(cloud.download_cloud(0)[1].y, clouds[0][1].y) << "other sub clouds stay untouched";

	EXPECT_THROW(cloud.update_cloud(2, std::vector<Point3f>(4)), VulkanError); // size must not change
	EXPECT_THROW(cloud.update_cloud(9, std::vector<Point3f>(1)), VulkanError);
	EXPECT_THROW((void)cloud.download_cloud(5), VulkanError);
}

TEST_F(VulkanTest, MetaCloudWithoutPointsIsValid)
{
	MetaPointCloud none(ctx(), std::vector<std::vector<Point3f>>{});
	EXPECT_TRUE(none.empty());
	EXPECT_EQ(none.num_clouds(), 0u);
	MetaPointCloud target = none.make_like();
	none.transform(Matrix4f::identity(), target); // no-op
	EXPECT_TRUE(none.download().empty());

	MetaPointCloud empty_clouds(ctx(), std::vector<std::vector<Point3f>>(3));
	EXPECT_TRUE(empty_clouds.empty());
	EXPECT_EQ(empty_clouds.num_clouds(), 3u);
}

TEST_F(VulkanTest, MetaCloudTransformMatchesReference)
{
	std::mt19937 rng(2);
	const Clouds clouds = random_clouds(rng, { 1000, 0, 7, 12345, 1 }, 2.0f);
	MetaPointCloud source(ctx(), clouds);
	MetaPointCloud target = source.make_like();

	const Matrix4f m = to_float(random_rigid(rng));
	source.transform(m, target);
	const std::vector<gvl_reference::Mat4> transforms(clouds.size(), to_reference(m));
	EXPECT_TRUE(near_points(target.download(), clouds, transforms, 2e-5));
	EXPECT_TRUE(near_points(source.download(), clouds, std::vector<gvl_reference::Mat4>(clouds.size()), 0.0)) << "source stays unchanged";
}

TEST_F(VulkanTest, MetaCloudSubCloudTransformsTouchOnlySelectedClouds)
{
	std::mt19937 rng(3);
	const Clouds clouds = random_clouds(rng, { 40, 0, 30, 50, 20, 10 }, 1.0f);
	MetaPointCloud source(ctx(), clouds);

	MetaPointCloud single = source.make_like();
	const Matrix4f m3 = to_float(random_rigid(rng));
	source.transform_sub_cloud(3, m3, single);
	std::vector<gvl_reference::Mat4> transforms(clouds.size());
	transforms[3] = to_reference(m3);
	std::vector<bool> mask(clouds.size(), false);
	mask[3] = true;
	EXPECT_TRUE(near_points(single.download(), clouds, transforms, 2e-5, mask)) << "only sub cloud 3 transformed, others zero";

	MetaPointCloud several = source.make_like();
	const Matrix4f m0 = to_float(random_rigid(rng)), m5 = to_float(random_rigid(rng)), m4 = to_float(random_rigid(rng));
	const std::vector<SubCloudTransform> list{ { 5, m5 }, { 0, m0 }, { 4, m4 } };
	source.transform_sub_clouds(list, several);
	transforms.assign(clouds.size(), gvl_reference::Mat4());
	transforms[0] = to_reference(m0); transforms[4] = to_reference(m4); transforms[5] = to_reference(m5);
	mask.assign(clouds.size(), false);
	mask[0] = mask[4] = mask[5] = true;
	EXPECT_TRUE(near_points(several.download(), clouds, transforms, 2e-5, mask));
}

// 700 sub clouds incl. many empty ones: exercises the sub cloud lookup of the shader.
TEST_F(VulkanTest, MetaCloudManySubClouds)
{
	std::mt19937 rng(4);
	std::uniform_int_distribution<uint32_t> size(0, 40);
	std::vector<uint32_t> sizes(700);
	for (auto& s : sizes)
		s = size(rng) < 8 ? 0 : size(rng);
	const Clouds clouds = random_clouds(rng, sizes, 1.0f);
	MetaPointCloud source(ctx(), clouds);
	MetaPointCloud target = source.make_like();

	std::vector<SubCloudTransform> list;
	std::vector<gvl_reference::Mat4> transforms(clouds.size());
	std::vector<bool> mask(clouds.size(), false);
	for (uint16_t k = 0; k < clouds.size(); k += 3)
	{
		const auto m = random_rigid(rng);
		const Matrix4f f = to_float(m);
		list.push_back({ k, f });
		transforms[k] = to_reference(f);
		mask[k] = true;
	}
	source.transform_sub_clouds(list, target);
	EXPECT_TRUE(near_points(target.download(), clouds, transforms, 2e-5, mask));
}

TEST_F(VulkanTest, MetaCloudTransformInPlace)
{
	std::mt19937 rng(5);
	const Clouds clouds = random_clouds(rng, { 100, 20, 30 }, 1.0f);
	MetaPointCloud cloud(ctx(), clouds);
	const Matrix4f a = to_float(random_rigid(rng)), b = to_float(random_rigid(rng));

	cloud.transform_self(a);
	std::vector<gvl_reference::Mat4> transforms(clouds.size(), to_reference(a));
	EXPECT_TRUE(near_points(cloud.download(), clouds, transforms, 2e-5));

	// Second step on sub cloud 1 only; reference applies b to the already transformed points.
	Clouds after_first;
	for (std::size_t k = 0; k < clouds.size(); ++k)
	{
		std::vector<Point3f> c;
		for (const auto& p : clouds[k])
		{
			const auto q = to_reference(a).apply(p);
			c.push_back({ static_cast<float>(q[0]), static_cast<float>(q[1]), static_cast<float>(q[2]) });
		}
		after_first.push_back(c);
	}
	cloud.transform_self_sub_cloud(1, b);
	transforms.assign(clouds.size(), gvl_reference::Mat4());
	transforms[1] = to_reference(b);
	const auto got = cloud.download();
	// Cloud 0 and 2 keep the first transformation, cloud 1 got the second one on top.
	EXPECT_TRUE(near_points(std::vector<Point3f>(got.begin() + 100, got.begin() + 120),
		Clouds{ after_first[1] }, { transforms[1] }, 1e-4));
	EXPECT_TRUE(near_points(std::vector<Point3f>(got.begin(), got.begin() + 100), Clouds{ clouds[0] }, { to_reference(a) }, 2e-5));
}

TEST_F(VulkanTest, MetaCloudRejectsInvalidTransforms)
{
	std::mt19937 rng(6);
	MetaPointCloud a(ctx(), random_clouds(rng, { 5, 6 }, 1.0f));
	MetaPointCloud different(ctx(), random_clouds(rng, { 5, 7 }, 1.0f));
	MetaPointCloud same = a.make_like();
	EXPECT_THROW(a.transform(Matrix4f::identity(), different), VulkanError);
	EXPECT_THROW(a.transform(Matrix4f::identity(), a), VulkanError);
	EXPECT_THROW(a.transform_sub_cloud(2, Matrix4f::identity(), same), VulkanError);
	EXPECT_THROW(a.transform_self_sub_cloud(2, Matrix4f::identity()), VulkanError);
}

// ------------------------------------------------------------------ insertion of meta clouds

TEST_F(VulkanTest, BitMapInsertMetaPointCloudMatchesReference)
{
	std::mt19937 rng(10);
	Clouds clouds = random_map_clouds(rng, { 3000, 0, 500, 7000, 1 });
	clouds[2].push_back({ -1.0f, 0.5f, 0.5f });        // outside
	clouds[3].push_back({ 1e9f, 0.5f, 0.5f });         // outside
	clouds[3].push_back({ std::nanf(""), 0.0f, 0.0f }); // outside
	MetaPointCloud cloud(ctx(), clouds);

	const std::vector<uint8_t> meanings{ 1, 9, 40, 200, 255 };
	BitVoxelMap map(ctx(), kDims, kSide);
	auto ref = BitVoxelMapReference(kDims.x, kDims.y, kDims.z, kSide);
	const uint32_t expected_outside = ref.insert_meta_clouds(clouds, meanings);
	EXPECT_EQ(expected_outside, 3u);
	EXPECT_EQ(map.insert_meta_point_cloud(cloud, meanings), expected_outside);
	EXPECT_TRUE(map_equals(map, ref));

	// Single meaning for everything.
	BitVoxelMap map2(ctx(), kDims, kSide);
	auto ref2 = BitVoxelMapReference(kDims.x, kDims.y, kDims.z, kSide);
	ref2.insert_meta_clouds(clouds, std::vector<uint8_t>(clouds.size(), 77));
	EXPECT_EQ(map2.insert_meta_point_cloud(cloud, uint8_t{ 77 }), expected_outside);
	EXPECT_TRUE(map_equals(map2, ref2));
}

TEST_F(VulkanTest, BitMapInsertMetaPointCloudRejectsInvalidArguments)
{
	std::mt19937 rng(11);
	MetaPointCloud cloud(ctx(), random_map_clouds(rng, { 10, 10 }));
	BitVoxelMap map(ctx(), kDims, kSide);
	const std::vector<uint8_t> one_meaning{ 1 };
	EXPECT_THROW((void)map.insert_meta_point_cloud(cloud, one_meaning), VulkanError);

	MetaPointCloud none(ctx(), std::vector<std::vector<Point3f>>{});
	EXPECT_EQ(map.insert_meta_point_cloud(none, uint8_t{ 1 }), 0u);
}

// ------------------------------------------------------------------ self collision

TEST_F(VulkanTest, BitMapSelfCollisionDefaultMasksMatchReference)
{
	std::mt19937 rng(20);
	// Twelve links in a small corner of the map: neighbouring links overlap a lot.
	const Clouds clouds = random_map_clouds(rng, std::vector<uint32_t>(12, 300), 8);
	MetaPointCloud cloud(ctx(), clouds);

	BitVoxelMap map(ctx(), kDims, kSide);
	auto ref = BitVoxelMapReference(kDims.x, kDims.y, kDims.z, kSide);
	for (int round = 0; round < 2; ++round) // the second round starts with a filled map
	{
		const auto expected = ref.insert_meta_clouds_self_collision(clouds);
		const auto result = map.insert_meta_point_cloud_with_self_collision_check(cloud);
		ASSERT_TRUE(expected.colliding) << "test data must contain self collisions";
		EXPECT_TRUE(result.colliding);
		EXPECT_EQ(std::vector<uint32_t>(result.colliding_meanings.begin(), result.colliding_meanings.end()), words_of(expected.meanings))
			<< "round " << round;
		EXPECT_EQ(result.points_outside, expected.outside);
		EXPECT_TRUE(map_equals(map, ref)) << "round " << round;
	}
}

TEST_F(VulkanTest, BitMapSelfCollisionWithoutOverlapReportsNothing)
{
	// Each sub cloud lives in its own slab of the map.
	std::mt19937 rng(21);
	Clouds clouds;
	std::uniform_real_distribution<float> frac(0.1f, 0.9f);
	for (uint32_t k = 0; k < 5; ++k)
	{
		std::vector<Point3f> c;
		for (int i = 0; i < 200; ++i)
			c.push_back({ (static_cast<float>(k * 4 + (i % 4)) + frac(rng)) * kSide, (static_cast<float>(i % 17) + frac(rng)) * kSide,
				(static_cast<float>(i % 13) + frac(rng)) * kSide });
		clouds.push_back(c);
	}
	MetaPointCloud cloud(ctx(), clouds);
	BitVoxelMap map(ctx(), kDims, kSide);
	auto ref = BitVoxelMapReference(kDims.x, kDims.y, kDims.z, kSide);
	const auto expected = ref.insert_meta_clouds_self_collision(clouds);
	ASSERT_FALSE(expected.colliding);
	const auto result = map.insert_meta_point_cloud_with_self_collision_check(cloud);
	EXPECT_FALSE(result.colliding);
	EXPECT_TRUE(map_equals(map, ref));
}

TEST_F(VulkanTest, BitMapSelfCollisionCustomMeaningsAndMasks)
{
	std::mt19937 rng(22);
	const Clouds clouds = random_map_clouds(rng, std::vector<uint32_t>(6, 400), 6);
	MetaPointCloud cloud(ctx(), clouds);
	const std::vector<uint8_t> meanings{ 10, 11, 12, 13, 14, 15 };

	// Neighbouring links (k-1, k+1) are not collision partners; own bit and the collision bit are excluded as well.
	std::vector<BitVector256> masks;
	std::vector<std::bitset<256>> ref_masks;
	for (uint32_t k = 0; k < 6; ++k)
	{
		std::bitset<256> mask;
		mask.set();
		mask.reset(2);
		for (int d = -1; d <= 1; ++d)
			if (static_cast<int>(k) + d >= 0 && static_cast<int>(k) + d < 6)
				mask.reset(meanings[k + d]);
		ref_masks.push_back(mask);
		BitVector256 words{};
		for (uint32_t b = 0; b < 256; ++b)
			if (mask.test(b))
				words[b / 32] |= 1u << (b % 32);
		masks.push_back(words);
	}

	BitVoxelMap map(ctx(), kDims, kSide);
	auto ref = BitVoxelMapReference(kDims.x, kDims.y, kDims.z, kSide);
	const auto expected = ref.insert_meta_clouds_self_collision(clouds, meanings, ref_masks);
	ASSERT_TRUE(expected.colliding);
	const auto result = map.insert_meta_point_cloud_with_self_collision_check(cloud, meanings, masks);
	EXPECT_TRUE(result.colliding);
	EXPECT_EQ(std::vector<uint32_t>(result.colliding_meanings.begin(), result.colliding_meanings.end()), words_of(expected.meanings));
	EXPECT_TRUE(map_equals(map, ref));
	// Neighbours may overlap without being reported: meanings 10..15 only appear if non-adjacent links met.
}

TEST_F(VulkanTest, BitMapSelfCollisionRejectsInvalidArguments)
{
	std::mt19937 rng(23);
	MetaPointCloud cloud(ctx(), random_map_clouds(rng, { 10, 10, 10 }));
	BitVoxelMap map(ctx(), kDims, kSide);
	const std::vector<uint8_t> two{ 5, 6 };
	EXPECT_THROW((void)map.insert_meta_point_cloud_with_self_collision_check(cloud, two), VulkanError);
	const std::vector<BitVector256> one_mask(1);
	EXPECT_THROW((void)map.insert_meta_point_cloud_with_self_collision_check(cloud, {}, one_mask), VulkanError);
	MetaPointCloud none(ctx(), std::vector<std::vector<Point3f>>{});
	EXPECT_FALSE(map.insert_meta_point_cloud_with_self_collision_check(none).colliding);
}

// Voxels that only carry the FREE bit are not collision partners (noneButEmpty ignores bit 0).
TEST_F(VulkanTest, BitMapSelfCollisionIgnoresFreeBitOnlyVoxels)
{
	std::mt19937 rng(24);
	const Clouds clouds = random_map_clouds(rng, std::vector<uint32_t>(3, 200), 5);
	MetaPointCloud cloud(ctx(), clouds);

	BitVoxelMap map(ctx(), kDims, kSide);
	auto ref = BitVoxelMapReference(kDims.x, kDims.y, kDims.z, kSide);
	// Mark everything the clouds will touch as FREE beforehand.
	for (const auto& c : clouds)
	{
		map.insert_point_cloud(c, eBVM_FREE);
		ref.insert_points(c, 0);
	}
	// Single sub cloud overlapping only FREE voxels and itself: no self collision.
	const Clouds single{ clouds[0] };
	MetaPointCloud one(ctx(), single);
	const auto expected = ref.insert_meta_clouds_self_collision(single);
	EXPECT_FALSE(expected.colliding);
	const auto result = map.insert_meta_point_cloud_with_self_collision_check(one);
	EXPECT_FALSE(result.colliding);
	EXPECT_TRUE(map_equals(map, ref));
}
