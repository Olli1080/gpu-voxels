// Slice S6e: the façade of the octree (point clouds, coordinate lists, meta point clouds with meanings) and OctreeStorage with the native format.
// The inserts are compared with Octree::insert_ids on a twin tree (checked against the reference in the S6d tests) fed with the ids this file computes
// itself on the host; storage is checked by round trips (dump, extract, find, collisions, further inserts), by decoding the bytes independently and by
// damaged / forged streams.
#include "vulkan_fixture.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <limits>
#include <random>
#include <sstream>
#include <string>
#include <vector>

#include <gpu_voxels/vulkan/MetaPointCloud.h>
#include <gpu_voxels/vulkan/Octree.h>
#include <gpu_voxels/vulkan/OctreeStorage.h>

#include "../reference/octree_reference.h"

using namespace gpu_voxels::vulkan;
using gvl_test::VulkanTest;
namespace ref = gvl_reference::octree;

namespace
{
	constexpr OctreeKind kKinds[] = { OctreeKind::Deterministic, OctreeKind::Probabilistic };
	const char* name(OctreeKind kind) { return kind == OctreeKind::Probabilistic ? "probabilistic" : "deterministic"; }

	std::vector<OctreeCube> sorted_cubes(const Octree& tree)
	{
		auto cubes = tree.extract();
		std::sort(cubes.begin(), cubes.end(), [](const OctreeCube& a, const OctreeCube& b) { return a.level != b.level ? a.level < b.level : a.id < b.id; });
		return cubes;
	}

	/// A voxel as the façade must see it: floor(p / side), valid below 2^14 on every axis (float arithmetic like the shader; the test points keep away from voxel borders).
	bool voxel_id(const Point3f& p, float side, uint64_t& id)
	{
		const float x = std::floor(p.x / side), y = std::floor(p.y / side), z = std::floor(p.z / side);
		if (!(x >= 0.0f && x < 16384.0f && y >= 0.0f && y < 16384.0f && z >= 0.0f && z < 16384.0f))
			return false;
		id = ref::morton(uint32_t(x), uint32_t(y), uint32_t(z));
		return true;
	}

	struct Cloud
	{
		std::vector<Point3f> points;
		std::vector<uint64_t> ids;  ///< ids of the points inside the tree
		uint32_t outside = 0;
	};

	/// `count` random points in the voxels 0 .. extent - 1 (inside a voxel, away from its border), about a third of them duplicates,
	/// plus `bad` points outside the tree (negative, too far, NaN, inf).
	Cloud random_cloud(std::mt19937_64& rng, size_t count, uint32_t extent, float side, size_t bad)
	{
		Cloud c;
		std::uniform_int_distribution<uint32_t> d(0, extent - 1);
		std::uniform_real_distribution<float> f(0.25f, 0.75f);
		for (size_t i = 0; i < count; ++i)
		{
			if (i % 3 == 2 && !c.points.empty())
				c.points.push_back(c.points[rng() % c.points.size()]);
			else
				c.points.push_back({ (d(rng) + f(rng)) * side, (d(rng) + f(rng)) * side, (d(rng) + f(rng)) * side });
		}
		const float nan = std::numeric_limits<float>::quiet_NaN(), inf = std::numeric_limits<float>::infinity();
		for (size_t i = 0; i < bad; ++i)
		{
			Point3f p{ 1.5f * side, 2.5f * side, 3.5f * side };
			switch (i % 8)
			{
			case 0: p.x = -0.5f * side; break;
			case 1: p.y = -3.5f * side; break;
			case 2: p.z = 16384.5f * side; break;
			case 3: p.x = nan; break;
			case 4: p.y = inf; break;
			case 5: p.z = -inf; break;
			case 6: p.x = 16384.5f * side; break;
			default: p.y = 16384.5f * side; break;
			}
			c.points.push_back(p);
		}
		if (bad > 0) // the last voxel of the grid: ids above 2^41 and the largest valid id
			c.points.push_back({ 16383.5f * side, 16383.5f * side, 16383.5f * side });
		std::shuffle(c.points.begin(), c.points.end(), rng);
		for (const Point3f& p : c.points)
		{
			uint64_t id = 0;
			if (voxel_id(p, side, id))
				c.ids.push_back(id);
			else
				++c.outside;
		}
		return c;
	}

	void build_base(Octree& a, Octree& b, std::mt19937_64& rng)
	{
		std::vector<OctreeVoxel> base;
		for (int i = 0; i < 300; ++i)
			base.push_back({ uint32_t(rng() % 32), uint32_t(rng() % 32), uint32_t(rng() % 32) });
		a.build(base);
		b.build(base);
	}
}

TEST_F(VulkanTest, OctreePointCloudInsertEqualsTheInsertOfItsVoxelIds)
{
	if (!ctx().info().buffer_device_address)
		GTEST_SKIP() << "bufferDeviceAddress not supported";
	for (const auto kind : kKinds)
	{
		for (const float side : { 0.05f, 1.0f, 0.37f })
		{
			for (const size_t count : { size_t(1), size_t(40), size_t(3000), size_t(60000) })
			{
				SCOPED_TRACE(std::string(name(kind)) + ", side " + std::to_string(side) + ", " + std::to_string(count) + " points");
				std::mt19937_64 rng(count * 7 + uint64_t(side * 100));
				Octree tree(ctx(), kind, side), twin(ctx(), kind);
				build_base(tree, twin, rng);
				const Cloud cloud = random_cloud(rng, count, count > 1000 ? 64 : 32, side, 12);
				const uint64_t dropped_before = tree.dropped_count();
				EXPECT_EQ(tree.insert_point_cloud(cloud.points), cloud.outside);
				twin.insert_ids(cloud.ids);
				EXPECT_EQ(tree.dropped_count(), dropped_before + cloud.outside);
				EXPECT_EQ(tree.dump(), twin.dump());
				EXPECT_NO_THROW(tree.check_tree());
				if (count == 3000)
				{
					// a second cloud over the same voxels: the probabilistic tree adds up, the deterministic one stays
					const Cloud again = random_cloud(rng, count, 32, side, 0);
					tree.insert_point_cloud(again.points);
					twin.insert_ids(again.ids);
					EXPECT_EQ(tree.dump(), twin.dump());
					EXPECT_EQ(sorted_cubes(tree), sorted_cubes(twin));
				}
			}
		}
	}
}

TEST_F(VulkanTest, OctreePointCloudMeaningsAndPropagation)
{
	if (!ctx().info().buffer_device_address)
		GTEST_SKIP() << "bufferDeviceAddress not supported";
	for (const auto kind : kKinds)
	{
		SCOPED_TRACE(name(kind));
		std::mt19937_64 rng(5);
		const float side = 0.1f;
		Octree tree(ctx(), kind, side), twin(ctx(), kind);
		build_base(tree, twin, rng);
		const Cloud cloud = random_cloud(rng, 2000, 32, side, 0);

		// eBVM_FREE inserts free voxels
		tree.insert_point_cloud(cloud.points, eBVM_FREE);
		twin.insert_ids(cloud.ids, OctreeDatum::Free);
		EXPECT_EQ(tree.dump(), twin.dump());

		// without propagate_now the tree is stale until propagate()
		const Cloud more = random_cloud(rng, 500, 32, side, 3);
		tree.insert_point_cloud(more.points, eBVM_OCCUPIED, false);
		twin.insert_ids(more.ids, OctreeDatum::Occupied, true);
		EXPECT_GT(tree.pending_update_count(), 0u);
		tree.propagate();
		EXPECT_EQ(tree.pending_update_count(), 0u);
		EXPECT_EQ(tree.dump(), twin.dump());

		// meanings the tree cannot store throw and change nothing
		const auto before = tree.dump();
		for (const uint8_t meaning : { uint8_t(eBVM_COLLISION), uint8_t(eBVM_UNKNOWN), uint8_t(eBVM_SWEPT_VOLUME_START), uint8_t(255) })
		{
			EXPECT_THROW(tree.insert_point_cloud(cloud.points, meaning), VulkanError) << int(meaning);
			EXPECT_THROW(tree.insert_coordinate_list({}, meaning), VulkanError) << int(meaning);
		}
		EXPECT_EQ(tree.dump(), before);

		// no points: nothing happens
		EXPECT_EQ(tree.insert_point_cloud({}), 0u);
		EXPECT_EQ(tree.dump(), before);
		// only points outside: all counted, nothing inserted
		const std::vector<Point3f> outside = { { -1.0f, 0.0f, 0.0f }, { 0.0f, std::numeric_limits<float>::quiet_NaN(), 0.0f } };
		EXPECT_EQ(tree.insert_point_cloud(outside), 2u);
		EXPECT_EQ(tree.dump(), before);
	}
}

TEST_F(VulkanTest, OctreeVoxelSideLength)
{
	if (!ctx().info().buffer_device_address)
		GTEST_SKIP() << "bufferDeviceAddress not supported";
	Octree tree(ctx(), OctreeKind::Deterministic);
	EXPECT_EQ(tree.voxel_side_length(), 0.0f);
	const std::vector<Point3f> points = { { 0.5f, 0.5f, 0.5f } };
	EXPECT_THROW(tree.insert_point_cloud(points), VulkanError);   // not set
	const auto empty = tree.dump();
	EXPECT_EQ(tree.dump(), empty);
	EXPECT_THROW(tree.set_voxel_side_length(-1.0f), VulkanError);
	EXPECT_THROW(tree.set_voxel_side_length(std::numeric_limits<float>::quiet_NaN()), VulkanError);
	EXPECT_THROW(tree.set_voxel_side_length(std::numeric_limits<float>::infinity()), VulkanError);
	EXPECT_THROW(Octree(ctx(), OctreeKind::Deterministic, -0.1f), VulkanError);
	EXPECT_EQ(tree.voxel_side_length(), 0.0f);
	tree.set_voxel_side_length(0.25f);
	EXPECT_EQ(tree.voxel_side_length(), 0.25f);
	EXPECT_EQ(tree.insert_point_cloud(points), 0u);
	const auto found = tree.find(std::vector<uint64_t>{ ref::morton(2, 2, 2) });
	ASSERT_EQ(found.size(), 1u);
	EXPECT_EQ(found[0].level, 0u);
	EXPECT_NE(found[0].status & 4, 0); // OCCUPIED
	// the side length changes the voxel, not the tree: 0.5 / 0.5 = voxel 1
	tree.set_voxel_side_length(0.5f);
	EXPECT_EQ(tree.insert_point_cloud(points), 0u);
	const auto other = tree.find(std::vector<uint64_t>{ ref::morton(1, 1, 1) });
	EXPECT_EQ(other[0].level, 0u);
	EXPECT_NE(other[0].status & 4, 0);
}

TEST_F(VulkanTest, OctreeCoordinateListInsert)
{
	if (!ctx().info().buffer_device_address)
		GTEST_SKIP() << "bufferDeviceAddress not supported";
	for (const auto kind : kKinds)
	{
		SCOPED_TRACE(name(kind));
		std::mt19937_64 rng(11);
		Octree tree(ctx(), kind), twin(ctx(), kind);
		build_base(tree, twin, rng);
		std::vector<Coord3ui> coords;
		std::vector<uint64_t> ids;
		uint32_t outside = 0;
		for (int i = 0; i < 1500; ++i)
		{
			Coord3ui c{ uint32_t(rng() % 40), uint32_t(rng() % 40), uint32_t(rng() % 40) };
			if (i % 50 == 7)
				c.y = 16384 + uint32_t(rng() % 100); // outside
			coords.push_back(c);
			if (c.x < 16384 && c.y < 16384 && c.z < 16384)
				ids.push_back(ref::morton(c.x, c.y, c.z));
			else
				++outside;
		}
		EXPECT_EQ(tree.insert_coordinate_list(coords), outside);
		twin.insert_ids(ids);
		EXPECT_EQ(tree.dump(), twin.dump());
		EXPECT_EQ(tree.dropped_count(), outside);
		EXPECT_EQ(tree.insert_coordinate_list(coords, eBVM_FREE), outside);
		twin.insert_ids(ids, OctreeDatum::Free);
		EXPECT_EQ(tree.dump(), twin.dump());
	}
}

TEST_F(VulkanTest, OctreeMetaPointCloudInsert)
{
	if (!ctx().info().buffer_device_address)
		GTEST_SKIP() << "bufferDeviceAddress not supported";
	for (const auto kind : kKinds)
	{
		SCOPED_TRACE(name(kind));
		std::mt19937_64 rng(23);
		const float side = 0.2f;
		const size_t sizes[] = { 0, 5, 300, 1000, 0, 40 };
		std::vector<Cloud> clouds;
		std::vector<std::vector<Point3f>> points;
		for (const size_t size : sizes)
		{
			clouds.push_back(random_cloud(rng, size, 32, side, size == 300 ? 6 : 0));
			points.push_back(clouds.back().points);
		}
		const MetaPointCloud meta(ctx(), points);
		auto ids_of = [&](size_t first, size_t last) // sub clouds first .. last - 1
		{
			std::vector<uint64_t> ids;
			uint32_t outside = 0;
			for (size_t i = first; i < last; ++i)
			{
				ids.insert(ids.end(), clouds[i].ids.begin(), clouds[i].ids.end());
				outside += clouds[i].outside;
			}
			return std::pair{ ids, outside };
		};

		// one meaning for everything
		Octree tree(ctx(), kind, side), twin(ctx(), kind);
		build_base(tree, twin, rng);
		const auto all = ids_of(0, clouds.size());
		EXPECT_EQ(tree.insert_meta_point_cloud(meta), all.second);
		twin.insert_ids(all.first);
		EXPECT_EQ(tree.dump(), twin.dump());
		EXPECT_EQ(tree.dropped_count(), twin.dropped_count() + all.second);
		EXPECT_NO_THROW(tree.check_tree());

		// a meaning per sub cloud: runs of equal meanings (empty sub clouds inside a run do not break it) are inserted as one batch, in order
		Octree mixed(ctx(), kind, side), mixed_twin(ctx(), kind);
		build_base(mixed, mixed_twin, rng);
		const std::vector<uint8_t> meanings = { eBVM_OCCUPIED, eBVM_OCCUPIED, eBVM_FREE, eBVM_FREE, eBVM_FREE, eBVM_OCCUPIED };
		const uint32_t outside = mixed.insert_meta_point_cloud(meta, meanings);
		mixed_twin.insert_ids(ids_of(0, 2).first, OctreeDatum::Occupied);
		mixed_twin.insert_ids(ids_of(2, 5).first, OctreeDatum::Free);
		mixed_twin.insert_ids(ids_of(5, 6).first, OctreeDatum::Occupied);
		EXPECT_EQ(outside, all.second);
		EXPECT_EQ(mixed.dump(), mixed_twin.dump());

		// the runs are propagated in between even without propagate_now
		Octree deferred(ctx(), kind, side), deferred_twin(ctx(), kind);
		build_base(deferred, deferred_twin, rng);
		deferred.insert_meta_point_cloud(meta, meanings, false);
		deferred.propagate();
		deferred_twin.insert_ids(ids_of(0, 2).first, OctreeDatum::Occupied);
		deferred_twin.insert_ids(ids_of(2, 5).first, OctreeDatum::Free);
		deferred_twin.insert_ids(ids_of(5, 6).first, OctreeDatum::Occupied);
		EXPECT_EQ(deferred.dump(), deferred_twin.dump());

		// the meanings are checked before anything is inserted
		const auto before = mixed.dump();
		EXPECT_THROW(mixed.insert_meta_point_cloud(meta, std::vector<uint8_t>{ eBVM_OCCUPIED }), VulkanError); // one per sub cloud
		EXPECT_THROW(mixed.insert_meta_point_cloud(meta, std::vector<uint8_t>{ eBVM_OCCUPIED, eBVM_OCCUPIED, eBVM_OCCUPIED, eBVM_OCCUPIED, eBVM_OCCUPIED, eBVM_COLLISION }), VulkanError);
		EXPECT_THROW(mixed.insert_meta_point_cloud(meta, eBVM_UNKNOWN), VulkanError);
		EXPECT_EQ(mixed.dump(), before);
		// propagate_now = false
		mixed.insert_meta_point_cloud(meta, eBVM_FREE, false);
		EXPECT_GT(mixed.pending_update_count(), 0u);
		mixed.propagate();
		EXPECT_NO_THROW(mixed.check_tree());
		// a cloud without any point
		const MetaPointCloud none(ctx(), std::vector<uint32_t>{ 0, 0 });
		const auto kept = mixed.dump();
		EXPECT_EQ(mixed.insert_meta_point_cloud(none), 0u);
		EXPECT_EQ(mixed.dump(), kept);
	}
}

namespace
{
	/// A tree with everything the stored form has to keep: a pruned region cut by inserts (garbage), nodes of higher levels, collision bits,
	/// pending updates (inserted without propagate), several chunks per level.
	void make_eventful_tree(Octree& tree, std::mt19937_64& rng)
	{
		std::vector<OctreeVoxel> voxels;
		for (uint32_t x = 0; x < 8; ++x)
			for (uint32_t y = 0; y < 8; ++y)
				for (uint32_t z = 0; z < 8; ++z)
					voxels.push_back({ x, y, z }); // a full cube: uniform nodes that keep their child blocks
		for (int i = 0; i < 400; ++i)
			voxels.push_back({ 100 + uint32_t(rng() % 64), uint32_t(rng() % 64), uint32_t(rng() % 64) });
		tree.build(voxels);
		const std::vector<OctreeVoxel> cut = { { 3, 3, 3 }, { 4, 5, 6 } };
		tree.insert(cut, OctreeDatum::Free);                              // cuts the uniform cube: garbage
		tree.insert_nodes(3, std::vector<uint64_t>{ 40, 41 }, OctreeDatum::Free);
		std::vector<uint64_t> probes;
		for (int i = 0; i < 200; ++i)
			probes.push_back(ref::morton(uint32_t(rng() % 8), uint32_t(rng() % 8), uint32_t(rng() % 8)));
		(void)tree.collide_with_ids(probes);                               // COLLISION bits
		std::vector<uint64_t> late;
		for (int i = 0; i < 300; ++i)
			late.push_back(ref::morton(200 + uint32_t(rng() % 50), uint32_t(rng() % 50), uint32_t(rng() % 50)));
		tree.insert_ids(late, OctreeDatum::Occupied, false);               // pending updates
	}
}

TEST_F(VulkanTest, OctreeStorageRoundTrip)
{
	if (!ctx().info().buffer_device_address)
		GTEST_SKIP() << "bufferDeviceAddress not supported";
	const OctreeNativeStorage native;
	const OctreeStorage& storage = native; // through the interface
	for (const auto kind : kKinds)
	{
		SCOPED_TRACE(name(kind));
		std::mt19937_64 rng(31);
		Octree original(ctx(), kind, 0.0625f);
		make_eventful_tree(original, rng);
		ASSERT_GT(original.pending_update_count(), 0u);
		ASSERT_GT(original.memory_usage().garbage_blocks[1] + original.memory_usage().garbage_blocks[2] + original.memory_usage().garbage_blocks[3], 0u);

		std::stringstream stream(std::ios::in | std::ios::out | std::ios::binary);
		storage.write(original, stream);
		EXPECT_EQ(original.pending_update_count(), 0u); // writing propagates

		Octree restored(ctx(), kind);
		// the target is replaced: it starts with other content
		restored.build(std::vector<OctreeVoxel>{ { 1, 2, 3 } });
		storage.read(restored, stream);
		EXPECT_EQ(restored.voxel_side_length(), 0.0625f);
		EXPECT_EQ(restored.dump(), original.dump());
		EXPECT_EQ(sorted_cubes(restored), sorted_cubes(original));
		EXPECT_EQ(restored.pending_update_count(), 0u);
		EXPECT_NO_THROW(restored.check_tree());
		EXPECT_EQ(restored.memory_usage().garbage_blocks[1] + restored.memory_usage().garbage_blocks[2] + restored.memory_usage().garbage_blocks[3], 0u);

		std::vector<uint64_t> samples;
		for (int i = 0; i < 2000; ++i)
			samples.push_back(ref::morton(uint32_t(rng() % 260), uint32_t(rng() % 70), uint32_t(rng() % 70)));
		EXPECT_EQ(restored.find(samples), original.find(samples));

		// collisions see the same tree (the stored collision bits too: clear them on one side only and compare after)
		EXPECT_EQ(restored.collide_with_ids(samples), original.collide_with_ids(samples));
		EXPECT_EQ(restored.dump(), original.dump());

		// and both trees go on alike
		std::vector<uint64_t> next;
		for (int i = 0; i < 500; ++i)
			next.push_back(ref::morton(uint32_t(rng() % 300), uint32_t(rng() % 70), uint32_t(rng() % 70)));
		original.insert_ids(next);
		restored.insert_ids(next);
		original.insert_nodes(2, std::vector<uint64_t>{ 3, 9 }, OctreeDatum::Free);
		restored.insert_nodes(2, std::vector<uint64_t>{ 3, 9 }, OctreeDatum::Free);
		EXPECT_EQ(restored.dump(), original.dump());
		EXPECT_NO_THROW(restored.check_tree());
		original.rebuild();
		restored.rebuild();
		EXPECT_EQ(restored.dump(), original.dump());
	}
}

TEST_F(VulkanTest, OctreeStorageOfTheEmptyAndTheSmallestTree)
{
	if (!ctx().info().buffer_device_address)
		GTEST_SKIP() << "bufferDeviceAddress not supported";
	const OctreeNativeStorage storage;
	for (const auto kind : kKinds)
	{
		SCOPED_TRACE(name(kind));
		Octree empty(ctx(), kind);
		std::stringstream stream(std::ios::in | std::ios::out | std::ios::binary);
		storage.write(empty, stream);
		Octree target(ctx(), kind, 2.0f);
		target.build(std::vector<OctreeVoxel>{ { 5, 5, 5 } });
		storage.read(target, stream);
		EXPECT_EQ(target.dump(), empty.dump());
		EXPECT_EQ(target.voxel_side_length(), 0.0f); // the stored state, unset
		EXPECT_NO_THROW(target.check_tree());
		// it is a working tree: an insert does what it does in a new empty tree
		target.insert(std::vector<OctreeVoxel>{ { 5, 5, 5 } });
		Octree fresh(ctx(), kind);
		fresh.insert(std::vector<OctreeVoxel>{ { 5, 5, 5 } });
		EXPECT_EQ(target.dump(), fresh.dump());
	}
}

namespace
{
	template<typename T>
	T at(const std::string& bytes, size_t offset)
	{
		T value;
		std::memcpy(&value, bytes.data() + offset, sizeof(T));
		return value;
	}

	uint64_t fnv1a(const std::string& bytes, size_t size)
	{
		uint64_t hash = 14695981039346656037ull;
		for (size_t i = 0; i < size; ++i)
		{
			hash ^= static_cast<uint8_t>(bytes[i]);
			hash *= 1099511628211ull;
		}
		return hash;
	}

	std::string written(Octree& tree)
	{
		std::stringstream stream(std::ios::in | std::ios::out | std::ios::binary);
		OctreeNativeStorage().write(tree, stream);
		return stream.str();
	}

	void put_checksum(std::string& bytes)
	{
		const uint64_t hash = fnv1a(bytes, bytes.size() - 8);
		std::memcpy(bytes.data() + bytes.size() - 8, &hash, 8);
	}
}

// The bytes are decoded here without the library: header, one record per node of dump(), checksum.
TEST_F(VulkanTest, OctreeNativeFormatLayout)
{
	if (!ctx().info().buffer_device_address)
		GTEST_SKIP() << "bufferDeviceAddress not supported";
	for (const auto kind : kKinds)
	{
		SCOPED_TRACE(name(kind));
		Octree tree(ctx(), kind, 0.5f);
		tree.build(std::vector<OctreeVoxel>{ { 1, 2, 3 }, { 9, 9, 9 } });
		tree.insert(std::vector<OctreeVoxel>{ { 1, 2, 4 } }, OctreeDatum::Free);
		const std::string bytes = written(tree);
		const auto nodes = tree.dump();
		ASSERT_EQ(bytes.size(), 32 + 16 * nodes.size() + 8);
		EXPECT_EQ(std::memcmp(bytes.data(), "GVLOCT\0\1", 8), 0);
		EXPECT_EQ(at<uint32_t>(bytes, 8), 1u);
		EXPECT_EQ(at<uint32_t>(bytes, 12), kind == OctreeKind::Probabilistic ? 1u : 0u);
		EXPECT_EQ(at<uint32_t>(bytes, 16), 0x3f000000u); // 0.5f
		EXPECT_EQ(at<uint32_t>(bytes, 20), 0u);
		EXPECT_EQ(at<uint64_t>(bytes, 24), nodes.size());
		for (size_t i = 0; i < nodes.size(); ++i)
		{
			const size_t at_i = 32 + 16 * i;
			EXPECT_EQ(at<uint64_t>(bytes, at_i), nodes[i].id) << i;
			EXPECT_EQ(uint8_t(bytes[at_i + 8]), nodes[i].level) << i;
			EXPECT_EQ(uint8_t(bytes[at_i + 9]), nodes[i].status) << i;
			EXPECT_EQ(int8_t(bytes[at_i + 10]), nodes[i].occupancy) << i;
			EXPECT_EQ(uint8_t(bytes[at_i + 11]), nodes[i].has_children ? 1 : 0) << i;
			EXPECT_EQ(at<uint32_t>(bytes, at_i + 12), 0u) << i;
		}
		EXPECT_EQ(at<uint64_t>(bytes, bytes.size() - 8), fnv1a(bytes, bytes.size() - 8));
		EXPECT_EQ(nodes.front().level, 0u); // ordered by (level, id): the leaves come first, the root last
		EXPECT_EQ(nodes.back().level, 14u);
	}
}

TEST_F(VulkanTest, OctreeStorageRejectsWhatIsNotAValidTreeAndKeepsTheTarget)
{
	if (!ctx().info().buffer_device_address)
		GTEST_SKIP() << "bufferDeviceAddress not supported";
	const OctreeNativeStorage storage;
	for (const auto kind : kKinds)
	{
		SCOPED_TRACE(name(kind));
		Octree source(ctx(), kind, 0.3f);
		source.build(std::vector<OctreeVoxel>{ { 1, 2, 3 }, { 40, 5, 6 } });
		const std::string good = written(source);

		Octree target(ctx(), kind, 0.7f);
		target.build(std::vector<OctreeVoxel>{ { 7, 7, 7 }, { 8, 8, 8 }, { 9, 9, 9 } });
		const auto kept = target.dump();
		auto expect_rejected = [&](const std::string& bytes, const std::string& what)
		{
			std::istringstream in(bytes, std::ios::binary);
			EXPECT_THROW(storage.read(target, in), VulkanError) << what;
			EXPECT_EQ(target.dump(), kept) << what;
			EXPECT_EQ(target.voxel_side_length(), 0.7f) << what;
		};

		// every cut of the stream, every single damaged byte
		for (size_t cut = 0; cut < good.size(); cut += (cut < 80 ? 1 : 37))
			expect_rejected(good.substr(0, cut), "cut at " + std::to_string(cut));
		expect_rejected(good.substr(0, good.size() - 1), "last byte missing");
		for (size_t i = 0; i < good.size(); i += (i < 80 ? 1 : 13))
		{
			std::string bad = good;
			bad[i] = char(bad[i] ^ 0x5a);
			expect_rejected(bad, "byte " + std::to_string(i) + " damaged");
		}
		{ // trailing bytes after the checksum belong to whatever follows in the stream: they are not read
			std::istringstream in(good + "tail", std::ios::binary);
			Octree other(ctx(), kind);
			EXPECT_NO_THROW(storage.read(other, in));
			EXPECT_EQ(other.dump(), source.dump());
		}

		// a tree of the other kind
		Octree wrong(ctx(), kind == OctreeKind::Probabilistic ? OctreeKind::Deterministic : OctreeKind::Probabilistic);
		{
			std::istringstream in(good, std::ios::binary);
			const auto wrong_kept = wrong.dump();
			EXPECT_THROW(storage.read(wrong, in), VulkanError);
			EXPECT_EQ(wrong.dump(), wrong_kept);
		}

		// version and reserved fields, with a valid checksum
		{
			std::string bytes = good;
			const uint32_t version = 2;
			std::memcpy(bytes.data() + 8, &version, 4);
			put_checksum(bytes);
			expect_rejected(bytes, "version 2");
			bytes = good;
			bytes[0] = 'X';
			put_checksum(bytes);
			expect_rejected(bytes, "wrong magic");
			bytes = good;
			bytes[20] = 1;
			put_checksum(bytes);
			expect_rejected(bytes, "reserved header word");
			bytes = good;
			bytes[32 + 12] = 1;
			put_checksum(bytes);
			expect_rejected(bytes, "reserved record word");
			bytes = good;
			bytes[good.size() - 8 - 16 + 11] = 2; // the root, which does have children
			put_checksum(bytes);
			expect_rejected(bytes, "has_children = 2");
			bytes = good;
			const float negative = -1.0f;
			std::memcpy(bytes.data() + 16, &negative, 4);
			put_checksum(bytes);
			expect_rejected(bytes, "negative side length");
			bytes = good;
			const uint64_t huge = uint64_t(1) << 40;
			std::memcpy(bytes.data() + 24, &huge, 8);
			put_checksum(bytes);
			expect_rejected(bytes, "huge node count");
		}

		// forged streams with a valid checksum but a tree that is not one: the structure check of load_nodes refuses them, the target stays
		{
			const uint64_t count = at<uint64_t>(good, 24);
			ASSERT_GT(count, 20u);
			std::string bytes = good;
			bytes.erase(32 + 16 * 5, 16); // one node of a block missing
			const uint64_t fewer = count - 1;
			std::memcpy(bytes.data() + 24, &fewer, 8);
			put_checksum(bytes);
			expect_rejected(bytes, "a node missing");

			bytes = good;
			bytes[32 + 8] = 15; // level 15 does not exist
			put_checksum(bytes);
			expect_rejected(bytes, "level 15");

			bytes = good;
			const size_t last = 32 + 16 * (count - 1); // the root: claims no children
			bytes[last + 11] = 0;
			put_checksum(bytes);
			expect_rejected(bytes, "root without children (blocks without a parent)");
		}
	}
}

TEST_F(VulkanTest, OctreeStorageFiles)
{
	if (!ctx().info().buffer_device_address)
		GTEST_SKIP() << "bufferDeviceAddress not supported";
	const OctreeNativeStorage storage;
	std::mt19937_64 rng(77);
	Octree original(ctx(), OctreeKind::Probabilistic, 0.125f);
	make_eventful_tree(original, rng);
	const auto path = (std::filesystem::temp_directory_path() / "gvl_octree_storage_test.octree").string();
	storage.write_file(original, path);
	Octree restored(ctx(), OctreeKind::Probabilistic);
	storage.read_file(restored, path);
	EXPECT_EQ(restored.dump(), original.dump());
	EXPECT_EQ(restored.voxel_side_length(), 0.125f);
	std::remove(path.c_str());

	EXPECT_THROW(storage.read_file(restored, path), VulkanError);   // gone
	const auto missing_dir = (std::filesystem::temp_directory_path() / "gvl_no_such_directory" / "x.octree").string();
	EXPECT_THROW(storage.write_file(original, missing_dir), VulkanError);
	EXPECT_EQ(restored.dump(), original.dump());
}

// The limit of one insert is that of the sort primitive (2^24 voxels): more points are refused before anything happens, from a point span and from a
// meta point cloud (whose points live on the GPU already).
TEST_F(VulkanTest, OctreePointCloudLimit)
{
	if (!ctx().info().buffer_device_address)
		GTEST_SKIP() << "bufferDeviceAddress not supported";
	Octree tree(ctx(), OctreeKind::Deterministic, 1.0f);
	tree.build(std::vector<OctreeVoxel>{ { 1, 2, 3 } });
	const auto before = tree.dump();
	{
		const std::vector<Point3f> points(size_t(Octree::kMaxVoxels) + 1, Point3f{ 0.5f, 0.5f, 0.5f });
		EXPECT_THROW(tree.insert_point_cloud(points), VulkanError);
		EXPECT_EQ(tree.dump(), before);
	}
	{
		const MetaPointCloud meta(ctx(), std::vector<uint32_t>{ Octree::kMaxVoxels + 1 });
		EXPECT_THROW(tree.insert_meta_point_cloud(meta), VulkanError);
		EXPECT_EQ(tree.dump(), before);
	}
	{ // exactly the limit works (all points in one voxel: one new voxel)
		const MetaPointCloud meta(ctx(), std::vector<uint32_t>{ Octree::kMaxVoxels });
		EXPECT_EQ(tree.insert_meta_point_cloud(meta), 0u);
		const auto found = tree.find(std::vector<uint64_t>{ 0 });
		EXPECT_EQ(found[0].level, 0u);
		EXPECT_NE(found[0].status & 4, 0);
	}
}

// More nodes than one read batch of the native format (65 536 records).
TEST_F(VulkanTest, OctreeStorageOfALargeTree)
{
	if (!ctx().info().buffer_device_address)
		GTEST_SKIP() << "bufferDeviceAddress not supported";
	const OctreeNativeStorage storage;
	for (const auto kind : kKinds)
	{
		SCOPED_TRACE(name(kind));
		std::mt19937_64 rng(404);
		std::vector<OctreeVoxel> voxels;
		for (int i = 0; i < 30000; ++i)
			voxels.push_back({ uint32_t(rng() % 256), uint32_t(rng() % 256), uint32_t(rng() % 256) });
		Octree original(ctx(), kind, 0.01f);
		original.build(voxels);
		ASSERT_GT(original.dump().size(), 3u * 65536u);
		std::stringstream stream(std::ios::in | std::ios::out | std::ios::binary);
		storage.write(original, stream);
		Octree restored(ctx(), kind);
		storage.read(restored, stream);
		EXPECT_EQ(restored.dump(), original.dump());
		EXPECT_NO_THROW(restored.check_tree());
	}
}
