// Slice S7-1: the façade GpuVoxels / GpuVoxelsMap. The classes behind it are tested elsewhere (S1 .. S6f); here the dispatch is checked: the right class
// for every map type, the same result as the class used directly (a twin built in this file with the same calls), what the host expects for the bit map,
// the rules of names, clearing, rebuilding, sizes, initialisation and the singleton.
#include "vulkan_fixture.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <memory>
#include <random>
#include <string>
#include <vector>

#include <gpu_voxels/vulkan/GpuVoxels.h>

#include "facade_support.h"

using namespace gpu_voxels::vulkan;
using gvl_test::snapshot;
using gvl_test::VulkanTest;

namespace
{
	constexpr Dim3ui kDims{ 40, 30, 20 };
	constexpr float kSide = 0.1f;

	struct TypeCase
	{
		MapType type;
		uint8_t a; ///< meaning of the first and third insert
		uint8_t b; ///< meaning of the second insert (and the second sub cloud)
	};

	const TypeCase kCases[] = {
		{ MT_BITVECTOR_VOXELMAP, 5, 9 },
		{ MT_PROBAB_VOXELMAP, eBVM_OCCUPIED, eBVM_FREE },
		{ MT_DISTANCE_VOXELMAP, eBVM_OCCUPIED, eBVM_OCCUPIED },
		{ MT_BITVECTOR_VOXELLIST, 5, 9 },
		{ MT_COUNTING_VOXELLIST, eBVM_OCCUPIED, eBVM_OCCUPIED },
		{ MT_BITVECTOR_OCTREE, eBVM_OCCUPIED, eBVM_FREE },
		{ MT_PROBAB_OCTREE, eBVM_OCCUPIED, eBVM_FREE },
	};

	struct Scenario
	{
		std::vector<Point3f> points;
		std::vector<Coord3ui> coords;
		std::vector<std::vector<Point3f>> sub_clouds;
	};

	Point3f mid_voxel(std::mt19937_64& rng, Dim3ui extent)
	{
		std::uniform_real_distribution<float> f(0.25f, 0.75f);
		return { (static_cast<float>(rng() % extent.x) + f(rng)) * kSide, (static_cast<float>(rng() % extent.y) + f(rng)) * kSide,
			(static_cast<float>(rng() % extent.z) + f(rng)) * kSide };
	}

	Scenario make_scenario(uint64_t seed, bool with_bad)
	{
		std::mt19937_64 rng(seed);
		Scenario s;
		for (int i = 0; i < 300; ++i)
			s.points.push_back(mid_voxel(rng, kDims));
		for (int i = 0; i < 100; ++i)
			s.coords.push_back({ static_cast<uint32_t>(rng() % kDims.x), static_cast<uint32_t>(rng() % kDims.y), static_cast<uint32_t>(rng() % kDims.z) });
		for (int c = 0; c < 3; ++c)
		{
			std::vector<Point3f> cloud;
			for (int i = 0; i < 60; ++i)
				cloud.push_back(mid_voxel(rng, kDims));
			s.sub_clouds.push_back(std::move(cloud));
		}
		if (with_bad)
		{
			s.points.push_back({ -0.5f * kSide, 1.5f * kSide, 1.5f * kSide });
			s.points.push_back({ 43.5f * kSide, 1.5f * kSide, 1.5f * kSide }); // outside the maps, inside the octree
			s.points.push_back({ std::numeric_limits<float>::quiet_NaN(), 1.5f * kSide, 1.5f * kSide });
			s.coords.push_back({ 45, 1, 1 });
			s.coords.push_back({ 20000, 1, 1 });
			s.sub_clouds[1].push_back({ 1.5f * kSide, -2.5f * kSide, 1.5f * kSide });
		}
		return s;
	}

	/// The calls the façade must forward, made on a class directly.
	template<class T>
	std::vector<uint32_t> run_direct_ops(T& m, const Scenario& s, const MetaPointCloud& cloud, uint8_t a, uint8_t b)
	{
		std::vector<uint32_t> out;
		out.push_back(m.insert_point_cloud(s.points, a));
		out.push_back(m.insert_coordinate_list(s.coords, b));
		out.push_back(m.insert_meta_point_cloud(cloud, a));
		const uint8_t meanings[3] = { a, b, a };
		if constexpr (std::is_same_v<T, DistanceVoxelMap>)
			out.push_back(m.insert_meta_point_cloud(cloud, eBVM_OCCUPIED)); // occupied is its only meaning
		else
			out.push_back(m.insert_meta_point_cloud(cloud, std::span<const uint8_t>(meanings)));
		return out;
	}

	struct RunResult
	{
		std::vector<uint32_t> outside;
		std::vector<uint64_t> state;
	};

	/// The class of `type` built directly (not through the façade), `s` null = nothing inserted.
	RunResult run_direct(Context& c, MapType type, const Scenario* s, const MetaPointCloud* cloud, uint8_t a, uint8_t b)
	{
		auto go = [&](auto& m) {
			RunResult r;
			if (s != nullptr)
				r.outside = run_direct_ops(m, *s, *cloud, a, b);
			r.state = snapshot(m);
			return r;
		};
		switch (type)
		{
		case MT_BITVECTOR_VOXELMAP: { BitVoxelMap m(c, kDims, kSide); return go(m); }
		case MT_PROBAB_VOXELMAP: { ProbVoxelMap m(c, kDims, kSide); return go(m); }
		case MT_DISTANCE_VOXELMAP: { DistanceVoxelMap m(c, kDims, kSide); return go(m); }
		case MT_BITVECTOR_VOXELLIST: { BitVoxelList m(c, kDims, kSide); return go(m); }
		case MT_COUNTING_VOXELLIST: { CountingVoxelList m(c, kDims, kSide); return go(m); }
		case MT_BITVECTOR_OCTREE: { Octree m(c, OctreeKind::Deterministic, kSide); return go(m); }
		case MT_PROBAB_OCTREE: { Octree m(c, OctreeKind::Probabilistic, kSide); return go(m); }
		default: break;
		}
		ADD_FAILURE() << "type not built";
		return {};
	}

	std::vector<uint32_t> run_facade_ops(GpuVoxelsMap& m, const Scenario& s, const MetaPointCloud& cloud, uint8_t a, uint8_t b)
	{
		std::vector<uint32_t> out;
		out.push_back(m.insert_point_cloud(s.points, a));
		out.push_back(m.insert_coordinate_list(s.coords, b));
		out.push_back(m.insert_meta_point_cloud(cloud, a));
		const uint8_t meanings[3] = { a, b, a };
		out.push_back(m.insert_meta_point_cloud(cloud, std::span<const uint8_t>(meanings)));
		return out;
	}

	class FacadeTest : public VulkanTest
	{
	protected:
		std::unique_ptr<GpuVoxels> make()
		{
			auto gvl = std::make_unique<GpuVoxels>(ctx());
			gvl->initialize(kDims, kSide);
			return gvl;
		}
		MetaPointCloud meta(const Scenario& s) { return MetaPointCloud(ctx(), std::span<const std::vector<Point3f>>(s.sub_clouds)); }
	};
}

TEST_F(FacadeTest, EveryBuiltTypeIsItsClass)
{
	for (const TypeCase& tc : kCases)
	{
		const auto gvl = make();
		const GpuVoxelsMapSharedPtr m = gvl->add_map(tc.type, "m");
		ASSERT_TRUE(m) << map_type_name(tc.type);
		EXPECT_EQ(m->type(), tc.type);
		const int held = int(m->is<BitVoxelMap>()) + int(m->is<ProbVoxelMap>()) + int(m->is<DistanceVoxelMap>()) + int(m->is<BitVoxelList>())
			+ int(m->is<CountingVoxelList>()) + int(m->is<Octree>());
		EXPECT_EQ(held, 1) << map_type_name(tc.type);
		EXPECT_EQ(m->is<BitVoxelMap>(), tc.type == MT_BITVECTOR_VOXELMAP);
		EXPECT_EQ(m->is<ProbVoxelMap>(), tc.type == MT_PROBAB_VOXELMAP);
		EXPECT_EQ(m->is<DistanceVoxelMap>(), tc.type == MT_DISTANCE_VOXELMAP);
		EXPECT_EQ(m->is<BitVoxelList>(), tc.type == MT_BITVECTOR_VOXELLIST);
		EXPECT_EQ(m->is<CountingVoxelList>(), tc.type == MT_COUNTING_VOXELLIST);
		EXPECT_EQ(m->is<Octree>(), tc.type == MT_BITVECTOR_OCTREE || tc.type == MT_PROBAB_OCTREE);
		if (const Octree* tree = m->as<Octree>())
			EXPECT_EQ(tree->kind(), tc.type == MT_PROBAB_OCTREE ? OctreeKind::Probabilistic : OctreeKind::Deterministic);
		const GpuVoxelsMap& view = *m;
		EXPECT_EQ(view.as<Octree>() != nullptr, m->is<Octree>()); // the const overload
		EXPECT_EQ(view.as<BitVoxelMap>() != nullptr, m->is<BitVoxelMap>());
	}
}

TEST_F(FacadeTest, TypesThatAreNotBuiltThrowAndAddNothing)
{
	const auto gvl = make();
	for (const MapType type : { MT_PROBAB_VOXELLIST, MT_BITVECTOR_MORTON_VOXELLIST, MT_PROBAB_MORTON_VOXELLIST })
	{
		try
		{
			(void)gvl->add_map(type, "x");
			ADD_FAILURE() << "no exception for " << map_type_name(type);
		}
		catch (const VulkanError& e)
		{
			EXPECT_NE(std::string(e.what()).find(map_type_name(type)), std::string::npos) << e.what();
			EXPECT_NE(std::string(e.what()).find("not implemented"), std::string::npos) << e.what();
		}
	}
	EXPECT_THROW((void)gvl->add_map(static_cast<MapType>(77), "x"), VulkanError);
	EXPECT_FALSE(gvl->get_map("x")); // nothing was registered
	EXPECT_TRUE(gvl->add_map(MT_BITVECTOR_VOXELMAP, "x"));
	// a constructor that throws (distance maps allow 1023 voxels per axis) adds nothing either
	GpuVoxels big(ctx());
	big.initialize({ 2000, 4, 4 }, kSide);
	EXPECT_THROW((void)big.add_map(MT_DISTANCE_VOXELMAP, "d"), VulkanError);
	EXPECT_FALSE(big.get_map("d"));
	EXPECT_TRUE(big.add_map(MT_BITVECTOR_VOXELMAP, "d")); // the name is free again
}

TEST_F(FacadeTest, NamesAreUnique)
{
	const auto gvl = make();
	const GpuVoxelsMapSharedPtr a = gvl->add_map(MT_BITVECTOR_VOXELMAP, "a");
	ASSERT_TRUE(a);
	EXPECT_FALSE(gvl->add_map(MT_PROBAB_VOXELMAP, "a")); // same name, other type: refused, the first stays
	EXPECT_EQ(gvl->get_map("a"), a);
	EXPECT_EQ(gvl->get_map("a")->type(), MT_BITVECTOR_VOXELMAP);
	const GpuVoxelsMapSharedPtr b = gvl->add_map(MT_PROBAB_VOXELMAP, "b");
	ASSERT_TRUE(b);
	EXPECT_NE(a, b);
	EXPECT_EQ(gvl->get_map("b"), b);
	EXPECT_FALSE(gvl->get_map("c"));
	EXPECT_TRUE(gvl->del_map("a"));
	EXPECT_FALSE(gvl->del_map("a"));
	EXPECT_FALSE(gvl->get_map("a"));
	EXPECT_EQ(gvl->get_map("b"), b); // the other map is untouched
	const GpuVoxelsMapSharedPtr again = gvl->add_map(MT_COUNTING_VOXELLIST, "a");
	ASSERT_TRUE(again);
	EXPECT_NE(again, a);
	EXPECT_EQ(again->type(), MT_COUNTING_VOXELLIST);
	EXPECT_TRUE(a->is<BitVoxelMap>()); // a handle that was handed out stays valid after del_map
	EXPECT_FALSE(gvl->clear_map("zz"));
	EXPECT_FALSE(gvl->clear_map("zz", 5));
	EXPECT_FALSE(gvl->del_map(""));
}

TEST_F(FacadeTest, InsertsDoWhatTheClassDoes)
{
	const Scenario s = make_scenario(7, true);
	const MetaPointCloud cloud = meta(s);
	for (const TypeCase& tc : kCases)
	{
		const auto gvl = make();
		const GpuVoxelsMapSharedPtr m = gvl->add_map(tc.type, "m");
		const RunResult direct = run_direct(ctx(), tc.type, &s, &cloud, tc.a, tc.b);
		const std::vector<uint32_t> outside = run_facade_ops(*m, s, cloud, tc.a, tc.b);
		EXPECT_EQ(outside, direct.outside) << map_type_name(tc.type);
		EXPECT_EQ(snapshot(*m), direct.state) << map_type_name(tc.type);
		EXPECT_NE(direct.state, run_direct(ctx(), tc.type, nullptr, nullptr, 0, 0).state) << map_type_name(tc.type) << ": the inserts changed nothing";
		const bool octree = tc.type == MT_BITVECTOR_OCTREE || tc.type == MT_PROBAB_OCTREE;
		EXPECT_GT(outside[0] + outside[1], octree ? 0u : 1u) << map_type_name(tc.type); // points / coordinates outside were counted
	}
}

TEST_F(FacadeTest, BitMapContentsAreWhatTheHostExpects)
{
	const auto gvl = make();
	const GpuVoxelsMapSharedPtr m = gvl->add_map(MT_BITVECTOR_VOXELMAP, "m");
	std::mt19937_64 rng(3);
	std::vector<uint32_t> expected(size_t(kDims.voxel_count()) * 8, 0);
	auto set = [&](uint64_t x, uint64_t y, uint64_t z, uint8_t meaning) {
		expected[((z * kDims.y + y) * kDims.x + x) * 8 + meaning / 32] |= 1u << (meaning % 32);
	};
	for (const uint8_t meaning : { uint8_t(5), uint8_t(40), uint8_t(200) })
	{
		std::vector<Point3f> points;
		for (int i = 0; i < 50; ++i)
			points.push_back(mid_voxel(rng, kDims));
		for (const Point3f& p : points)
			set(uint64_t(p.x / kSide), uint64_t(p.y / kSide), uint64_t(p.z / kSide), meaning);
		EXPECT_EQ(m->insert_point_cloud(points, meaning), 0u);
	}
	const std::vector<Coord3ui> coords = { { 0, 0, 0 }, { 39, 29, 19 }, { 12, 3, 7 } };
	for (const Coord3ui& c : coords)
		set(c.x, c.y, c.z, 9);
	EXPECT_EQ(m->insert_coordinate_list(coords, 9), 0u);
	EXPECT_EQ(m->as<BitVoxelMap>()->download_words(), expected);
}

TEST_F(FacadeTest, MeaningsPerSubCloud)
{
	const Scenario s = make_scenario(11, false);
	const MetaPointCloud cloud = meta(s);
	for (const TypeCase& tc : kCases)
	{
		const auto gvl = make();
		const GpuVoxelsMapSharedPtr m = gvl->add_map(tc.type, "m");
		const std::vector<uint64_t> before = snapshot(*m);
		const uint8_t one[1] = { tc.a };
		const uint8_t four[4] = { tc.a, tc.a, tc.a, tc.a };
		EXPECT_THROW((void)m->insert_meta_point_cloud(cloud, std::span<const uint8_t>(one)), VulkanError) << map_type_name(tc.type);
		EXPECT_THROW((void)m->insert_meta_point_cloud(cloud, std::span<const uint8_t>(four)), VulkanError) << map_type_name(tc.type);
		EXPECT_EQ(snapshot(*m), before) << map_type_name(tc.type);
	}
	// the distance map has one meaning: a sub cloud with another one is refused before anything is inserted
	const auto gvl = make();
	const GpuVoxelsMapSharedPtr d = gvl->add_map(MT_DISTANCE_VOXELMAP, "d");
	const uint8_t bad[3] = { eBVM_OCCUPIED, 5, eBVM_OCCUPIED };
	try
	{
		(void)d->insert_meta_point_cloud(cloud, std::span<const uint8_t>(bad));
		ADD_FAILURE() << "no exception";
	}
	catch (const VulkanError& e)
	{
		EXPECT_NE(std::string(e.what()).find("eBVM_OCCUPIED"), std::string::npos) << e.what();
	}
	EXPECT_EQ(snapshot(*d), run_direct(ctx(), MT_DISTANCE_VOXELMAP, nullptr, nullptr, 0, 0).state);
	const uint8_t ok[3] = { eBVM_OCCUPIED, eBVM_OCCUPIED, eBVM_OCCUPIED };
	EXPECT_EQ(d->insert_meta_point_cloud(cloud, std::span<const uint8_t>(ok)), 0u);
	const GpuVoxelsMapSharedPtr e = gvl->add_map(MT_DISTANCE_VOXELMAP, "e");
	EXPECT_EQ(e->insert_meta_point_cloud(cloud, eBVM_OCCUPIED), 0u);
	EXPECT_EQ(snapshot(*d), snapshot(*e));
	// meanings the class does not accept throw through the façade too
	const GpuVoxelsMapSharedPtr tree = gvl->add_map(MT_BITVECTOR_OCTREE, "t");
	EXPECT_THROW((void)tree->insert_point_cloud(s.points, 5), VulkanError);
	EXPECT_THROW((void)d->insert_point_cloud(s.points, 5), VulkanError);
}

TEST_F(FacadeTest, ClearMap)
{
	const Scenario s = make_scenario(13, false);
	const MetaPointCloud cloud = meta(s);
	for (const TypeCase& tc : kCases)
	{
		const auto gvl = make();
		const GpuVoxelsMapSharedPtr m = gvl->add_map(tc.type, "m");
		const std::vector<uint64_t> fresh = snapshot(*m);
		run_facade_ops(*m, s, cloud, tc.a, tc.b);
		ASSERT_NE(snapshot(*m), fresh) << map_type_name(tc.type);
		EXPECT_TRUE(gvl->clear_map("m"));
		EXPECT_EQ(snapshot(*m), fresh) << map_type_name(tc.type);
	}
}

TEST_F(FacadeTest, ClearOneMeaning)
{
	const Scenario s = make_scenario(17, false);
	const auto gvl = make();
	// bit map: only the bit of the meaning goes (checked against the class)
	{
		const GpuVoxelsMapSharedPtr m = gvl->add_map(MT_BITVECTOR_VOXELMAP, "bit");
		BitVoxelMap twin(ctx(), kDims, kSide);
		for (auto* target : { m->as<BitVoxelMap>(), &twin })
		{
			target->insert_point_cloud(s.points, 5);
			target->insert_coordinate_list(s.coords, 9);
		}
		const std::vector<uint64_t> both = snapshot(*m);
		EXPECT_TRUE(gvl->clear_map("bit", 5));
		twin.clear_bit_voxel_meaning(5);
		EXPECT_EQ(snapshot(*m), snapshot(twin));
		EXPECT_NE(snapshot(*m), both);
		EXPECT_NE(snapshot(*m), run_direct(ctx(), MT_BITVECTOR_VOXELMAP, nullptr, nullptr, 0, 0).state); // meaning 9 is still there
	}
	// prob. map and octrees: occupied clears everything, any other meaning throws and changes nothing
	for (const MapType type : { MT_PROBAB_VOXELMAP, MT_BITVECTOR_OCTREE, MT_PROBAB_OCTREE })
	{
		const GpuVoxelsMapSharedPtr m = gvl->add_map(type, std::string("occ") + map_type_name(type));
		const std::vector<uint64_t> fresh = snapshot(*m);
		m->insert_point_cloud(s.points, eBVM_OCCUPIED);
		const std::vector<uint64_t> filled = snapshot(*m);
		ASSERT_NE(filled, fresh);
		EXPECT_THROW(m->clear_bit_voxel_meaning(eBVM_FREE), VulkanError) << map_type_name(type);
		EXPECT_THROW(m->clear_bit_voxel_meaning(5), VulkanError) << map_type_name(type);
		EXPECT_EQ(snapshot(*m), filled) << map_type_name(type);
		m->clear_bit_voxel_meaning(eBVM_OCCUPIED);
		EXPECT_EQ(snapshot(*m), fresh) << map_type_name(type);
	}
	// distance map and lists do not support it
	for (const MapType type : { MT_DISTANCE_VOXELMAP, MT_BITVECTOR_VOXELLIST, MT_COUNTING_VOXELLIST })
	{
		const GpuVoxelsMapSharedPtr m = gvl->add_map(type, std::string("un") + map_type_name(type));
		m->insert_point_cloud(s.points, eBVM_OCCUPIED);
		const std::vector<uint64_t> filled = snapshot(*m);
		for (const uint8_t meaning : { uint8_t(eBVM_OCCUPIED), uint8_t(5) })
			try
			{
				m->clear_bit_voxel_meaning(meaning);
				ADD_FAILURE() << "no exception for " << map_type_name(type);
			}
			catch (const VulkanError& e)
			{
				EXPECT_NE(std::string(e.what()).find("not supported"), std::string::npos) << e.what();
				EXPECT_NE(std::string(e.what()).find(map_type_name(type)), std::string::npos) << e.what();
			}
		EXPECT_EQ(snapshot(*m), filled);
		EXPECT_THROW(gvl->clear_map(std::string("un") + map_type_name(type), 5), VulkanError);
	}
}

TEST_F(FacadeTest, Rebuild)
{
	const Scenario s = make_scenario(19, false);
	const auto gvl = make();
	for (const MapType type : { MT_BITVECTOR_VOXELMAP, MT_PROBAB_VOXELMAP, MT_DISTANCE_VOXELMAP, MT_BITVECTOR_VOXELLIST, MT_COUNTING_VOXELLIST })
	{
		const GpuVoxelsMapSharedPtr m = gvl->add_map(type, map_type_name(type));
		m->insert_point_cloud(s.points, eBVM_OCCUPIED);
		const std::vector<uint64_t> before = snapshot(*m);
		EXPECT_FALSE(m->needs_rebuild()) << map_type_name(type);
		EXPECT_FALSE(m->rebuild()) << map_type_name(type);
		EXPECT_FALSE(m->rebuild_if_needed()) << map_type_name(type);
		EXPECT_EQ(snapshot(*m), before) << map_type_name(type);
	}
	for (const MapType type : { MT_BITVECTOR_OCTREE, MT_PROBAB_OCTREE })
	{
		const GpuVoxelsMapSharedPtr m = gvl->add_map(type, map_type_name(type));
		Octree& tree = *m->as<Octree>();
		m->insert_point_cloud(s.points, eBVM_OCCUPIED);
		const std::vector<uint64_t> before = snapshot(*m);
		EXPECT_FALSE(m->needs_rebuild()) << map_type_name(type); // default policy: 200 MB
		EXPECT_FALSE(m->rebuild_if_needed());
		EXPECT_TRUE(m->rebuild());
		EXPECT_EQ(snapshot(*m), before);
		OctreeRebuildPolicy policy;
		policy.max_bytes = 1;
		tree.set_rebuild_policy(policy);
		EXPECT_TRUE(m->needs_rebuild()) << map_type_name(type);
		EXPECT_TRUE(m->rebuild_if_needed()) << map_type_name(type);
		EXPECT_EQ(snapshot(*m), before);
		// a hard insert of a node that covers everything leaves the blocks below it as dead data that only a rebuild frees; what the tree shows
		// (extract) stays the same
		auto cubes = [&tree] {
			auto c = tree.extract();
			std::sort(c.begin(), c.end(), [](const OctreeCube& x, const OctreeCube& y) { return x.level != y.level ? x.level < y.level : x.id < y.id; });
			return c;
		};
		const uint64_t covering = 0;
		tree.insert_nodes(6, std::span<const uint64_t>(&covering, 1));
		const std::vector<OctreeCube> shown = cubes();
		const uint64_t bytes = tree.memory_usage().bytes;
		EXPECT_TRUE(m->rebuild_if_needed()) << map_type_name(type); // the policy of one byte is still set
		EXPECT_LT(tree.memory_usage().bytes, bytes) << map_type_name(type);
		EXPECT_EQ(cubes(), shown) << map_type_name(type);
		tree.insert_nodes(6, std::span<const uint64_t>(&covering, 1), OctreeDatum::Free);
		const uint64_t bytes2 = tree.memory_usage().bytes;
		EXPECT_TRUE(m->rebuild()) << map_type_name(type);
		EXPECT_LE(tree.memory_usage().bytes, bytes2) << map_type_name(type);
	}
}

TEST(FacadeNames, MapTypeNames)
{
	EXPECT_STREQ(map_type_name(MT_BITVECTOR_VOXELMAP), "MT_BITVECTOR_VOXELMAP");
	EXPECT_STREQ(map_type_name(MT_BITVECTOR_VOXELLIST), "MT_BITVECTOR_VOXELLIST");
	EXPECT_STREQ(map_type_name(MT_BITVECTOR_OCTREE), "MT_BITVECTOR_OCTREE");
	EXPECT_STREQ(map_type_name(MT_BITVECTOR_MORTON_VOXELLIST), "MT_BITVECTOR_MORTON_VOXELLIST");
	EXPECT_STREQ(map_type_name(MT_PROBAB_VOXELMAP), "MT_PROBAB_VOXELMAP");
	EXPECT_STREQ(map_type_name(MT_PROBAB_VOXELLIST), "MT_PROBAB_VOXELLIST");
	EXPECT_STREQ(map_type_name(MT_PROBAB_OCTREE), "MT_PROBAB_OCTREE");
	EXPECT_STREQ(map_type_name(MT_PROBAB_MORTON_VOXELLIST), "MT_PROBAB_MORTON_VOXELLIST");
	EXPECT_STREQ(map_type_name(MT_COUNTING_VOXELLIST), "MT_COUNTING_VOXELLIST");
	EXPECT_STREQ(map_type_name(MT_DISTANCE_VOXELMAP), "MT_DISTANCE_VOXELMAP");
	EXPECT_STREQ(map_type_name(static_cast<MapType>(77)), "unknown map type");
}

TEST_F(FacadeTest, Sizes)
{
	const Scenario s = make_scenario(23, false);
	const auto gvl = make();
	const uint64_t voxels = kDims.voxel_count();
	for (const TypeCase& tc : kCases)
	{
		const GpuVoxelsMapSharedPtr m = gvl->add_map(tc.type, std::string("s") + map_type_name(tc.type));
		const bool octree = tc.type == MT_BITVECTOR_OCTREE || tc.type == MT_PROBAB_OCTREE;
		const Dim3ui expected = octree ? Dim3ui{ 16384, 16384, 16384 } : kDims;
		EXPECT_EQ(m->dimensions(), expected) << map_type_name(tc.type);
		EXPECT_EQ(m->voxel_side_length(), kSide) << map_type_name(tc.type);
		const Point3f metric = m->metric_dimensions();
		EXPECT_FLOAT_EQ(metric.x, float(expected.x) * kSide);
		EXPECT_FLOAT_EQ(metric.y, float(expected.y) * kSide);
		EXPECT_FLOAT_EQ(metric.z, float(expected.z) * kSide);
		const uint64_t empty = m->memory_usage_bytes();
		switch (tc.type)
		{
		case MT_BITVECTOR_VOXELMAP: EXPECT_EQ(empty, voxels * 32); break;
		case MT_PROBAB_VOXELMAP: case MT_DISTANCE_VOXELMAP: EXPECT_EQ(empty, voxels * 4); break;
		default: break;
		}
		m->insert_point_cloud(s.points, tc.a);
		if (octree)
		{
			EXPECT_EQ(m->memory_usage_bytes(), m->as<Octree>()->memory_usage().bytes);
			EXPECT_GT(m->memory_usage_bytes(), empty);
		}
		else if (tc.type == MT_BITVECTOR_VOXELLIST || tc.type == MT_COUNTING_VOXELLIST)
			EXPECT_GT(m->memory_usage_bytes(), empty) << map_type_name(tc.type);
	}
	// the dimensions of the façade
	EXPECT_EQ(gvl->dimensions(), kDims);
	EXPECT_EQ(gvl->voxel_side_length(), kSide);
}

TEST_F(FacadeTest, InsertsIntoNamedMaps)
{
	const Scenario s = make_scenario(29, true);
	const MetaPointCloud cloud = meta(s);
	const auto gvl = make();
	const GpuVoxelsMapSharedPtr m = gvl->add_map(MT_BITVECTOR_VOXELMAP, "bit");
	BitVoxelMap twin(ctx(), kDims, kSide);

	uint32_t outside = 99;
	EXPECT_FALSE(gvl->insert_point_cloud_into_map(s.points, "nope", 5, &outside));
	EXPECT_FALSE(gvl->insert_meta_point_cloud_into_map(cloud, "nope", 5, &outside));
	const uint8_t meanings[3] = { 5, 6, 7 };
	EXPECT_FALSE(gvl->insert_meta_point_cloud_into_map(cloud, "nope", std::span<const uint8_t>(meanings), &outside));
	EXPECT_EQ(outside, 99u); // not touched
	EXPECT_EQ(snapshot(*m), snapshot(twin));

	outside = 99;
	EXPECT_TRUE(gvl->insert_point_cloud_into_map(s.points, "bit", 5, &outside));
	EXPECT_EQ(outside, twin.insert_point_cloud(s.points, 5));
	EXPECT_EQ(outside, 3u); // negative, behind the edge, NaN
	EXPECT_EQ(snapshot(*m), snapshot(twin));

	outside = 99;
	EXPECT_TRUE(gvl->insert_meta_point_cloud_into_map(cloud, "bit", 8, &outside));
	EXPECT_EQ(outside, twin.insert_meta_point_cloud(cloud, 8));
	EXPECT_EQ(outside, 1u);
	EXPECT_EQ(snapshot(*m), snapshot(twin));

	outside = 99;
	EXPECT_TRUE(gvl->insert_meta_point_cloud_into_map(cloud, "bit", std::span<const uint8_t>(meanings), &outside));
	EXPECT_EQ(outside, twin.insert_meta_point_cloud(cloud, std::span<const uint8_t>(meanings)));
	EXPECT_EQ(snapshot(*m), snapshot(twin));

	EXPECT_TRUE(gvl->insert_point_cloud_into_map(s.points, "bit", 5)); // no counter wanted
	EXPECT_TRUE(gvl->insert_meta_point_cloud_into_map(cloud, "bit", 8));
	EXPECT_TRUE(gvl->insert_meta_point_cloud_into_map(cloud, "bit", std::span<const uint8_t>(meanings)));

	EXPECT_TRUE(gvl->clear_map("bit", 5));
	twin.clear_bit_voxel_meaning(5);
	EXPECT_EQ(snapshot(*m), snapshot(twin));
	EXPECT_TRUE(gvl->clear_map("bit"));
	twin.clear_map();
	EXPECT_EQ(snapshot(*m), snapshot(twin));

	EXPECT_FALSE(gvl->visualize_map("bit"));
	EXPECT_FALSE(gvl->visualize_map("bit", false));
	EXPECT_FALSE(gvl->visualize_map("nope"));
}

TEST_F(FacadeTest, Initialize)
{
	GpuVoxels gvl(ctx());
	EXPECT_FALSE(gvl.is_initialized());
	try
	{
		(void)gvl.add_map(MT_BITVECTOR_VOXELMAP, "a");
		ADD_FAILURE() << "no exception";
	}
	catch (const VulkanError& e)
	{
		EXPECT_NE(std::string(e.what()).find("initialize"), std::string::npos) << e.what(); // not the error of a map without size
	}
	EXPECT_EQ(&gvl.context(), &ctx());
	EXPECT_THROW(gvl.initialize({ 0, 4, 4 }, 0.1f), VulkanError);
	EXPECT_THROW(gvl.initialize({ 4, 0, 4 }, 0.1f), VulkanError);
	EXPECT_THROW(gvl.initialize({ 4, 4, 0 }, 0.1f), VulkanError);
	for (const float bad : { 0.0f, -0.1f, std::numeric_limits<float>::quiet_NaN(), std::numeric_limits<float>::infinity() })
		EXPECT_THROW(gvl.initialize({ 4, 4, 4 }, bad), VulkanError) << bad;
	EXPECT_FALSE(gvl.is_initialized()); // failed calls changed nothing
	EXPECT_THROW((void)gvl.add_map(MT_BITVECTOR_VOXELMAP, "a"), VulkanError);

	gvl.initialize({ 8, 6, 4 }, 0.25f);
	EXPECT_TRUE(gvl.is_initialized());
	EXPECT_EQ(gvl.dimensions(), (Dim3ui{ 8, 6, 4 }));
	EXPECT_EQ(gvl.voxel_side_length(), 0.25f);
	const GpuVoxelsMapSharedPtr a = gvl.add_map(MT_BITVECTOR_VOXELMAP, "a");
	EXPECT_EQ(a->dimensions(), (Dim3ui{ 8, 6, 4 }));
	EXPECT_EQ(a->voxel_side_length(), 0.25f);
	EXPECT_THROW(gvl.initialize({ 9, 9, 9 }, 0.5f), VulkanError); // maps exist
	EXPECT_EQ(gvl.dimensions(), (Dim3ui{ 8, 6, 4 }));
	EXPECT_TRUE(gvl.del_map("a"));
	gvl.initialize({ 9, 9, 9 }, 0.5f);
	EXPECT_EQ(gvl.add_map(MT_PROBAB_VOXELMAP, "p")->dimensions(), (Dim3ui{ 9, 9, 9 }));
	EXPECT_EQ(gvl.get_map("p")->voxel_side_length(), 0.5f);
}

TEST_F(FacadeTest, Singleton)
{
	GpuVoxelsSharedPtr first = GpuVoxels::get_instance();
	ASSERT_TRUE(first);
	EXPECT_EQ(GpuVoxels::get_instance(), first);
	EXPECT_FALSE(first->is_initialized());
	EXPECT_THROW((void)first->context(), VulkanError);
	EXPECT_THROW((void)first->add_map(MT_BITVECTOR_VOXELMAP, "a"), VulkanError);
	EXPECT_THROW(first->initialize({ 0, 1, 1 }, 0.1f), VulkanError);
	EXPECT_THROW((void)first->context(), VulkanError); // a refused initialize creates no context

	first->initialize({ 16, 16, 16 }, 0.05f); // creates its own context
	EXPECT_TRUE(first->is_initialized());
	EXPECT_NE(&first->context(), &ctx());
	GpuVoxelsMapSharedPtr m = first->add_map(MT_BITVECTOR_VOXELMAP, "m");
	const Point3f p{ 0.125f, 0.125f, 0.125f };
	EXPECT_EQ(m->insert_point_cloud(std::span<const Point3f>(&p, 1), 5), 0u);
	EXPECT_EQ(m->as<BitVoxelMap>()->download_words()[(2 * 16 + 2) * 16 * 8 + 2 * 8], 1u << 5);

	GpuVoxelsSharedPtr second = GpuVoxels::get_instance();
	EXPECT_EQ(second, first);
	m.reset(); // a handle must not outlive the context of its instance
	first.reset();
	EXPECT_TRUE(second->is_initialized()); // still alive while someone holds it
	second.reset();
	const GpuVoxelsSharedPtr fresh = GpuVoxels::get_instance();
	EXPECT_FALSE(fresh->is_initialized()); // a new one after the last holder let go
}
