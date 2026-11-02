// Slice S7-4: GpuVoxelsMap::merge / merge_metric. The expectations come from host models of the lists and maps built from what the classes
// download (the inserts themselves are tested in S1 .. S4), not from the classes' own merge.
#include "vulkan_fixture.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <functional>
#include <limits>
#include <map>
#include <optional>
#include <random>
#include <string>
#include <vector>

#include <gpu_voxels/vulkan/GpuVoxels.h>

#include "../reference/bit_voxel_map_reference.h"
#include "facade_support.h"

using namespace gpu_voxels::vulkan;
using gvl_test::snapshot;
using gvl_test::VulkanTest;
namespace ref = gvl_reference;

namespace
{
	constexpr Dim3ui kDims{ 40, 30, 20 };
	constexpr float kSide = 0.1f;

	uint32_t linear(Dim3ui dims, Coord3ui c) { return static_cast<uint32_t>((uint64_t(c.z) * dims.y + c.y) * dims.x + c.x); }

	struct BitEntry
	{
		Coord3ui coordinates;
		std::array<uint32_t, 8> bits;
		bool operator==(const BitEntry& o) const { return coordinates.x == o.coordinates.x && coordinates.y == o.coordinates.y && coordinates.z == o.coordinates.z && bits == o.bits; }
	};
	struct CountEntry
	{
		Coord3ui coordinates;
		int8_t count;
		bool operator==(const CountEntry& o) const { return coordinates.x == o.coordinates.x && coordinates.y == o.coordinates.y && coordinates.z == o.coordinates.z && count == o.count; }
	};
	using BitModel = std::map<uint32_t, BitEntry>;
	using CountModel = std::map<uint32_t, CountEntry>;

	BitModel model_of(const BitVoxelList& list)
	{
		BitModel model;
		const auto ids = list.download_ids();
		const auto coordinates = list.download_coordinates();
		const auto bits = list.download_bit_vectors();
		for (std::size_t i = 0; i < ids.size(); ++i)
			model[ids[i]] = { coordinates[i], bits[i] };
		EXPECT_EQ(model.size(), ids.size()) << "ids are unique";
		return model;
	}

	CountModel model_of(const CountingVoxelList& list)
	{
		CountModel model;
		const auto ids = list.download_ids();
		const auto coordinates = list.download_coordinates();
		const auto counts = list.download_counts();
		for (std::size_t i = 0; i < ids.size(); ++i)
			model[ids[i]] = { coordinates[i], counts[i] };
		EXPECT_EQ(model.size(), ids.size()) << "ids are unique";
		return model;
	}

	std::vector<Coord3ui> random_coords(std::mt19937_64& rng, int n, Dim3ui dims)
	{
		std::vector<Coord3ui> coordinates;
		for (int i = 0; i < n; ++i)
			coordinates.push_back({ static_cast<uint32_t>(rng() % dims.x), static_cast<uint32_t>(rng() % dims.y), static_cast<uint32_t>(rng() % dims.z) });
		return coordinates;
	}

	void fill_bit_list(GpuVoxelsMap& list, std::mt19937_64& rng, Dim3ui dims, int n)
	{
		for (const uint8_t meaning : { uint8_t(3), uint8_t(5), uint8_t(40), uint8_t(200) })
			list.insert_coordinate_list(random_coords(rng, n, dims), meaning);
	}

	/// Counting lists: `rounds` rounds over the same coordinates give counts that wrap around int8 when two lists are added.
	void fill_counting_list(GpuVoxelsMap& list, std::mt19937_64& rng, Dim3ui dims, int n, int rounds)
	{
		const auto fixed = random_coords(rng, 6, dims);
		for (int i = 0; i < rounds; ++i)
			list.insert_coordinate_list(fixed, 0);
		list.insert_coordinate_list(random_coords(rng, n, dims), 0);
		list.insert_coordinate_list(random_coords(rng, n, dims), 0);
	}

	class MergeTest : public VulkanTest
	{
	protected:
		std::unique_ptr<GpuVoxelsMap> make(MapType type, Dim3ui dims = kDims, float side = kSide) { return std::make_unique<GpuVoxelsMap>(ctx(), type, dims, side); }
	};

	std::optional<Coord3ui> moved(Coord3ui c, Offset3i o, Dim3ui dims)
	{
		const int64_t x = int64_t(c.x) + o.x, y = int64_t(c.y) + o.y, z = int64_t(c.z) + o.z;
		if (x < 0 || y < 0 || z < 0 || x >= dims.x || y >= dims.y || z >= dims.z)
			return std::nullopt;
		return Coord3ui{ static_cast<uint32_t>(x), static_cast<uint32_t>(y), static_cast<uint32_t>(z) };
	}

	std::array<uint32_t, 8> one_bit(uint8_t bit)
	{
		std::array<uint32_t, 8> bits{};
		bits[bit / 32] = 1u << (bit % 32);
		return bits;
	}

	void expect_error(const std::function<void()>& call, const std::string& text)
	{
		try
		{
			call();
			ADD_FAILURE() << "no exception, expected one with '" << text << "'";
		}
		catch (const VulkanError& e)
		{
			EXPECT_NE(std::string(e.what()).find(text), std::string::npos) << "message was: " << e.what();
		}
	}

	const Offset3i kOffsets[] = { { 0, 0, 0 }, { 2, -1, 3 }, { -100, 0, 0 }, { 30, 0, 0 }, { -3, 5, -2 } };
}

TEST_F(MergeTest, ListsFromListsMatchTheHostModel)
{
	std::mt19937_64 rng(401);
	for (const MapType target_type : { MT_BITVECTOR_VOXELLIST, MT_COUNTING_VOXELLIST })
		for (const MapType source_type : { MT_BITVECTOR_VOXELLIST, MT_COUNTING_VOXELLIST })
			for (const Offset3i offset : kOffsets)
				for (const std::optional<uint8_t> meaning : { std::optional<uint8_t>(), std::optional<uint8_t>(9), std::optional<uint8_t>(200) })
				{
					if (target_type == MT_COUNTING_VOXELLIST && meaning)
						continue; // refused, see UnsupportedCombinationsAndArguments
					const std::string label = std::string(map_type_name(target_type)) + " <- " + map_type_name(source_type) + " offset " + std::to_string(offset.x) + "," +
						std::to_string(offset.y) + "," + std::to_string(offset.z) + (meaning ? " meaning " + std::to_string(*meaning) : std::string());
					auto target = make(target_type);
					auto source = make(source_type);
					if (target_type == MT_BITVECTOR_VOXELLIST) fill_bit_list(*target, rng, kDims, 300); else fill_counting_list(*target, rng, kDims, 300, 100);
					if (source_type == MT_BITVECTOR_VOXELLIST) fill_bit_list(*source, rng, kDims, 300); else fill_counting_list(*source, rng, kDims, 300, 100);
					// Make sure the two lists share some voxels (OR / sum of existing entries).
					const auto shared = random_coords(rng, 40, kDims);
					target->insert_coordinate_list(shared, 7);
					source->insert_coordinate_list(shared, 11);

					BitModel bit_expected;
					CountModel count_expected;
					BitModel source_bits;
					CountModel source_counts;
					if (target_type == MT_BITVECTOR_VOXELLIST) bit_expected = model_of(*target->as<BitVoxelList>()); else count_expected = model_of(*target->as<CountingVoxelList>());
					if (source_type == MT_BITVECTOR_VOXELLIST) source_bits = model_of(*source->as<BitVoxelList>()); else source_counts = model_of(*source->as<CountingVoxelList>());
					const auto source_before = snapshot(*source);

					uint32_t dropped = 0;
					auto apply = [&](Coord3ui c, const std::array<uint32_t, 8>& bits, int8_t count)
					{
						const auto to = moved(c, offset, kDims);
						if (!to)
						{
							++dropped;
							return;
						}
						const uint32_t id = linear(kDims, *to);
						if (target_type == MT_BITVECTOR_VOXELLIST)
						{
							auto payload = bits;
							if (meaning)
								payload = one_bit(*meaning);
							else if (source_type == MT_COUNTING_VOXELLIST)
								payload = {};
							auto& entry = bit_expected.try_emplace(id, BitEntry{ *to, {} }).first->second;
							for (int w = 0; w < 8; ++w)
								entry.bits[w] |= payload[w];
						}
						else
						{
							auto& entry = count_expected.try_emplace(id, CountEntry{ *to, 0 }).first->second;
							entry.count = static_cast<int8_t>(entry.count + (source_type == MT_COUNTING_VOXELLIST ? count : 0));
						}
					};
					if (source_type == MT_BITVECTOR_VOXELLIST)
						for (const auto& [id, entry] : source_bits)
							apply(entry.coordinates, entry.bits, 0);
					else
						for (const auto& [id, entry] : source_counts)
							apply(entry.coordinates, {}, entry.count);

					const uint32_t returned = target->merge(*source, offset, meaning);
					EXPECT_EQ(returned, dropped) << label;
					if (target_type == MT_BITVECTOR_VOXELLIST)
						EXPECT_TRUE(model_of(*target->as<BitVoxelList>()) == bit_expected) << label;
					else
						EXPECT_TRUE(model_of(*target->as<CountingVoxelList>()) == count_expected) << label;
					EXPECT_EQ(snapshot(*source), source_before) << label << ": the source is not changed";
					if (offset.x == -100)
						EXPECT_EQ(returned, source_type == MT_BITVECTOR_VOXELLIST ? source_bits.size() : source_counts.size()) << "everything moved outside";
				}
}

TEST_F(MergeTest, CountsAreAddedAndWrapLikeInt8)
{
	auto target = make(MT_COUNTING_VOXELLIST);
	auto source = make(MT_COUNTING_VOXELLIST);
	const std::vector<Coord3ui> voxel{ { 4, 5, 6 } };
	for (int i = 0; i < 100; ++i)
	{
		target->insert_coordinate_list(voxel, 0);
		source->insert_coordinate_list(voxel, 0);
	}
	ASSERT_EQ(target->as<CountingVoxelList>()->download_counts(), std::vector<int8_t>{ 100 });
	target->merge(*source);
	EXPECT_EQ(target->as<CountingVoxelList>()->download_counts(), std::vector<int8_t>{ static_cast<int8_t>(200) }) << "100 + 100 wraps to -56";
	EXPECT_EQ(source->as<CountingVoxelList>()->download_counts(), std::vector<int8_t>{ 100 });
}

TEST_F(MergeTest, BitListFromBitMap)
{
	std::mt19937_64 rng(402);
	for (const std::optional<uint8_t> meaning : { std::optional<uint8_t>(), std::optional<uint8_t>(17) })
	{
		auto map = make(MT_BITVECTOR_VOXELMAP);
		auto list = make(MT_BITVECTOR_VOXELLIST);
		for (const uint8_t m : { uint8_t(3), uint8_t(5), uint8_t(40) })
			map->insert_coordinate_list(random_coords(rng, 500, kDims), m);
		map->insert_coordinate_list(random_coords(rng, 50, kDims), eBVM_FREE); // free only: not occupied
		fill_bit_list(*list, rng, kDims, 300);
		list->insert_coordinate_list(random_coords(rng, 100, kDims), 200);

		BitModel expected = model_of(*list->as<BitVoxelList>());
		const auto words = map->as<BitVoxelMap>()->download_words();
		uint32_t occupied = 0;
		for (uint32_t z = 0; z < kDims.z; ++z)
			for (uint32_t y = 0; y < kDims.y; ++y)
				for (uint32_t x = 0; x < kDims.x; ++x)
				{
					const uint32_t id = linear(kDims, { x, y, z });
					std::array<uint32_t, 8> bits;
					std::copy_n(words.begin() + static_cast<std::ptrdiff_t>(id) * 8, 8, bits.begin());
					std::array<uint32_t, 8> without_free = bits;
					without_free[0] &= ~1u;
					if (std::all_of(without_free.begin(), without_free.end(), [](uint32_t w) { return w == 0; }))
						continue;
					++occupied;
					auto& entry = expected.try_emplace(id, BitEntry{ { x, y, z }, {} }).first->second;
					const auto payload = meaning ? one_bit(*meaning) : bits;
					for (int w = 0; w < 8; ++w)
						entry.bits[w] |= payload[w];
				}
		ASSERT_GT(occupied, 100u);

		EXPECT_EQ(list->merge(*map, {}, meaning), 0u);
		EXPECT_TRUE(model_of(*list->as<BitVoxelList>()) == expected) << (meaning ? "meaning" : "payload copied");
		expect_error([&] { (void)list->merge(*map, { 1, 0, 0 }, meaning); }, "an offset is not supported");
	}
}

TEST_F(MergeTest, MapsFromLists)
{
	std::mt19937_64 rng(403);
	for (const MapType source_type : { MT_BITVECTOR_VOXELLIST, MT_COUNTING_VOXELLIST })
		for (const MapType target_type : { MT_BITVECTOR_VOXELMAP, MT_PROBAB_VOXELMAP, MT_DISTANCE_VOXELMAP })
		{
			const Dim3ui map_dims{ 20, 30, 20 }; // smaller than the reference map of the list: the voxels with x >= 20 lie outside
			auto source = make(source_type, kDims);
			if (source_type == MT_BITVECTOR_VOXELLIST) fill_bit_list(*source, rng, kDims, 400); else fill_counting_list(*source, rng, kDims, 400, 3);
			const std::vector<Coord3ui> listed = source_type == MT_BITVECTOR_VOXELLIST ? source->as<BitVoxelList>()->download_coordinates() : source->as<CountingVoxelList>()->download_coordinates();
			uint32_t outside = 0;
			for (const auto& c : listed)
				outside += c.x >= map_dims.x ? 1 : 0;
			ASSERT_GT(outside, 10u);

			auto target = make(target_type, map_dims);
			const auto previous = random_coords(rng, 200, map_dims);
			target->insert_coordinate_list(previous, target_type == MT_BITVECTOR_VOXELMAP ? uint8_t(5) : uint8_t(eBVM_OCCUPIED));
			auto twin = make(target_type, map_dims);
			twin->insert_coordinate_list(previous, target_type == MT_BITVECTOR_VOXELMAP ? uint8_t(5) : uint8_t(eBVM_OCCUPIED));
			EXPECT_EQ(twin->insert_coordinate_list(listed, eBVM_OCCUPIED), outside);

			ref::BitVoxelMapReference reference(map_dims.x, map_dims.y, map_dims.z, kSide);
			reference.insert_coordinates(previous, 5);

			EXPECT_EQ(target->merge(*source), outside) << map_type_name(target_type) << " <- " << map_type_name(source_type);
			EXPECT_EQ(snapshot(*target), snapshot(*twin)) << "the coordinates of the list are inserted as occupied";
			if (target_type == MT_BITVECTOR_VOXELMAP)
			{
				EXPECT_EQ(reference.insert_coordinates(listed, eBVM_OCCUPIED), outside);
				EXPECT_TRUE(target->as<BitVoxelMap>()->download_words() == reference.to_words(0, map_dims.voxel_count()));
			}
		}
}

TEST_F(MergeTest, UnsupportedCombinationsAndArguments)
{
	std::mt19937_64 rng(404);
	const MapType types[] = { MT_BITVECTOR_VOXELMAP, MT_PROBAB_VOXELMAP, MT_DISTANCE_VOXELMAP, MT_BITVECTOR_VOXELLIST, MT_COUNTING_VOXELLIST, MT_BITVECTOR_OCTREE,
		MT_PROBAB_OCTREE };
	auto supported = [](MapType target, MapType source)
	{
		const bool list = source == MT_BITVECTOR_VOXELLIST || source == MT_COUNTING_VOXELLIST;
		switch (target)
		{
		case MT_BITVECTOR_VOXELMAP:
		case MT_PROBAB_VOXELMAP:
		case MT_DISTANCE_VOXELMAP:
		case MT_COUNTING_VOXELLIST: return list;
		case MT_BITVECTOR_VOXELLIST: return list || source == MT_BITVECTOR_VOXELMAP;
		default: return false;
		}
	};
	for (const MapType target_type : types)
		for (const MapType source_type : types)
		{
			auto target = make(target_type);
			auto source = make(source_type);
			const std::vector<Coord3ui> some{ { 1, 2, 3 }, { 4, 5, 6 } };
			if (source_type == MT_PROBAB_VOXELMAP || source_type == MT_BITVECTOR_OCTREE || source_type == MT_PROBAB_OCTREE || source_type == MT_DISTANCE_VOXELMAP)
				source->insert_coordinate_list(some, eBVM_OCCUPIED);
			else
				source->insert_coordinate_list(some, 5);
			const auto before = snapshot(*target);
			if (supported(target_type, source_type))
				EXPECT_NO_THROW((void)target->merge(*source)) << map_type_name(target_type) << " <- " << map_type_name(source_type);
			else
			{
				expect_error([&] { (void)target->merge(*source); }, std::string("merge of ") + map_type_name(source_type));
				expect_error([&] { (void)target->merge(*source); }, std::string("is not supported by the map type ") + map_type_name(target_type));
				EXPECT_EQ(snapshot(*target), before) << map_type_name(target_type) << " is unchanged";
			}
		}

	// Arguments.
	auto bit_list = make(MT_BITVECTOR_VOXELLIST);
	auto other_bit_list = make(MT_BITVECTOR_VOXELLIST);
	auto counting_list = make(MT_COUNTING_VOXELLIST);
	auto bit_map = make(MT_BITVECTOR_VOXELMAP);
	fill_bit_list(*other_bit_list, rng, kDims, 50);
	expect_error([&] { (void)bit_list->merge(*bit_list); }, "a map cannot be merged into itself");
	expect_error([&] { (void)counting_list->merge(*counting_list); }, "a map cannot be merged into itself");
	expect_error([&] { (void)bit_map->merge(*other_bit_list, { 0, 0, 1 }); }, "merge into a map: an offset is not supported");
	expect_error([&] { (void)bit_map->merge(*other_bit_list, {}, 5); }, "merge into a map: a new meaning is not supported");
	expect_error([&] { (void)counting_list->merge(*other_bit_list, {}, 5); }, "merge into a counting list: a new meaning is not supported");
	EXPECT_TRUE(snapshot(*bit_map) == snapshot(*make(MT_BITVECTOR_VOXELMAP))) << "refused merges change nothing";
	EXPECT_TRUE(snapshot(*counting_list) == snapshot(*make(MT_COUNTING_VOXELLIST)));

	// Another voxel side length or other reference dimensions of the list.
	auto other_side = make(MT_BITVECTOR_VOXELLIST, kDims, 0.2f);
	auto other_dims = make(MT_BITVECTOR_VOXELLIST, { 41, 30, 20 }, kSide);
	other_side->insert_coordinate_list(std::vector<Coord3ui>{ { 1, 1, 1 } }, 5);
	other_dims->insert_coordinate_list(std::vector<Coord3ui>{ { 1, 1, 1 } }, 5);
	expect_error([&] { (void)bit_list->merge(*other_side); }, "merge: the voxel side lengths differ");
	expect_error([&] { (void)bit_map->merge(*other_side); }, "merge: the voxel side lengths differ");
	expect_error([&] { (void)counting_list->merge(*other_side); }, "merge: the voxel side lengths differ");
	EXPECT_NO_THROW((void)bit_list->merge(*bit_map)); // a map of the same size is fine for a list
	expect_error([&] { (void)bit_list->merge(*other_dims); }, "reference dimensions differ");
	expect_error([&] { (void)counting_list->merge(*other_dims); }, "reference dimensions differ");
	EXPECT_TRUE(bit_list->as<BitVoxelList>()->empty()) << "nothing merged";
	// A list of other dimensions is fine for a map (only its coordinates count).
	EXPECT_NO_THROW((void)bit_map->merge(*other_dims));
}

TEST_F(MergeTest, MetricOffsets)
{
	std::mt19937_64 rng(405);
	auto source = make(MT_BITVECTOR_VOXELLIST);
	fill_bit_list(*source, rng, kDims, 100);
	struct Case { Point3f metric; Offset3i voxels; };
	const Case cases[] = {
		{ { 0.25f, 0.0f, 0.0f }, { 2, 0, 0 } },
		{ { -0.05f, 0.31f, 0.0f }, { -1, 3, 0 } }, // lists: floor
		{ { -0.25f, -0.01f, 0.1f }, { -3, -1, 1 } },
		{ { 0.0f, 0.0f, 0.0f }, { 0, 0, 0 } },
	};
	for (const auto& c : cases)
	{
		auto by_metric = make(MT_BITVECTOR_VOXELLIST);
		auto by_voxels = make(MT_BITVECTOR_VOXELLIST);
		const uint32_t a = by_metric->merge_metric(*source, c.metric);
		const uint32_t b = by_voxels->merge(*source, c.voxels);
		EXPECT_EQ(a, b);
		EXPECT_EQ(snapshot(*by_metric), snapshot(*by_voxels)) << c.metric.x << " " << c.metric.y << " " << c.metric.z;
	}
	auto counting_source = make(MT_COUNTING_VOXELLIST);
	fill_counting_list(*counting_source, rng, kDims, 100, 2);
	{
		auto by_metric = make(MT_COUNTING_VOXELLIST);
		auto by_voxels = make(MT_COUNTING_VOXELLIST);
		by_metric->merge_metric(*counting_source, { -0.05f, 0.0f, 0.0f });
		by_voxels->merge(*counting_source, { -1, 0, 0 });
		EXPECT_EQ(snapshot(*by_metric), snapshot(*by_voxels)) << "counting lists use the floor as well";
	}

	// Maps truncate toward zero (CUDA casts): -0.05 m is no offset at all, a list would move by one voxel.
	auto map = make(MT_BITVECTOR_VOXELMAP);
	auto twin = make(MT_BITVECTOR_VOXELMAP);
	EXPECT_NO_THROW((void)map->merge_metric(*source, { -0.05f, 0.09f, 0.0f }));
	(void)twin->merge(*source);
	EXPECT_EQ(snapshot(*map), snapshot(*twin));
	expect_error([&] { (void)map->merge_metric(*source, { -0.15f, 0.0f, 0.0f }); }, "an offset is not supported");

	// Exactly 2^31 voxels does not fit an int32 (a power of two as side length makes the division exact).
	auto half_side_list = make(MT_BITVECTOR_VOXELLIST, kDims, 0.5f);
	auto half_side_source = make(MT_BITVECTOR_VOXELLIST, kDims, 0.5f);
	expect_error([&] { (void)half_side_list->merge_metric(*half_side_source, { 2147483648.0f * 0.5f, 0.0f, 0.0f }); }, "beyond 2^31 voxels");
	expect_error([&] { (void)half_side_list->merge_metric(*half_side_source, { 0.0f, -2147483648.0f * 0.5f, 0.0f }); }, "beyond 2^31 voxels");
	EXPECT_NO_THROW((void)half_side_list->merge_metric(*half_side_source, { 1073741824.0f * 0.5f, 0.0f, 0.0f }));

	const float inf = std::numeric_limits<float>::infinity();
	expect_error([&] { (void)source->merge_metric(*make(MT_BITVECTOR_VOXELLIST), { inf, 0.0f, 0.0f }); }, "the offset is not finite");
	expect_error([&] { (void)source->merge_metric(*make(MT_BITVECTOR_VOXELLIST), { 0.0f, std::numeric_limits<float>::quiet_NaN(), 0.0f }); }, "the offset is not finite");
	expect_error([&] { (void)source->merge_metric(*make(MT_BITVECTOR_VOXELLIST), { 0.0f, 0.0f, 3.0e9f * kSide }); }, "the offset is not finite");
	expect_error([&] { (void)source->merge_metric(*make(MT_BITVECTOR_VOXELLIST), { -3.0e9f * kSide, 0.0f, 0.0f }); }, "beyond 2^31 voxels");
}

TEST_F(MergeTest, TheListClassesCheckTheirArgumentsThemselves)
{
	BitVoxelList a(ctx(), kDims, kSide);
	BitVoxelList other_side(ctx(), kDims, 0.2f);
	BitVoxelList other_dims(ctx(), { 41, 30, 20 }, kSide);
	CountingVoxelList n(ctx(), kDims, kSide);
	BitVoxelMap other_map(ctx(), { 41, 30, 20 }, kSide);
	expect_error([&] { (void)a.merge(a); }, "VoxelList: a list cannot be merged into itself");
	expect_error([&] { (void)n.merge(n); }, "VoxelList: a list cannot be merged into itself");
	expect_error([&] { (void)a.merge(other_side); }, "VoxelList: the voxel side lengths differ");
	expect_error([&] { (void)n.merge(other_side); }, "VoxelList: the voxel side lengths differ");
	expect_error([&] { (void)a.merge(other_dims); }, "VoxelList: reference dimensions differ");
	expect_error([&] { (void)n.merge(other_dims); }, "VoxelList: reference dimensions differ");
	expect_error([&] { (void)a.merge(other_map); }, "the reference dimensions do not match the map dimensions");
}

TEST_F(MergeTest, AssignEntries)
{
	BitVoxelList list(ctx(), { 4, 1, 1 }, 0.5f);
	const std::vector<uint32_t> ids{ 1, 3 };
	const std::vector<Coord3ui> coordinates{ { 1, 0, 0 }, { 3, 0, 0 } };
	std::vector<uint32_t> bits(16, 0);
	bits[0] = 0xAu;
	bits[8 + 7] = 0x80000000u;
	list.assign_entries(ids, coordinates, bits);
	EXPECT_EQ(list.download_ids(), ids);
	EXPECT_EQ(list.download_voxel_words(), bits);
	const auto good = snapshot(list);
	EXPECT_FALSE(VoxelListBase::entries_problem({ 4, 1, 1 }, VoxelListBase::Kind::Bit, ids, coordinates, bits).has_value());

	// Refused entries leave the list as it was.
	const std::vector<uint32_t> short_bits(15, 0);
	expect_error([&] { list.assign_entries(ids, coordinates, short_bits); }, "do not have the same number of entries");
	expect_error([&] { list.assign_entries(std::vector<uint32_t>{ 1 }, coordinates, bits); }, "do not have the same number of entries");
	expect_error([&] { list.assign_entries(ids, std::vector<Coord3ui>{ { 1, 0, 0 } }, bits); }, "do not have the same number of entries");
	expect_error([&] { list.assign_entries(std::vector<uint32_t>{ 1, 4 }, std::vector<Coord3ui>{ { 1, 0, 0 }, { 4, 0, 0 } }, bits); }, "entry 1 lies outside the reference map");
	expect_error([&] { list.assign_entries(std::vector<uint32_t>{ 1, 2 }, coordinates, bits); }, "the id of entry 1 is not the index of its coordinates");
	expect_error([&] { list.assign_entries(std::vector<uint32_t>{ 3, 1 }, std::vector<Coord3ui>{ { 3, 0, 0 }, { 1, 0, 0 } }, bits); },
		"the ids are not strictly ascending at entry 1");
	EXPECT_EQ(snapshot(list), good);
	// All entries can be replaced by none.
	list.assign_entries({}, {}, {});
	EXPECT_TRUE(list.empty());

	CountingVoxelList counting(ctx(), { 4, 1, 1 }, 0.5f);
	counting.assign_entries(ids, coordinates, std::vector<uint32_t>{ 1u, 0xFFFFFF80u });
	EXPECT_EQ(counting.download_counts(), (std::vector<int8_t>{ 1, -128 }));
	const auto counting_good = snapshot(counting);
	expect_error([&] { counting.assign_entries(ids, coordinates, std::vector<uint32_t>{ 1u, 0x80u }); }, "the count of entry 1 is not a sign extended int8");
	expect_error([&] { counting.assign_entries(ids, coordinates, std::vector<uint32_t>{ 0x100u, 1u }); }, "the count of entry 0 is not a sign extended int8");
	expect_error([&] { counting.assign_entries(ids, coordinates, std::vector<uint32_t>{ 1u }); }, "do not have the same number of entries");
	EXPECT_EQ(snapshot(counting), counting_good);
}
