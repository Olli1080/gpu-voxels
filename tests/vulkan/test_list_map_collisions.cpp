// Slice S7-5: collisions of a BitVoxelList with a bit map / probabilistic map (collide_with, collide_with_types, collide_with_type_mask), with
// offsets. The expectations come from the host model in list_collision_model.h (the rules of the CUDA kernels applied to the downloaded states).
#include "vulkan_fixture.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <functional>
#include <initializer_list>
#include <limits>
#include <random>
#include <string>
#include <vector>

#include <gpu_voxels/vulkan/BitVoxelMap.h>
#include <gpu_voxels/vulkan/ProbVoxelMap.h>
#include <gpu_voxels/vulkan/VoxelList.h>

#include "list_collision_model.h"

using namespace gpu_voxels::vulkan;
using namespace gvl_test::collision_model;
using gvl_test::VulkanTest;

namespace
{
	constexpr Dim3ui kDims{ 14, 13, 12 };
	constexpr float kSide = 0.1f;

	class ListMapCollisionTest : public VulkanTest
	{
	protected:
		/// A list with entries of many meanings, among them FREE-only entries (meaning 0) and swept volume meanings.
		BitVoxelList make_list(uint32_t seed)
		{
			std::mt19937_64 rng(seed);
			BitVoxelList list(ctx(), kDims, kSide);
			for (const uint8_t meaning : { uint8_t(0), uint8_t(1), uint8_t(34), uint8_t(63), uint8_t(102), uint8_t(200) })
				list.insert_coordinate_list(random_coords(rng, 130), meaning);
			return list;
		}

		static std::vector<Coord3ui> random_coords(std::mt19937_64& rng, int n)
		{
			std::vector<Coord3ui> coordinates;
			for (int i = 0; i < n; ++i)
				coordinates.push_back({ static_cast<uint32_t>(rng() % kDims.x), static_cast<uint32_t>(rng() % kDims.y), static_cast<uint32_t>(rng() % kDims.z) });
			return coordinates;
		}

		BitVoxelMap make_bit_map()
		{
			std::mt19937_64 rng(77);
			BitVoxelMap map(ctx(), kDims, kSide);
			map.insert_coordinate_list(random_coords(rng, 700), 1);  // occupied
			map.insert_coordinate_list(random_coords(rng, 300), 0);  // FREE only: not occupied
			map.insert_coordinate_list(random_coords(rng, 100), 99);
			return map;
		}

		ProbVoxelMap make_prob_map()
		{
			std::mt19937_64 rng(78);
			ProbVoxelMap map(ctx(), kDims, kSide);
			map.insert_coordinate_list(random_coords(rng, 500), 1);   // 72 (one update) or more
			map.insert_coordinate_list(random_coords(rng, 500), 1);   // the second update saturates some of them at 127
			map.insert_coordinate_list(random_coords(rng, 400), 0);   // free updates: negative values
			return map;
		}

		static void expect_error(const std::function<void()>& call, const std::string& text)
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

		/// Runs one call on a fresh list and compares the result and the marked list with the model.
		template<class Map, class Call>
		void check(const Map& map, Mode mode, Offset3i offset, float threshold, const BitVector256& mask, const Call& call, const std::string& what)
		{
			BitVoxelList list = make_list(5);
			const Expected expected = model(entries_of(list), view_of(map), mode, offset, threshold, mask);
			const auto [count, types] = call(list);
			EXPECT_EQ(count, expected.count) << what;
			if (mode == Mode::Types)
				EXPECT_EQ(types, expected.types) << what;
			const auto after = list.download_bit_vectors();
			ASSERT_EQ(after.size(), expected.bits_after.size());
			for (std::size_t i = 0; i < after.size(); ++i)
				ASSERT_EQ(after[i], expected.bits_after[i]) << what << ", entry " << i;
		}

		static inline const Offset3i kOffsets[] = { { 0, 0, 0 }, { 1, 0, 0 }, { -1, 0, 0 }, { 0, 2, 0 }, { 0, 0, -3 }, { -2, 3, 1 }, { 13, 0, 0 },
			{ -13, 0, 0 }, { 0, 12, 0 }, { 0, 0, 11 }, { 0, 0, -12 }, { 14, 0, 0 }, { 0, -13, 0 }, { 0, 0, 12 }, { 5, -4, 3 } };
	};

	/// A list voxel is only marked when it collides; the call result is (count, types).
	using Result = std::pair<uint32_t, BitVector256>;
}

TEST_F(ListMapCollisionTest, ModelHasSomethingToCheck)
{
	// Guard against a model that is always zero: the default case must have many, but not all, collisions.
	BitVoxelList list = make_list(5);
	const auto entries = entries_of(list);
	ASSERT_GT(entries.size(), 300u);
	for (const bool probabilistic : { false, true })
	{
		const MapView view = probabilistic ? view_of(make_prob_map()) : view_of(make_bit_map());
		const auto e = model(entries, view, Mode::Default, {}, 1.0f, {});
		EXPECT_GT(e.count, 20u) << probabilistic;
		EXPECT_LT(e.count, entries.size()) << probabilistic;
	}
	const auto types = model(entries, view_of(make_bit_map()), Mode::Types, {}, 1.0f, {});
	EXPECT_GT(types.count, model(entries, view_of(make_bit_map()), Mode::Default, {}, 1.0f, {}).count) << "FREE-only entries collide in the types form";
}

TEST_F(ListMapCollisionTest, CollideWithMatchesTheDefaultColliderForEveryOffset)
{
	const BitVoxelMap bit_map = make_bit_map();
	const ProbVoxelMap prob_map = make_prob_map();
	for (const Offset3i offset : kOffsets)
	{
		const std::string what = "offset " + std::to_string(offset.x) + "," + std::to_string(offset.y) + "," + std::to_string(offset.z);
		check(bit_map, Mode::Default, offset, 1.0f, {}, [&](BitVoxelList& l) { return Result{ l.collide_with(bit_map, offset), {} }; }, "bit map, " + what);
		for (const float threshold : { 1.0f, 0.5f, 0.1f, 0.0f, 0.9f })
			check(prob_map, Mode::Default, offset, threshold, {},
				[&](BitVoxelList& l) { return Result{ l.collide_with(prob_map, threshold, offset), {} }; }, "prob map, threshold " + std::to_string(threshold) + ", " + what);
	}
}

TEST_F(ListMapCollisionTest, CollideWithTypesMatchesTheTypesKernelForEveryOffset)
{
	const BitVoxelMap bit_map = make_bit_map();
	const ProbVoxelMap prob_map = make_prob_map();
	for (const Offset3i offset : kOffsets)
	{
		const std::string what = "offset " + std::to_string(offset.x) + "," + std::to_string(offset.y) + "," + std::to_string(offset.z);
		check(bit_map, Mode::Types, offset, 1.0f, {}, [&](BitVoxelList& l) { const auto r = l.collide_with_types(bit_map, 1.0f, offset); return Result{ r.count, r.types }; }, "bit map, " + what);
		// The threshold is raw (float(occupancy) >= threshold), -128 includes the unknown voxels, NaN excludes all.
		for (const float threshold : { 1.0f, 0.5f, 0.0f, 72.0f, 73.0f, 127.0f, 128.0f, -10.0f, -11.0f, -128.0f, -1000.0f, std::numeric_limits<float>::quiet_NaN() })
			check(prob_map, Mode::Types, offset, threshold, {},
				[&](BitVoxelList& l) { const auto r = l.collide_with_types(prob_map, threshold, offset); return Result{ r.count, r.types }; },
				"prob map, threshold " + std::to_string(threshold) + ", " + what);
	}
}

TEST_F(ListMapCollisionTest, CollideWithTypeMaskMatchesTheMaskKernelForEveryOffset)
{
	const BitVoxelMap bit_map = make_bit_map();
	const ProbVoxelMap prob_map = make_prob_map();
	const std::vector<BitVector256> masks = { {}, bits_of({ 34 }), bits_of({ 63 }), bits_of({ 102 }), bits_of({ 0 }), bits_of({ 1 }), bits_of({ 2 }),
		bits_of({ 3, 200 }), bits_of({ 255 }), bits_of({ 31, 32, 63, 64 }), bits_of({ 34, 63, 102 }), BitVector256{ ~0u, ~0u, ~0u, ~0u, ~0u, ~0u, ~0u, ~0u } };
	for (std::size_t m = 0; m < masks.size(); ++m)
	{
		for (const Offset3i offset : kOffsets)
		{
			const std::string what = "mask " + std::to_string(m) + ", offset " + std::to_string(offset.x) + "," + std::to_string(offset.y) + "," + std::to_string(offset.z);
			if (m < 4 || offset == Offset3i{} || offset == Offset3i{ -2, 3, 1 })
			{
				check(bit_map, Mode::Mask, offset, 1.0f, masks[m], [&](BitVoxelList& l) { return Result{ l.collide_with_type_mask(bit_map, masks[m], 1.0f, offset), {} }; }, "bit map, " + what);
				for (const float threshold : { 1.0f, 72.0f, -128.0f })
					check(prob_map, Mode::Mask, offset, threshold, masks[m],
						[&](BitVoxelList& l) { return Result{ l.collide_with_type_mask(prob_map, masks[m], threshold, offset), {} }; }, "prob map, threshold " + std::to_string(threshold) + ", " + what);
			}
		}
	}
	// The sandbox examples call it without a mask first: nothing collides and nothing is marked.
	BitVoxelList list = make_list(5);
	const auto before = list.download_bit_vectors();
	EXPECT_EQ(list.collide_with_type_mask(prob_map, {}, 1.0f), 0u);
	EXPECT_EQ(list.download_bit_vectors(), before);
}

TEST_F(ListMapCollisionTest, ListVoxelsWithOnlyTheFreeBitDoNotCollideByDefaultButDoInTheTypesForms)
{
	// Entries: A at (2,3,4) with only FREE, B at (5,6,7) with OCCUPIED. The map is occupied at both voxels.
	BitVoxelList list(ctx(), kDims, kSide);
	list.insert_coordinate_list(std::vector<Coord3ui>{ { 2, 3, 4 } }, eBVM_FREE);
	list.insert_coordinate_list(std::vector<Coord3ui>{ { 5, 6, 7 } }, eBVM_OCCUPIED);
	BitVoxelMap bit_map(ctx(), kDims, kSide);
	bit_map.insert_coordinate_list(std::vector<Coord3ui>{ { 2, 3, 4 }, { 5, 6, 7 } }, 50);
	ProbVoxelMap prob_map(ctx(), kDims, kSide);
	for (int i = 0; i < 2; ++i)
		prob_map.insert_coordinate_list(std::vector<Coord3ui>{ { 2, 3, 4 }, { 5, 6, 7 } }, eBVM_OCCUPIED);

	EXPECT_EQ(list.collide_with(bit_map), 1u) << "only B: the list voxel is inspected too";
	EXPECT_EQ(list.download_bit_vectors()[0], bits_of({ 0 })) << "A is not marked";
	EXPECT_EQ(list.download_bit_vectors()[1], bits_of({ 1, 2 })) << "B is marked with eBVM_COLLISION";
	EXPECT_EQ(list.collide_with(prob_map, 1.0f), 1u);

	BitVoxelList fresh(ctx(), kDims, kSide);
	fresh.insert_coordinate_list(std::vector<Coord3ui>{ { 2, 3, 4 } }, eBVM_FREE);
	fresh.insert_coordinate_list(std::vector<Coord3ui>{ { 5, 6, 7 } }, eBVM_OCCUPIED);
	const auto types = fresh.collide_with_types(bit_map);
	EXPECT_EQ(types.count, 2u) << "the types form does not look at the list voxel";
	EXPECT_EQ(types.types, bits_of({ 0, 1 }));
	EXPECT_EQ(fresh.collide_with_type_mask(prob_map, bits_of({ 0 }), 1.0f), 1u) << "the mask may select the FREE bit";
	EXPECT_EQ(fresh.collide_with_type_mask(prob_map, bits_of({ 0, 1 }), 1.0f), 2u);
	EXPECT_EQ(fresh.collide_with_type_mask(prob_map, bits_of({ 1 }), 1.0f), 1u);
}

TEST_F(ListMapCollisionTest, AnOffsetMovesInThreeDimensionsAndDoesNotWrapAcrossRows)
{
	// CUDA adds the offset to the linear index: (13, 4, 5) + (1, 0, 0) would meet (0, 5, 5). Here the entry leaves the map.
	BitVoxelList list(ctx(), kDims, kSide);
	list.insert_coordinate_list(std::vector<Coord3ui>{ { 13, 4, 5 } }, eBVM_OCCUPIED);
	BitVoxelMap bit_map(ctx(), kDims, kSide);
	bit_map.insert_coordinate_list(std::vector<Coord3ui>{ { 0, 5, 5 }, { 12, 4, 5 } }, eBVM_OCCUPIED);
	EXPECT_EQ(list.collide_with(bit_map, { 1, 0, 0 }), 0u);
	EXPECT_EQ(list.collide_with(bit_map, { -1, 0, 0 }), 1u) << "(13, 4, 5) - 1 = (12, 4, 5)";
	EXPECT_EQ(list.collide_with_types(bit_map, 1.0f, { 1, 0, 0 }).count, 0u);
	EXPECT_EQ(list.collide_with_type_mask(bit_map, bits_of({ 1 }), 1.0f, { 1, 0, 0 }), 0u);
	// The same at the top: y + 1 at the last row would continue in the next slice.
	BitVoxelList top(ctx(), kDims, kSide);
	top.insert_coordinate_list(std::vector<Coord3ui>{ { 3, 12, 5 } }, eBVM_OCCUPIED);
	BitVoxelMap next_slice(ctx(), kDims, kSide);
	next_slice.insert_coordinate_list(std::vector<Coord3ui>{ { 3, 0, 6 } }, eBVM_OCCUPIED);
	EXPECT_EQ(top.collide_with(next_slice, { 0, 1, 0 }), 0u);
}

TEST_F(ListMapCollisionTest, ArgumentsAreChecked)
{
	BitVoxelList list = make_list(5);
	BitVoxelMap other_dims(ctx(), { 15, 13, 12 }, kSide);
	ProbVoxelMap other_prob(ctx(), { 14, 13, 13 }, kSide);
	expect_error([&] { (void)list.collide_with(other_dims); }, "the reference dimensions do not match the map dimensions");
	expect_error([&] { (void)list.collide_with(other_prob); }, "the reference dimensions do not match the map dimensions");
	expect_error([&] { (void)list.collide_with_types(other_dims); }, "the reference dimensions do not match the map dimensions");
	expect_error([&] { (void)list.collide_with_types(other_prob); }, "the reference dimensions do not match the map dimensions");
	expect_error([&] { (void)list.collide_with_type_mask(other_dims, bits_of({ 1 })); }, "the reference dimensions do not match the map dimensions");
	expect_error([&] { (void)list.collide_with_type_mask(other_prob, bits_of({ 1 })); }, "the reference dimensions do not match the map dimensions");
	// A refused call changes nothing, an offset that moves every entry out of the map is no error.
	BitVoxelMap bit_map = make_bit_map();
	const auto before = list.download_bit_vectors();
	EXPECT_EQ(list.collide_with(bit_map, { 14, 0, 0 }), 0u);
	EXPECT_EQ(list.collide_with(bit_map, { 0, 0, std::numeric_limits<int32_t>::min() }), 0u);
	EXPECT_EQ(list.collide_with(bit_map, { std::numeric_limits<int32_t>::max(), 0, 0 }), 0u);
	EXPECT_EQ(list.download_bit_vectors(), before);

	// An empty list collides with nothing, also in the types form.
	BitVoxelList empty(ctx(), kDims, kSide);
	EXPECT_EQ(empty.collide_with(bit_map), 0u);
	const auto types = empty.collide_with_types(bit_map);
	EXPECT_EQ(types.count, 0u);
	EXPECT_EQ(types.types, BitVector256{});
	EXPECT_EQ(empty.collide_with_type_mask(bit_map, bits_of({ 1 })), 0u);

	// Maps of another context are refused.
	Context other_context;
	BitVoxelMap foreign(other_context, kDims, kSide);
	ProbVoxelMap foreign_prob(other_context, kDims, kSide);
	expect_error([&] { (void)list.collide_with(foreign); }, "belongs to another context");
	expect_error([&] { (void)list.collide_with(foreign_prob); }, "belongs to another context");
	expect_error([&] { (void)list.collide_with_types(foreign); }, "belongs to another context");
	expect_error([&] { (void)list.collide_with_types(foreign_prob); }, "belongs to another context");
	expect_error([&] { (void)list.collide_with_type_mask(foreign, bits_of({ 1 })); }, "belongs to another context");
	expect_error([&] { (void)list.collide_with_type_mask(foreign_prob, bits_of({ 1 })); }, "belongs to another context");
}
