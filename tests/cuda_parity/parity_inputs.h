// CUDA parity, scenario 1: list x map collisions (BitVoxelList::collideWith / collideWithTypes / collideWithTypeMask against a bit map and a
// probabilistic map). Plain C++ without any GPU API: it is included by the CUDA dump program (cuda_collisions_dump.cu) and by the Vulkan test
// (tests/vulkan/test_cuda_parity.cpp), so that both sides run the same inputs and the same calls and print the same lines.
// Method and results: Agent/Analysis/2026-11-03_cuda_parity.md
#ifndef GVL_CUDA_PARITY_INPUTS_H_INCLUDED
#define GVL_CUDA_PARITY_INPUTS_H_INCLUDED

#include <array>
#include <map>
#include <set>
#include <cstdint>
#include <cstdio>
#include <limits>
#include <string>
#include <utility>
#include <vector>

namespace gvl_parity
{
	inline constexpr int kDim = 16; // maps and lists: 16 x 16 x 16 voxels of side 1

	struct Lcg
	{
		uint64_t state;
		explicit Lcg(uint64_t seed) : state(seed) {}
		uint32_t next()
		{
			state = state * 6364136223846793005ULL + 1442695040888963407ULL;
			return static_cast<uint32_t>(state >> 33);
		}
	};

	/// Points are voxel centres (x + 0.5), so that the voxel does not depend on the rounding of floor(p / side).
	struct PointSet
	{
		int meaning;
		std::vector<std::array<float, 3>> points;
	};

	inline PointSet random_set(Lcg& rng, int meaning, int count)
	{
		PointSet set{ meaning, {} };
		for (int i = 0; i < count; ++i)
		{
			const float x = static_cast<float>(rng.next() % kDim) + 0.5f;
			const float y = static_cast<float>(rng.next() % kDim) + 0.5f;
			const float z = static_cast<float>(rng.next() % kDim) + 0.5f;
			set.points.push_back({ x, y, z });
		}
		return set;
	}

	/// Bit map content: five meanings (bit indices), inserted in this order.
	inline std::vector<PointSet> bit_map_sets()
	{
		Lcg rng(1);
		std::vector<PointSet> sets;
		for (const int meaning : { 1, 5, 40, 130, 200 })
			sets.push_back(random_set(rng, meaning, 60));
		sets.push_back(random_set(rng, 0, 60)); // voxels with only the FREE bit (not occupied)
		return sets;
	}

	/// Probabilistic map content: occupied twice (same points: sums beyond 127 saturate), +71, -29 and a free update, one call each.
	inline std::vector<PointSet> prob_map_sets()
	{
		std::vector<PointSet> sets;
		{ Lcg rng(2); sets.push_back(random_set(rng, 1, 120)); }
		{ Lcg rng(2); sets.push_back(random_set(rng, 1, 120)); }
		{ Lcg rng(3); sets.push_back(random_set(rng, 200, 80)); }
		{ Lcg rng(4); sets.push_back(random_set(rng, 100, 80)); }
		{ Lcg rng(5); sets.push_back(random_set(rng, 0, 40)); }
		// swept volume meanings 129 / 130 / 131 add 0 / 1 / 2: voxels with the occupancy 0, 1, 2 where the threshold rules differ
		{ Lcg rng(7); sets.push_back(random_set(rng, 129, 20)); }
		{ Lcg rng(8); sets.push_back(random_set(rng, 130, 20)); }
		{ Lcg rng(9); sets.push_back(random_set(rng, 131, 20)); }
		return sets;
	}

	/// List content: meaning 0 (only the FREE bit) and six more; entries at the same voxel merge their bits.
	inline std::vector<PointSet> list_sets()
	{
		Lcg rng(6);
		std::vector<PointSet> sets;
		sets.push_back(random_set(rng, 0, 40));
		for (const int meaning : { 1, 5, 40, 77, 130, 200 })
			sets.push_back(random_set(rng, meaning, 50));
		return sets;
	}

	enum class Form { Default, Types, Mask };

	struct Call
	{
		Form form;
		bool prob;
		float threshold;
		std::array<int, 3> offset;
		int mask; ///< index into masks(), Form::Mask only
	};

	inline std::vector<std::vector<int>> masks()
	{
		std::vector<int> all;
		for (int b = 0; b < 256; ++b)
			all.push_back(b);
		return { {}, { 1 }, { 5 }, { 5, 40, 130 }, { 0 }, all, { 77, 200, 255 } };
	}

	inline std::vector<Call> calls()
	{
		const std::vector<std::array<int, 3>> offsets{ { 0, 0, 0 }, { 1, 0, 0 }, { -1, 0, 0 }, { 0, 1, 0 }, { 0, -1, 0 }, { 0, 0, 1 }, { 0, 0, -1 },
			{ 3, 2, 1 }, { 15, 0, 0 }, { -15, 0, 0 }, { 0, 15, 0 } };
		std::vector<Call> out;
		for (const Form form : { Form::Default, Form::Types })
			for (const bool prob : { false, true })
				for (const float threshold : { 1.0f, 0.5f, 0.0f })
					for (const auto& offset : offsets)
						out.push_back({ form, prob, threshold, offset, 0 });
		for (const bool prob : { false, true })
			for (const float threshold : { 1.0f, 0.5f })
				for (const auto& offset : offsets)
					for (int mask = 0; mask < static_cast<int>(masks().size()); ++mask)
						out.push_back({ Form::Mask, prob, threshold, offset, mask });
		return out;
	}

	inline std::string bits_text(const std::vector<int>& bits)
	{
		std::string text = "{";
		for (std::size_t i = 0; i < bits.size(); ++i)
			text += (i ? "," : "") + std::to_string(bits[i]);
		return text + "}";
	}

	/// One result line; `types` is only printed for Form::Types.
	inline std::string line(const Call& c, uint64_t count, const std::vector<int>& types)
	{
		char head[160];
		std::snprintf(head, sizeof head, "%s %s thr=%.1f off=%d,%d,%d", c.form == Form::Default ? "default" : c.form == Form::Types ? "types" : "mask",
			c.prob ? "prob" : "bit", static_cast<double>(c.threshold), c.offset[0], c.offset[1], c.offset[2]);
		std::string text = head;
		if (c.form == Form::Mask)
			text += " mask=" + std::to_string(c.mask);
		text += " count=" + std::to_string(count);
		if (c.form == Form::Types)
			text += " types=" + bits_text(types);
		return text;
	}

	// ---- scenario 2: merge of a bit list into a bit list with an offset (GpuVoxelsMap::merge) ----

	/// Content of the list that is merged into (a second list from the same generator, some voxels coincide with the source).
	inline std::vector<PointSet> merge_target_sets()
	{
		Lcg rng(10);
		std::vector<PointSet> sets;
		sets.push_back(random_set(rng, 1, 60));
		sets.push_back(random_set(rng, 77, 60));
		sets.push_back(random_set(rng, 0, 20));
		return sets;
	}

	struct MergeCall
	{
		std::array<int, 3> offset;
		int meaning; ///< -1: no new meaning
	};

	inline std::vector<MergeCall> merge_calls()
	{
		const std::vector<std::array<int, 3>> offsets{ { 0, 0, 0 }, { 1, 0, 0 }, { -1, 0, 0 }, { 0, 1, 0 }, { 0, -1, 0 }, { 0, 0, 1 }, { 0, 0, -1 },
			{ 3, 2, 1 }, { 15, 0, 0 }, { -15, 0, 0 }, { 0, 15, 0 }, { -3, -3, -3 }, { 5, 5, 5 }, { 16, 0, 0 }, { 0, 0, -20 } };
		std::vector<MergeCall> out;
		for (const int meaning : { -1, 9 })
			for (const auto& offset : offsets)
				out.push_back({ offset, meaning });
		return out;
	}

	struct ListEntry
	{
		uint32_t x, y, z;
		std::vector<int> bits;
	};

	/// FNV-1a over the entries in list order: coordinates, number of set bits, the set bits.
	inline uint64_t hash_entries(const std::vector<ListEntry>& entries)
	{
		uint64_t h = 14695981039346656037ULL;
		const auto add = [&h](uint32_t v)
		{
			for (int i = 0; i < 4; ++i)
			{
				h ^= (v >> (8 * i)) & 0xFFu;
				h *= 1099511628211ULL;
			}
		};
		for (const ListEntry& e : entries)
		{
			add(e.x);
			add(e.y);
			add(e.z);
			add(static_cast<uint32_t>(e.bits.size()));
			for (const int b : e.bits)
				add(static_cast<uint32_t>(b));
		}
		return h;
	}

	inline std::string merge_line(const MergeCall& c, const std::vector<ListEntry>& entries)
	{
		char text[200];
		std::snprintf(text, sizeof text, "merge off=%d,%d,%d meaning=%d entries=%zu hash=%016llx", c.offset[0], c.offset[1], c.offset[2], c.meaning, entries.size(),
			static_cast<unsigned long long>(hash_entries(entries)));
		return text;
	}
	// ---- scenario 4: probabilistic map, insertPointCloud with swept volume meanings (one call each) ----
	// Same-sign updates inside one call give the same result in every order (CUDA applies them with atomics), so the whole map is compared after every call.

	inline std::vector<PointSet> prob_insert_sets()
	{
		Lcg rng(20);
		std::vector<PointSet> sets;
		// 1 occupied (+72), 200 (+71), 100 (-29), 0 free (-10), 129 / 130 / 131 (+0 / +1 / +2), 254 (+125), 4 (-125), 50 (-79), 3 unknown (sets the voxel unknown)
		for (const int meaning : { 1, 1, 200, 100, 0, 129, 130, 131, 254, 4, 50, 3, 1 })
		{
			// distinct voxels inside one call: CUDA loses all but one update of a voxel that is hit twice in one call (see prob_duplicate_sets)
			PointSet set{ meaning, {} };
			std::set<std::array<int, 3>> used;
			while (set.points.size() < 80)
			{
				const PointSet one = random_set(rng, meaning, 1);
				const std::array<int, 3> voxel{ static_cast<int>(one.points[0][0]), static_cast<int>(one.points[0][1]), static_cast<int>(one.points[0][2]) };
				if (used.insert(voxel).second)
					set.points.push_back(one.points[0]);
			}
			sets.push_back(set);
		}
		return sets;
	}

	/// The same voxel twice (meaning 1: +72 twice) and four times (meaning 0: -10 four times) in one call, each on a fresh map.
	inline std::vector<PointSet> prob_duplicate_sets()
	{
		return { { 1, { { 5.5f, 5.5f, 5.5f }, { 5.5f, 5.5f, 5.5f } } }, { 0, { { 7.5f, 7.5f, 7.5f }, { 7.5f, 7.5f, 7.5f }, { 7.5f, 7.5f, 7.5f }, { 7.5f, 7.5f, 7.5f } } } };
	}

	/// FNV-1a over the occupancy of all voxels (the int8 values as bytes).
	inline uint64_t hash_occupancy(const std::vector<int>& values)
	{
		uint64_t h = 14695981039346656037ULL;
		for (const int v : values)
		{
			h ^= static_cast<uint8_t>(static_cast<int8_t>(v));
			h *= 1099511628211ULL;
		}
		return h;
	}

	// ---- scenario 3: voxel map x voxel map collisions with an offset (BitVoxelMap / ProbVoxelMap::collideWith(map, threshold, offset)) ----
	// A collision marks the voxel of the map it is called on (eBVM_COLLISION), so every call starts with fresh maps and the whole marked map is compared.

	/// `count` random voxel centres that are all different (CUDA loses all but one update of a probabilistic voxel hit twice in one call, K42).
	inline PointSet distinct_set(Lcg& rng, int meaning, int count)
	{
		PointSet set{ meaning, {} };
		std::set<std::array<int, 3>> used;
		while (static_cast<int>(set.points.size()) < count)
		{
			const PointSet one = random_set(rng, meaning, 1);
			const std::array<int, 3> voxel{ static_cast<int>(one.points[0][0]), static_cast<int>(one.points[0][1]), static_cast<int>(one.points[0][2]) };
			if (used.insert(voxel).second)
				set.points.push_back(one.points[0]);
		}
		return set;
	}

	/// The map the collision is called on (`this`, side 0) and the map it is called with (`other`, side 1), as bit map or probabilistic map.
	inline std::vector<PointSet> map_sets(bool prob, int side)
	{
		Lcg rng(prob ? 30 + side : 40 + side);
		std::vector<PointSet> sets;
		if (!prob)
			for (const int meaning : { 1, 5, 40, 130, 0 })
				sets.push_back(distinct_set(rng, meaning, 160));
		else
		{
			// occupied twice at the same voxels (72 + 72: the voxels saturate at 127, so that threshold 1.0 finds collisions), then +71, -29, free, +1, unknown
			Lcg again = rng;
			sets.push_back(distinct_set(rng, 1, 160));
			sets.push_back(distinct_set(again, 1, 160));
			for (const int meaning : { 200, 100, 0, 130, 3 })
				sets.push_back(distinct_set(rng, meaning, 160));
		}
		return sets;
	}

	struct MapCall
	{
		bool this_prob;
		bool other_prob;
		float threshold;
		std::array<int, 3> offset;
	};

	inline std::vector<MapCall> map_calls()
	{
		const std::vector<std::array<int, 3>> offsets{ { 0, 0, 0 }, { 1, 0, 0 }, { -1, 0, 0 }, { 0, 1, 0 }, { 0, -1, 0 }, { 0, 0, 1 }, { 0, 0, -1 },
			{ 3, 2, 1 }, { 15, 0, 0 }, { -15, 0, 0 }, { 0, 15, 0 }, { -3, -3, -3 }, { 5, 5, 5 }, { 16, 0, 0 }, { 0, 0, -20 } };
		std::vector<MapCall> out;
		for (const bool this_prob : { false, true })
			for (const bool other_prob : { false, true })
				for (const float threshold : { 1.0f, 0.5f, 0.0f })
				{
					if (!this_prob && !other_prob && threshold != 1.0f)
						continue; // two bit maps: the threshold has no meaning
					for (const auto& offset : offsets)
						out.push_back({ this_prob, other_prob, threshold, offset });
				}
		return out;
	}

	/// FNV-1a over a bit map: for every voxel the number of set bits and the set bits.
	inline uint64_t hash_bit_map(const std::vector<std::vector<int>>& voxels)
	{
		uint64_t h = 14695981039346656037ULL;
		const auto add = [&h](uint32_t v)
		{
			for (int i = 0; i < 4; ++i)
			{
				h ^= (v >> (8 * i)) & 0xFFu;
				h *= 1099511628211ULL;
			}
		};
		for (const auto& bits : voxels)
		{
			add(static_cast<uint32_t>(bits.size()));
			for (const int b : bits)
				add(static_cast<uint32_t>(b));
		}
		return h;
	}

	/// `after_hash` = hash of the map the call was made on after the call (marked voxels included).
	inline std::string map_line(const MapCall& c, uint64_t count, uint64_t after_hash)
	{
		char text[200];
		std::snprintf(text, sizeof text, "mapmap %s x %s thr=%.1f off=%d,%d,%d count=%llu hash=%016llx", c.this_prob ? "prob" : "bit", c.other_prob ? "prob" : "bit",
			static_cast<double>(c.threshold), c.offset[0], c.offset[1], c.offset[2], static_cast<unsigned long long>(count), static_cast<unsigned long long>(after_hash));
		return text;
	}

	inline std::string prob_line(const std::string& what, const std::vector<int>& values)
	{
		std::size_t unknown = 0, positive = 0, negative = 0;
		for (const int v : values)
		{
			unknown += v == -128 ? 1 : 0;
			positive += v > 0 ? 1 : 0;
			negative += (v < 0 && v != -128) ? 1 : 0;
		}
		// the distinct values with their number of voxels (unknown = -128 is left out): shows which values differ when the hashes do
		std::map<int, std::size_t> histogram;
		for (const int v : values)
			if (v != -128)
				++histogram[v];
		std::string values_text;
		for (const auto& [value, count] : histogram)
			values_text += (values_text.empty() ? "" : ",") + std::to_string(value) + ":" + std::to_string(count);
		char text[220];
		std::snprintf(text, sizeof text, "prob %s unknown=%zu positive=%zu negative=%zu hash=%016llx values=", what.c_str(), unknown, positive, negative,
			static_cast<unsigned long long>(hash_occupancy(values)));
		return text + values_text;
	}

	// ---- scenario 5: ProbVoxelMap::insertSensorData (CUDA) / insert_sensor_data (Vulkan), ray casting and the robot cut ----
	// One measurement per call: CUDA runs one thread per measurement without atomics, and every ray visits the sensor voxel, so several rays in one call lose
	// updates (K42) and would not be a defined input. Every group starts with a fresh map; within a group the calls accumulate (saturation at +-127 included).

	struct SensorCall
	{
		std::array<float, 3> pose;
		std::array<float, 3> hit;
		bool raycast;
		int robot_bit; ///< -1: no robot cut; otherwise the bit of the robot map that marks the robot (cut_real_robot with this robot_voxel_meaning)
	};

	struct SensorGroup
	{
		std::string name;
		std::vector<SensorCall> calls;
	};

	/// The robot map: bit 5 in x, y, z = 0..7, bit 7 in x, y = 8..15, z = 0..7 and bit 40 (second 32-bit word) in x = 0..7, y = 8..15, z = 0..7 (512 voxels each, voxel centres).
	inline std::vector<PointSet> sensor_robot_sets()
	{
		PointSet a{ 5, {} }, b{ 7, {} }, c{ 40, {} };
		for (int z = 0; z < 8; ++z)
			for (int y = 0; y < 8; ++y)
				for (int x = 0; x < 8; ++x)
				{
					a.points.push_back({ x + 0.5f, y + 0.5f, z + 0.5f });
					b.points.push_back({ x + 8.5f, y + 8.5f, z + 0.5f });
					c.points.push_back({ x + 0.5f, y + 8.5f, z + 0.5f });
				}
		return { a, b, c };
	}

	inline std::vector<SensorGroup> sensor_groups()
	{
		Lcg rng(50);
		const auto centre = [&rng](int lo, int hi) { return static_cast<float>(lo + static_cast<int>(rng.next() % static_cast<uint32_t>(hi - lo))) + 0.5f; };
		const auto fraction = [&rng]() { return static_cast<float>(rng.next() % 1600) / 100.0f; }; // 0.00 .. 15.99
		const float nan = std::numeric_limits<float>::quiet_NaN();
		const float inf = std::numeric_limits<float>::infinity();
		std::vector<SensorGroup> groups;

		// A: four sensor poses (voxel centre and not), hits at voxel centres and at arbitrary positions, with and without ray casting
		{
			SensorGroup g{ "rays", {} };
			const std::vector<std::array<float, 3>> poses{ { 0.5f, 0.5f, 0.5f }, { 8.25f, 8.75f, 8.5f }, { 15.5f, 3.5f, 9.5f }, { 2.5f, 14.5f, 7.5f } };
			for (const bool raycast : { true, false })
				for (const auto& pose : poses)
					for (int i = 0; i < 10; ++i)
					{
						const std::array<float, 3> hit = i % 2 == 0 ? std::array<float, 3>{ centre(0, 16), centre(0, 16), centre(0, 16) } : std::array<float, 3>{ fraction(), fraction(), fraction() };
						g.calls.push_back({ pose, hit, raycast, -1 });
					}
			groups.push_back(g);
		}
		// B: special rays: hit in the sensor voxel (also off centre), neighbour, along an axis, corner to corner in both directions
		{
			SensorGroup g{ "special", {} };
			const std::array<float, 3> p{ 5.5f, 5.5f, 5.5f };
			const std::vector<std::pair<std::array<float, 3>, std::array<float, 3>>> rays{ { p, p }, { p, { 5.9f, 5.1f, 5.2f } }, { p, { 6.5f, 5.5f, 5.5f } }, { p, { 5.5f, 5.5f, 0.5f } },
				{ p, { 0.5f, 5.5f, 5.5f } }, { p, { 15.5f, 15.5f, 15.5f } }, { p, { 0.5f, 0.5f, 0.5f } }, { { 0.5f, 0.5f, 0.5f }, { 15.5f, 15.5f, 15.5f } }, { { 15.5f, 15.5f, 15.5f }, { 0.5f, 0.5f, 0.5f } },
				{ { 0.5f, 15.5f, 0.5f }, { 15.5f, 0.5f, 15.5f } }, { { 3.5f, 3.5f, 3.5f }, { 4.5f, 12.5f, 7.5f } } };
			for (const bool raycast : { true, false })
				for (const auto& [pose, hit] : rays)
					g.calls.push_back({ pose, hit, raycast, -1 });
			groups.push_back(g);
		}
		// C: measurements and sensor poses beyond the upper border, NaN hits and infinity: dropped by CUDA and Vulkan alike
		{
			SensorGroup g{ "outside", {} };
			const std::array<float, 3> in{ 5.5f, 5.5f, 5.5f };
			const std::vector<std::array<float, 3>> hits{ { 16.5f, 5.5f, 5.5f }, { 5.5f, 16.0f, 5.5f }, { 5.5f, 5.5f, 1e9f }, { nan, 5.5f, 5.5f }, { 5.5f, inf, 5.5f }, { 5.5f, 5.5f, 20.5f } };
			const std::vector<std::array<float, 3>> poses{ { 16.5f, 5.5f, 5.5f }, { 5.5f, 5.5f, 16.0f }, { 5.5f, inf, 5.5f } };
			for (const bool raycast : { true, false })
			{
				for (const auto& hit : hits)
					g.calls.push_back({ in, hit, raycast, -1 });
				for (const auto& pose : poses)
					g.calls.push_back({ pose, { 9.5f, 9.5f, 9.5f }, raycast, -1 });
			}
			groups.push_back(g);
		}
		// C2: negative coordinates, -infinity and a NaN sensor pose. CUDA converts floor(c / side) to uint32 on the GPU, which saturates: these become voxel 0 on
		// that axis (they are NOT dropped); Vulkan drops them (K44). The test compares CUDA with Vulkan fed the clamped coordinates (0.5) instead.
		{
			SensorGroup g{ "negative", {} };
			const std::array<float, 3> in{ 5.5f, 5.5f, 5.5f };
			const std::vector<std::array<float, 3>> hits{ { -0.5f, 5.5f, 5.5f }, { 5.5f, -0.01f, 5.5f }, { 5.5f, 5.5f, -inf }, { -7.5f, -7.5f, -7.5f }, { -0.5f, 12.5f, 3.5f } };
			const std::vector<std::array<float, 3>> poses{ { -0.5f, 5.5f, 5.5f }, { nan, 5.5f, 5.5f }, { 5.5f, -inf, 5.5f }, { -3.5f, -3.5f, 14.5f } };
			for (const bool raycast : { true, false })
			{
				for (const auto& hit : hits)
					g.calls.push_back({ in, hit, raycast, -1 });
				for (const auto& pose : poses)
					g.calls.push_back({ pose, { 9.5f, 9.5f, 9.5f }, raycast, -1 });
			}
			groups.push_back(g);
		}
		// D: robot cut: hits inside the bit 5, bit 7 and bit 40 blocks and outside all, cut by bit 5, 7, 40 and a bit that no voxel has
		{
			SensorGroup g{ "robot", {} };
			const std::array<float, 3> pose{ 12.5f, 4.5f, 12.5f };
			for (const bool raycast : { true, false })
				for (const int bit : { 5, 7, 9, 40 })
					for (int i = 0; i < 16; ++i)
					{
						std::array<float, 3> hit;
						if (i < 4)
							hit = { centre(0, 8), centre(0, 8), centre(0, 8) }; // bit 5 block
						else if (i < 8)
							hit = { centre(8, 16), centre(8, 16), centre(0, 8) }; // bit 7 block
						else if (i < 12)
							hit = { centre(0, 8), centre(8, 16), centre(0, 8) }; // bit 40 block
						else
							hit = { centre(0, 16), centre(0, 16), centre(8, 16) }; // above all blocks
						g.calls.push_back({ pose, hit, raycast, bit });
					}
			// the ray crosses the robot but the hit does not: not cut
			g.calls.push_back({ { 0.5f, 0.5f, 0.5f }, { 12.5f, 12.5f, 12.5f }, true, 5 });
			groups.push_back(g);
		}
		return groups;
	}

	/// What CUDA does with negative and NaN coordinates (measured, group "negative"): the GPU conversion of floor(c / side) to uint32 saturates, so a negative
	/// coordinate (-inf included) and a NaN sensor pose coordinate become voxel 0 on that axis. A NaN hit is dropped by CUDA's isnan check, as in Vulkan.
	inline SensorCall cuda_saturated(SensorCall c)
	{
		for (int i = 0; i < 3; ++i)
		{
			if (c.hit[i] < 0.0f)
				c.hit[i] = 0.5f;
			if (!(c.pose[i] >= 0.0f))
				c.pose[i] = 0.5f;
		}
		return c;
	}

	/// Points for insertPointCloud (one point per call, meaning 1, cumulative on one map): the same question as the group "negative" for the point cloud kernel.
	/// Negative coordinates, -inf and NaN, one point beyond the border, and two ordinary points.
	inline std::vector<std::array<float, 3>> pointcloud_edge_points()
	{
		const float nan = std::numeric_limits<float>::quiet_NaN();
		const float inf = std::numeric_limits<float>::infinity();
		return { { -0.5f, 5.5f, 5.5f }, { 5.5f, -0.01f, 5.5f }, { 5.5f, 5.5f, -inf }, { nan, 9.5f, 9.5f }, { -7.5f, -7.5f, -7.5f }, { 16.5f, 5.5f, 5.5f }, { 5.5f, inf, 5.5f }, { 3.5f, 3.5f, 3.5f }, { 12.5f, 1.5f, 7.5f } };
	}

	/// What the unfixed CUDA did with such a point: every negative or NaN coordinate became voxel 0 on its axis.
	inline std::array<float, 3> cuda_saturated_point(std::array<float, 3> p)
	{
		for (float& c : p)
			if (!(c >= 0.0f))
				c = 0.5f;
		return p;
	}

	inline std::string sensor_label(const std::string& group, std::size_t index, const SensorCall& c)
	{
		return "sensor " + group + " " + std::to_string(index) + " ray=" + (c.raycast ? "1" : "0") + " robot=" + std::to_string(c.robot_bit);
	}
}

#endif
