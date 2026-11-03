// CUDA parity, scenario 1: list x map collisions (BitVoxelList::collideWith / collideWithTypes / collideWithTypeMask against a bit map and a
// probabilistic map). Plain C++ without any GPU API: it is included by the CUDA dump program (cuda_collisions_dump.cu) and by the Vulkan test
// (tests/vulkan/test_cuda_parity.cpp), so that both sides run the same inputs and the same calls and print the same lines.
// Method and results: Agent/Analysis/2026-11-03_cuda_parity.md
#ifndef GVL_CUDA_PARITY_INPUTS_H_INCLUDED
#define GVL_CUDA_PARITY_INPUTS_H_INCLUDED

#include <array>
#include <cstdint>
#include <cstdio>
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
}

#endif
