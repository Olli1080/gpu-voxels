// -- BEGIN LICENSE BLOCK ----------------------------------------------
// This file is part of the GPU Voxels Software Library.
// Licensed under the CDDL (see LICENSE.txt in the top directory).
// -- END LICENSE BLOCK ------------------------------------------------

#include <gpu_voxels/vulkan/Primitives.h>

#include <algorithm>

#include "Shaders.h"

namespace gpu_voxels::vulkan
{
	namespace
	{
		constexpr uint32_t kScanBlock = 512;
		constexpr uint32_t kSortTile = 256;
		constexpr uint32_t kMaxSortGroups = 65535;

		struct CountPush { uint32_t n; };
		struct PassPush { uint32_t n, shift, groups, mask; };

		uint32_t grid_stride_groups(const Context& context, uint64_t items, uint32_t local_size)
		{
			const uint64_t needed = (items + local_size - 1) / local_size;
			const uint64_t cap = std::min<uint64_t>(context.limits().maxComputeWorkGroupCount[0], 65535);
			return static_cast<uint32_t>(std::clamp<uint64_t>(needed, 1, cap));
		}

		std::unique_ptr<DeviceBuffer> make_buffer(Context& context, KeepAlive& keep_alive, uint64_t words)
		{
			return std::make_unique<DeviceBuffer>(context, std::max<uint64_t>(words, 1) * sizeof(uint32_t));
		}
	}

	struct Primitives::Kernels
	{
		explicit Kernels(Context& context)
			: scan_blocks(context, GVL_SHADER_SPAN(scan_blocks_comp), 3, sizeof(CountPush)),
			  scan_add(context, GVL_SHADER_SPAN(scan_add_comp), 2, sizeof(CountPush)),
			  iota(context, GVL_SHADER_SPAN(fill_iota_comp), 1, sizeof(CountPush)),
			  histogram(context, GVL_SHADER_SPAN(radix_histogram_comp), 2, sizeof(PassPush)),
			  scatter(context, GVL_SHADER_SPAN(radix_scatter_comp), 5, sizeof(PassPush)),
			  histogram64(context, GVL_SHADER_SPAN(radix_histogram64_comp), 2, sizeof(PassPush)),
			  scatter64(context, GVL_SHADER_SPAN(radix_scatter64_comp), 5, sizeof(PassPush))
		{
		}

		ComputePipeline scan_blocks;
		ComputePipeline scan_add;
		ComputePipeline iota;
		ComputePipeline histogram;
		ComputePipeline scatter;
		ComputePipeline histogram64;
		ComputePipeline scatter64;
	};

	Primitives::Primitives(Context& context) : context_(context), kernels_(std::make_shared<Kernels>(context)) {}

	void Primitives::record_iota(CommandBatch& batch, DeviceBuffer& out, uint32_t n)
	{
		if (n == 0)
			return;
		if (out.size() < static_cast<uint64_t>(n) * sizeof(uint32_t))
			throw VulkanError("Primitives::record_iota: buffer too small");
		batch.dispatch(kernels_->iota, { &out }, grid_stride_groups(context_, n, 256), CountPush{ n });
	}

	void Primitives::record_exclusive_scan(CommandBatch& batch, const DeviceBuffer& in, DeviceBuffer& out, uint32_t n,
		KeepAlive& keep_alive)
	{
		if (n == 0)
			return;
		if (n > kMaxElements)
			throw VulkanError("Primitives::record_exclusive_scan: too many elements");
		if (&in == &out)
			throw VulkanError("Primitives::record_exclusive_scan: in and out must differ");
		const uint64_t bytes = static_cast<uint64_t>(n) * sizeof(uint32_t);
		if (in.size() < bytes || out.size() < bytes)
			throw VulkanError("Primitives::record_exclusive_scan: buffer too small");

		const uint32_t blocks = (n + kScanBlock - 1) / kScanBlock;
		keep_alive.push_back(make_buffer(context_, keep_alive, blocks));
		DeviceBuffer& sums = *keep_alive.back();
		batch.dispatch(kernels_->scan_blocks, { &in, &out, &sums }, blocks, CountPush{ n });
		if (blocks > 1)
		{
			keep_alive.push_back(make_buffer(context_, keep_alive, blocks));
			DeviceBuffer& offsets = *keep_alive.back();
			batch.barrier();
			record_exclusive_scan(batch, sums, offsets, blocks, keep_alive);
			batch.barrier();
			batch.dispatch(kernels_->scan_add, { &out, &offsets }, grid_stride_groups(context_, n, 256), CountPush{ n });
		}
	}

	void Primitives::record_sort_pairs(CommandBatch& batch, DeviceBuffer& keys, DeviceBuffer& values, uint32_t n,
		uint32_t key_bits, KeepAlive& keep_alive)
	{
		if (n > 1 && (key_bits == 0 || key_bits > 32))
			throw VulkanError("Primitives::record_sort_pairs: key_bits must be in 1..32");
		record_sort(batch, keys, values, n, key_bits, false, keep_alive);
	}

	void Primitives::record_sort_pairs64(CommandBatch& batch, DeviceBuffer& keys, DeviceBuffer& values, uint32_t n,
		uint32_t key_bits, KeepAlive& keep_alive)
	{
		if (n > 1 && (key_bits == 0 || key_bits > 64))
			throw VulkanError("Primitives::record_sort_pairs64: key_bits must be in 1..64");
		record_sort(batch, keys, values, n, key_bits, true, keep_alive);
	}

	void Primitives::record_sort(CommandBatch& batch, DeviceBuffer& keys, DeviceBuffer& values, uint32_t n, uint32_t key_bits,
		bool wide, KeepAlive& keep_alive)
	{
		if (n <= 1)
			return;
		static_assert(kMaxSortElements == kMaxSortGroups * kSortTile);
		const uint32_t groups = (n + kSortTile - 1) / kSortTile;
		if (n > kMaxSortElements)
			throw VulkanError("Primitives::record_sort_pairs: too many elements");
		const uint64_t key_words = wide ? 2ull * n : n;
		const uint64_t key_bytes = key_words * sizeof(uint32_t);
		const uint64_t bytes = static_cast<uint64_t>(n) * sizeof(uint32_t);
		if (keys.size() < key_bytes || values.size() < bytes)
			throw VulkanError("Primitives::record_sort_pairs: buffer too small");

		keep_alive.push_back(make_buffer(context_, keep_alive, key_words));
		DeviceBuffer& keys_tmp = *keep_alive.back();
		keep_alive.push_back(make_buffer(context_, keep_alive, n));
		DeviceBuffer& values_tmp = *keep_alive.back();
		keep_alive.push_back(make_buffer(context_, keep_alive, 256ull * groups));
		DeviceBuffer& histogram = *keep_alive.back();
		keep_alive.push_back(make_buffer(context_, keep_alive, 256ull * groups));
		DeviceBuffer& offsets = *keep_alive.back();

		DeviceBuffer* src_keys = &keys;
		DeviceBuffer* src_values = &values;
		DeviceBuffer* dst_keys = &keys_tmp;
		DeviceBuffer* dst_values = &values_tmp;
		const uint32_t passes = (key_bits + 7) / 8;
		for (uint32_t pass = 0; pass < passes; ++pass)
		{
			const uint32_t bits_left = key_bits - pass * 8;
			const PassPush push{ n, pass * 8, groups, bits_left >= 8 ? 255u : ((1u << bits_left) - 1u) };
			batch.dispatch(wide ? kernels_->histogram64 : kernels_->histogram, { src_keys, &histogram }, groups, push);
			batch.barrier();
			record_exclusive_scan(batch, histogram, offsets, 256 * groups, keep_alive);
			batch.barrier();
			batch.dispatch(wide ? kernels_->scatter64 : kernels_->scatter, { src_keys, src_values, dst_keys, dst_values, &offsets }, groups, push);
			batch.barrier();
			std::swap(src_keys, dst_keys);
			std::swap(src_values, dst_values);
		}
		if (src_keys != &keys) // an odd number of passes left the result in the temporary buffers
		{
			batch.copy(*src_keys, keys, key_bytes);
			batch.copy(*src_values, values, bytes);
		}
	}

	std::vector<uint32_t> Primitives::exclusive_scan(const std::vector<uint32_t>& input)
	{
		if (input.empty())
			return {};
		const uint32_t n = static_cast<uint32_t>(input.size());
		DeviceBuffer in(context_, input.size() * sizeof(uint32_t));
		DeviceBuffer out(context_, input.size() * sizeof(uint32_t));
		in.upload<uint32_t>(input);
		KeepAlive keep_alive;
		CommandBatch batch(context_);
		record_exclusive_scan(batch, in, out, n, keep_alive);
		batch.submit_and_wait();
		return out.download_vector<uint32_t>(input.size());
	}

	void Primitives::sort_pairs(std::vector<uint32_t>& keys, std::vector<uint32_t>& values, uint32_t key_bits)
	{
		if (keys.size() != values.size())
			throw VulkanError("Primitives::sort_pairs: keys and values differ in size");
		if (keys.size() <= 1)
			return;
		const uint32_t n = static_cast<uint32_t>(keys.size());
		DeviceBuffer key_buffer(context_, keys.size() * sizeof(uint32_t));
		DeviceBuffer value_buffer(context_, values.size() * sizeof(uint32_t));
		key_buffer.upload<uint32_t>(keys);
		value_buffer.upload<uint32_t>(values);
		KeepAlive keep_alive;
		CommandBatch batch(context_);
		record_sort_pairs(batch, key_buffer, value_buffer, n, key_bits, keep_alive);
		batch.submit_and_wait();
		keys = key_buffer.download_vector<uint32_t>(keys.size());
		values = value_buffer.download_vector<uint32_t>(values.size());
	}

	void Primitives::sort_pairs64(std::vector<uint64_t>& keys, std::vector<uint32_t>& values, uint32_t key_bits)
	{
		if (keys.size() != values.size())
			throw VulkanError("Primitives::sort_pairs64: keys and values differ in size");
		if (keys.size() <= 1)
			return;
		const uint32_t n = static_cast<uint32_t>(keys.size());
		DeviceBuffer key_buffer(context_, keys.size() * sizeof(uint64_t));
		DeviceBuffer value_buffer(context_, values.size() * sizeof(uint32_t));
		key_buffer.upload<uint64_t>(keys);
		value_buffer.upload<uint32_t>(values);
		KeepAlive keep_alive;
		CommandBatch batch(context_);
		record_sort_pairs64(batch, key_buffer, value_buffer, n, key_bits, keep_alive);
		batch.submit_and_wait();
		keys = key_buffer.download_vector<uint64_t>(keys.size());
		values = value_buffer.download_vector<uint32_t>(values.size());
	}
}
