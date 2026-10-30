// -- BEGIN LICENSE BLOCK ----------------------------------------------
// This file is part of the GPU Voxels Software Library.
// Licensed under the CDDL (see LICENSE.txt in the top directory).
// -- END LICENSE BLOCK ------------------------------------------------

/*!\file
 * \brief  Data-parallel primitives replacing the Thrust calls of the CUDA code: exclusive scan and stable radix sort.
 *
 * The record_* functions append commands to a CommandBatch (ordered by barriers) and may allocate temporary buffers;
 * these are handed to `keep_alive`, which must outlive the batch execution. The blocking helpers below them are
 * convenience wrappers for tests and simple callers.
 */
#ifndef GPU_VOXELS_VULKAN_PRIMITIVES_H_INCLUDED
#define GPU_VOXELS_VULKAN_PRIMITIVES_H_INCLUDED

#include <cstdint>
#include <memory>
#include <vector>

#include <gpu_voxels/vulkan/CommandBatch.h>
#include <gpu_voxels/vulkan/ComputePipeline.h>
#include <gpu_voxels/vulkan/Context.h>
#include <gpu_voxels/vulkan/DeviceBuffer.h>

namespace gpu_voxels::vulkan
{
	using KeepAlive = std::vector<std::unique_ptr<DeviceBuffer>>;

	class Primitives
	{
	public:
		/// Maximum number of elements of one scan / sort (limited by the 65535 work groups of the second scan level).
		static constexpr uint32_t kMaxElements = 1u << 24;

		explicit Primitives(Context& context);

		[[nodiscard]] Context& context() const { return context_; }

		/// out[i] = in[0] + ... + in[i-1] (uint32 arithmetic, wraps). `in` and `out` must be different buffers of >= n words.
		void record_exclusive_scan(CommandBatch& batch, const DeviceBuffer& in, DeviceBuffer& out, uint32_t n, KeepAlive& keep_alive);

		/// Stable ascending sort of the pairs (keys[i], values[i]), i < n, by the low `key_bits` bits of the keys
		/// (1..32; the higher bits must be zero or are ignored for ordering). Both buffers hold >= n words and are sorted in place.
		void record_sort_pairs(CommandBatch& batch, DeviceBuffer& keys, DeviceBuffer& values, uint32_t n, uint32_t key_bits,
			KeepAlive& keep_alive);

		/// Same for 64-bit keys: element i has the key words keys[2i] (low) and keys[2i+1] (high), the layout of a little-endian
		/// uint64_t array; key_bits is 1..64; `keys` holds >= 2n words. Needed for the octree (Morton ids have 42 bits).
		void record_sort_pairs64(CommandBatch& batch, DeviceBuffer& keys, DeviceBuffer& values, uint32_t n, uint32_t key_bits,
			KeepAlive& keep_alive);

		/// out[i] = i for i < n
		void record_iota(CommandBatch& batch, DeviceBuffer& out, uint32_t n);

		// ---- blocking helpers ----
		[[nodiscard]] std::vector<uint32_t> exclusive_scan(const std::vector<uint32_t>& input);
		/// Sorts in place; both vectors must have the same size.
		void sort_pairs(std::vector<uint32_t>& keys, std::vector<uint32_t>& values, uint32_t key_bits = 32);
		void sort_pairs64(std::vector<uint64_t>& keys, std::vector<uint32_t>& values, uint32_t key_bits = 64);

	private:
		void record_sort(CommandBatch& batch, DeviceBuffer& keys, DeviceBuffer& values, uint32_t n, uint32_t key_bits, bool wide,
			KeepAlive& keep_alive);

		Context& context_;
		struct Kernels;
		std::shared_ptr<Kernels> kernels_;
	};
}

#endif
