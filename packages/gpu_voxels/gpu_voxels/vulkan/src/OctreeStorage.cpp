// -- BEGIN LICENSE BLOCK ----------------------------------------------
// This file is part of the GPU Voxels Software Library.
// Licensed under the CDDL (see LICENSE.txt in the top directory).
// -- END LICENSE BLOCK ------------------------------------------------

#include <gpu_voxels/vulkan/OctreeStorage.h>

#include <algorithm>
#include <bit>
#include <cmath>
#include <cstring>
#include <fstream>
#include <istream>
#include <ostream>
#include <vector>

#include <gpu_voxels/vulkan/vulkan_common.h>

namespace gpu_voxels::vulkan
{
	static_assert(std::endian::native == std::endian::little, "the native octree format is little endian");

	namespace
	{
		constexpr char kMagic[8] = { 'G', 'V', 'L', 'O', 'C', 'T', '\0', '\1' };
		constexpr uint64_t kMaxNodes = uint64_t(1) << 32;
		constexpr std::size_t kReadBatch = 65536; // records per read: memory stays bounded by the data actually present

		uint64_t fnv1a(uint64_t hash, const uint8_t* data, std::size_t size)
		{
			for (std::size_t i = 0; i < size; ++i)
			{
				hash ^= data[i];
				hash *= 1099511628211ull;
			}
			return hash;
		}
		constexpr uint64_t kFnvBasis = 14695981039346656037ull;

		template<typename T>
		void put(uint8_t* at, T value)
		{
			std::memcpy(at, &value, sizeof(T));
		}

		template<typename T>
		T get(const uint8_t* at)
		{
			T value;
			std::memcpy(&value, at, sizeof(T));
			return value;
		}

		uint32_t kind_code(OctreeKind kind) { return kind == OctreeKind::Probabilistic ? 1u : 0u; }
	}

	void OctreeNativeStorage::write(Octree& tree, std::ostream& out) const
	{
		tree.propagate();
		const std::vector<OctreeNode> nodes = tree.dump();
		std::vector<uint8_t> bytes(kHeaderBytes + nodes.size() * kRecordBytes + kTrailerBytes, 0);
		std::memcpy(bytes.data(), kMagic, 8);
		put<uint32_t>(bytes.data() + 8, kVersion);
		put<uint32_t>(bytes.data() + 12, kind_code(tree.kind()));
		put<float>(bytes.data() + 16, tree.voxel_side_length());
		put<uint32_t>(bytes.data() + 20, 0);
		put<uint64_t>(bytes.data() + 24, nodes.size());
		uint8_t* record = bytes.data() + kHeaderBytes;
		for (const OctreeNode& n : nodes)
		{
			put<uint64_t>(record, n.id);
			record[8] = static_cast<uint8_t>(n.level);
			record[9] = n.status;
			record[10] = static_cast<uint8_t>(n.occupancy);
			record[11] = n.has_children ? 1 : 0;
			record += kRecordBytes;
		}
		put<uint64_t>(record, fnv1a(kFnvBasis, bytes.data(), bytes.size() - kTrailerBytes));
		out.write(reinterpret_cast<const char*>(bytes.data()), static_cast<std::streamsize>(bytes.size()));
		out.flush();
		if (!out)
			throw VulkanError("OctreeNativeStorage::write: writing to the stream failed");
	}

	void OctreeNativeStorage::read(Octree& tree, std::istream& in) const
	{
		auto read_exact = [&](uint8_t* into, std::size_t size)
		{
			in.read(reinterpret_cast<char*>(into), static_cast<std::streamsize>(size));
			if (static_cast<std::size_t>(in.gcount()) != size)
				throw VulkanError("OctreeNativeStorage::read: the data ends early (truncated file?)");
		};

		uint8_t header[kHeaderBytes];
		read_exact(header, kHeaderBytes);
		if (std::memcmp(header, kMagic, 8) != 0)
			throw VulkanError("OctreeNativeStorage::read: not an octree file (wrong magic)");
		if (get<uint32_t>(header + 8) != kVersion)
			throw VulkanError("OctreeNativeStorage::read: unsupported version " + std::to_string(get<uint32_t>(header + 8)));
		const uint32_t kind = get<uint32_t>(header + 12);
		if (kind > 1 || get<uint32_t>(header + 20) != 0)
			throw VulkanError("OctreeNativeStorage::read: corrupt header");
		if (kind != kind_code(tree.kind()))
			throw VulkanError(std::string("OctreeNativeStorage::read: the file holds a ") + (kind == 1 ? "probabilistic" : "deterministic") +
				" tree, the target is " + (kind == 1 ? "deterministic" : "probabilistic"));
		const float side = get<float>(header + 16);
		if (!std::isfinite(side) || side < 0.0f)
			throw VulkanError("OctreeNativeStorage::read: corrupt voxel side length");
		const uint64_t count = get<uint64_t>(header + 24);
		if (count > kMaxNodes)
			throw VulkanError("OctreeNativeStorage::read: corrupt node count");

		uint64_t hash = fnv1a(kFnvBasis, header, kHeaderBytes);
		std::vector<OctreeNode> nodes;
		std::vector<uint8_t> buffer;
		for (uint64_t done = 0; done < count;)
		{
			const std::size_t batch = static_cast<std::size_t>(std::min<uint64_t>(kReadBatch, count - done));
			buffer.resize(batch * kRecordBytes);
			read_exact(buffer.data(), buffer.size());
			hash = fnv1a(hash, buffer.data(), buffer.size());
			for (std::size_t i = 0; i < batch; ++i)
			{
				const uint8_t* record = buffer.data() + i * kRecordBytes;
				OctreeNode node;
				node.id = get<uint64_t>(record);
				node.level = record[8];
				node.status = record[9];
				node.occupancy = static_cast<int8_t>(record[10]);
				if (record[11] > 1 || get<uint32_t>(record + 12) != 0)
					throw VulkanError("OctreeNativeStorage::read: corrupt node record");
				node.has_children = record[11] != 0;
				nodes.push_back(node);
			}
			done += batch;
		}
		uint8_t trailer[kTrailerBytes];
		read_exact(trailer, kTrailerBytes);
		if (get<uint64_t>(trailer) != hash)
			throw VulkanError("OctreeNativeStorage::read: checksum mismatch (the data is damaged)");

		tree.load_nodes(nodes); // checks the structure; the tree stays as it was if this throws
		tree.set_voxel_side_length(side);
	}

	void OctreeStorage::write_file(Octree& tree, const std::string& path) const
	{
		std::ofstream out(path, std::ios::binary | std::ios::trunc);
		if (!out)
			throw VulkanError("OctreeStorage::write_file: cannot open " + path);
		write(tree, out);
		out.close();
		if (!out)
			throw VulkanError("OctreeStorage::write_file: writing " + path + " failed");
	}

	void OctreeStorage::read_file(Octree& tree, const std::string& path) const
	{
		std::ifstream in(path, std::ios::binary);
		if (!in)
			throw VulkanError("OctreeStorage::read_file: cannot open " + path);
		read(tree, in);
	}
}
