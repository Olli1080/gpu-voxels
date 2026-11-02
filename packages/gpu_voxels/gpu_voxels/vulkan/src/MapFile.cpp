// -- BEGIN LICENSE BLOCK ----------------------------------------------
// This file is part of the GPU Voxels Software Library.
// Licensed under the CDDL (see LICENSE.txt in the top directory).
// -- END LICENSE BLOCK ------------------------------------------------

#include <gpu_voxels/vulkan/MapFile.h>

#include <bit>
#include <cstring>
#include <fstream>
#include <string>
#include <system_error>

#include <gpu_voxels/vulkan/GpuVoxels.h>

static_assert(std::endian::native == std::endian::little, "the map files assume a little endian host");

namespace gpu_voxels::vulkan
{
	namespace
	{
		constexpr uint8_t kMagic[8] = { 'G', 'V', 'L', 'M', 'A', 'P', 0, 1 };
		constexpr uint32_t kVersion = 1;
		constexpr std::size_t kHeaderBytes = 48;
		constexpr std::size_t kTrailerBytes = 8;
		constexpr uint64_t kFnvBasis = 14695981039346656037ull;

		uint64_t fnv1a(uint64_t hash, const uint8_t* data, std::size_t size)
		{
			for (std::size_t i = 0; i < size; ++i)
			{
				hash ^= data[i];
				hash *= 1099511628211ull;
			}
			return hash;
		}

		template<class T>
		void put(std::vector<uint8_t>& out, T value)
		{
			const auto* bytes = reinterpret_cast<const uint8_t*>(&value);
			out.insert(out.end(), bytes, bytes + sizeof(T));
		}

		template<class T>
		T get(const uint8_t* at)
		{
			T value;
			std::memcpy(&value, at, sizeof(T));
			return value;
		}

		[[noreturn]] void fail(const std::filesystem::path& file, const std::string& what)
		{
			throw MapFileError(file.filename().string() + ": " + what);
		}

		/// Header + payload + trailer; the payload is appended by the caller between begin_file and end_file.
		std::vector<uint8_t> begin_file(MapType type, Dim3ui dims, float side, uint64_t count)
		{
			std::vector<uint8_t> bytes(kMagic, kMagic + 8);
			put<uint32_t>(bytes, kVersion);
			put<uint32_t>(bytes, static_cast<uint32_t>(type));
			put<uint32_t>(bytes, dims.x);
			put<uint32_t>(bytes, dims.y);
			put<uint32_t>(bytes, dims.z);
			put<float>(bytes, side);
			put<uint32_t>(bytes, 0);
			put<uint32_t>(bytes, 0);
			put<uint64_t>(bytes, count);
			return bytes;
		}

		void end_file(const std::filesystem::path& file, std::vector<uint8_t>& bytes)
		{
			put<uint64_t>(bytes, fnv1a(kFnvBasis, bytes.data(), bytes.size()));
			std::ofstream out(file, std::ios::binary | std::ios::trunc);
			if (!out)
				fail(file, "could not open the file for writing");
			out.write(reinterpret_cast<const char*>(bytes.data()), static_cast<std::streamsize>(bytes.size()));
			out.close();
			if (!out)
				fail(file, "writing the file failed");
		}

		template<class T>
		void append(std::vector<uint8_t>& bytes, const std::vector<T>& values)
		{
			const auto* raw = reinterpret_cast<const uint8_t*>(values.data());
			bytes.insert(bytes.end(), raw, raw + values.size() * sizeof(T));
		}

		/// What the reader needs of the target.
		struct Target
		{
			MapType type;
			Dim3ui dims;
			float side;
			bool is_list;
		};

		/// A file whose header, size and checksum are verified; `payload` is the part between header and trailer.
		struct Loaded
		{
			std::vector<uint8_t> bytes;
			uint64_t count = 0;
			const uint8_t* payload() const { return bytes.data() + kHeaderBytes; }
		};

		/// \param payload_bytes_per_item bytes of the payload per voxel (maps) or per entry (lists)
		Loaded load(const std::filesystem::path& file, const Target& target, uint64_t payload_bytes_per_item)
		{
			std::ifstream in(file, std::ios::binary);
			if (!in)
				fail(file, "could not open the file");
			std::error_code ec;
			const uint64_t file_size = std::filesystem::file_size(file, ec);
			if (ec)
				fail(file, "could not get the size of the file");
			if (file_size < kHeaderBytes + kTrailerBytes)
				fail(file, "the file is too short to be a map file");

			uint8_t header[kHeaderBytes];
			in.read(reinterpret_cast<char*>(header), kHeaderBytes);
			if (static_cast<std::size_t>(in.gcount()) != kHeaderBytes)
				fail(file, "could not read the header");
			if (std::memcmp(header, kMagic, 8) != 0)
				fail(file, "not a map file (wrong magic)");
			if (get<uint32_t>(header + 8) != kVersion)
				fail(file, "unsupported version " + std::to_string(get<uint32_t>(header + 8)));
			const uint32_t type = get<uint32_t>(header + 12);
			if (type != static_cast<uint32_t>(target.type))
				fail(file, "the file holds the map type " + std::to_string(type) + ", the target is " + map_type_name(target.type));
			const Dim3ui dims{ get<uint32_t>(header + 16), get<uint32_t>(header + 20), get<uint32_t>(header + 24) };
			if (!(dims == target.dims))
				fail(file, std::string("the dimensions of the file (") + std::to_string(dims.x) + " x " + std::to_string(dims.y) + " x " + std::to_string(dims.z) +
					") differ from those of the " + (target.is_list ? "reference map" : "map") + " (" + std::to_string(target.dims.x) + " x " +
					std::to_string(target.dims.y) + " x " + std::to_string(target.dims.z) + ")");
			if (get<float>(header + 28) != target.side)
				fail(file, "the voxel side length of the file (" + std::to_string(get<float>(header + 28)) + ") differs from that of the target (" +
					std::to_string(target.side) + ")");
			if (get<uint32_t>(header + 32) != 0 || get<uint32_t>(header + 36) != 0)
				fail(file, "corrupt header (reserved fields are not 0)");

			Loaded loaded;
			loaded.count = get<uint64_t>(header + 40);
			if (target.is_list ? loaded.count > VoxelListBase::kMaxEntries : loaded.count != target.dims.voxel_count())
				fail(file, "corrupt header (the count of " + std::string(target.is_list ? "entries" : "voxels") + " is " + std::to_string(loaded.count) + ")");
			const uint64_t expected = kHeaderBytes + loaded.count * payload_bytes_per_item + kTrailerBytes;
			if (file_size != expected)
				fail(file, "the file has " + std::to_string(file_size) + " bytes, the header announces " + std::to_string(expected) +
					(file_size < expected ? " (truncated file?)" : " (data after the end?)"));

			loaded.bytes.resize(static_cast<std::size_t>(file_size));
			std::memcpy(loaded.bytes.data(), header, kHeaderBytes);
			in.read(reinterpret_cast<char*>(loaded.bytes.data() + kHeaderBytes), static_cast<std::streamsize>(file_size - kHeaderBytes));
			if (static_cast<uint64_t>(in.gcount()) != file_size - kHeaderBytes)
				fail(file, "could not read the data");
			const std::size_t body = loaded.bytes.size() - kTrailerBytes;
			if (get<uint64_t>(loaded.bytes.data() + body) != fnv1a(kFnvBasis, loaded.bytes.data(), body))
				fail(file, "checksum mismatch (the file is damaged)");
			return loaded;
		}

		template<class T>
		std::vector<T> copy_out(const uint8_t* at, uint64_t count)
		{
			std::vector<T> values(static_cast<std::size_t>(count));
			if (count > 0)
				std::memcpy(values.data(), at, values.size() * sizeof(T));
			return values;
		}

		Target target_of(MapType type, Dim3ui dims, float side, bool is_list) { return { type, dims, side, is_list }; }

		void write_list(const std::filesystem::path& file, MapType type, const VoxelListBase& list, std::vector<uint8_t> payload_tail)
		{
			const uint32_t n = list.size();
			std::vector<uint8_t> bytes = begin_file(type, list.reference_dimensions(), list.voxel_side_length(), n);
			append(bytes, list.download_ids());
			append(bytes, list.download_coordinates());
			bytes.insert(bytes.end(), payload_tail.begin(), payload_tail.end());
			end_file(file, bytes);
		}
	}

	void write_map_file(const std::filesystem::path& file, const BitVoxelMap& map)
	{
		std::vector<uint8_t> bytes = begin_file(MT_BITVECTOR_VOXELMAP, map.dimensions(), map.voxel_side_length(), map.voxel_count());
		append(bytes, map.download_words());
		end_file(file, bytes);
	}

	void write_map_file(const std::filesystem::path& file, const ProbVoxelMap& map)
	{
		std::vector<uint8_t> bytes = begin_file(MT_PROBAB_VOXELMAP, map.dimensions(), map.voxel_side_length(), map.voxel_count());
		append(bytes, map.download_occupancy());
		end_file(file, bytes);
	}

	void write_map_file(const std::filesystem::path& file, const DistanceVoxelMap& map)
	{
		std::vector<uint8_t> bytes = begin_file(MT_DISTANCE_VOXELMAP, map.dimensions(), map.voxel_side_length(), map.voxel_count());
		append(bytes, map.download_words());
		end_file(file, bytes);
	}

	void write_map_file(const std::filesystem::path& file, const BitVoxelList& list)
	{
		std::vector<uint8_t> tail;
		const std::vector<uint32_t> words = list.download_voxel_words();
		const auto* raw = reinterpret_cast<const uint8_t*>(words.data());
		tail.assign(raw, raw + words.size() * sizeof(uint32_t));
		write_list(file, MT_BITVECTOR_VOXELLIST, list, std::move(tail));
	}

	void write_map_file(const std::filesystem::path& file, const CountingVoxelList& list)
	{
		const std::vector<int8_t> counts = list.download_counts();
		const auto* raw = reinterpret_cast<const uint8_t*>(counts.data());
		write_list(file, MT_COUNTING_VOXELLIST, list, std::vector<uint8_t>(raw, raw + counts.size()));
	}

	void read_map_file(const std::filesystem::path& file, BitVoxelMap& map)
	{
		const Loaded loaded = load(file, target_of(MT_BITVECTOR_VOXELMAP, map.dimensions(), map.voxel_side_length(), false), BitVoxelMap::kBytesPerVoxel);
		map.upload_words(copy_out<uint32_t>(loaded.payload(), loaded.count * BitVoxelMap::kWordsPerVoxel));
	}

	void read_map_file(const std::filesystem::path& file, ProbVoxelMap& map)
	{
		const Loaded loaded = load(file, target_of(MT_PROBAB_VOXELMAP, map.dimensions(), map.voxel_side_length(), false), 1);
		map.upload_occupancy(copy_out<int8_t>(loaded.payload(), loaded.count));
	}

	void read_map_file(const std::filesystem::path& file, DistanceVoxelMap& map)
	{
		const Loaded loaded = load(file, target_of(MT_DISTANCE_VOXELMAP, map.dimensions(), map.voxel_side_length(), false), DistanceVoxelMap::kBytesPerVoxel);
		const std::vector<uint32_t> words = copy_out<uint32_t>(loaded.payload(), loaded.count);
		for (std::size_t i = 0; i < words.size(); ++i)
			if ((words[i] >> 30) != 0)
				fail(file, "voxel " + std::to_string(i) + " has bits above bit 29");
		map.upload_words(words);
	}

	void read_map_file(const std::filesystem::path& file, BitVoxelList& list)
	{
		const Loaded loaded = load(file, target_of(MT_BITVECTOR_VOXELLIST, list.reference_dimensions(), list.voxel_side_length(), true), 4 + 12 + 32);
		const uint8_t* at = loaded.payload();
		const std::vector<uint32_t> ids = copy_out<uint32_t>(at, loaded.count);
		const std::vector<Coord3ui> coordinates = copy_out<Coord3ui>(at + loaded.count * 4, loaded.count);
		const std::vector<uint32_t> bits = copy_out<uint32_t>(at + loaded.count * 16, loaded.count * 8);
		if (const auto problem = VoxelListBase::entries_problem(list.reference_dimensions(), VoxelListBase::Kind::Bit, ids, coordinates, bits))
			fail(file, *problem);
		list.assign_entries(ids, coordinates, bits);
	}

	void read_map_file(const std::filesystem::path& file, CountingVoxelList& list)
	{
		const Loaded loaded = load(file, target_of(MT_COUNTING_VOXELLIST, list.reference_dimensions(), list.voxel_side_length(), true), 4 + 12 + 1);
		const uint8_t* at = loaded.payload();
		const std::vector<uint32_t> ids = copy_out<uint32_t>(at, loaded.count);
		const std::vector<Coord3ui> coordinates = copy_out<Coord3ui>(at + loaded.count * 4, loaded.count);
		std::vector<uint32_t> words(static_cast<std::size_t>(loaded.count));
		for (std::size_t i = 0; i < words.size(); ++i)
			words[i] = static_cast<uint32_t>(static_cast<int32_t>(static_cast<int8_t>(at[loaded.count * 16 + i])));
		if (const auto problem = VoxelListBase::entries_problem(list.reference_dimensions(), VoxelListBase::Kind::Counting, ids, coordinates, words))
			fail(file, *problem);
		list.assign_entries(ids, coordinates, words);
	}
}
