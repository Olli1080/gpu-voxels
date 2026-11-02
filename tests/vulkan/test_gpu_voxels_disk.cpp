// Slice S7-4: GpuVoxelsMap::write_to_disk / read_from_disk and the map files of MapFile.h. The format is checked against bytes built by hand in this
// file (header, payload, checksum), round trips of every map type restore the whole state, and every kind of damaged file is refused without touching
// the target.
#include "vulkan_fixture.h"

#include <algorithm>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <functional>
#include <iterator>
#include <random>
#include <string>
#include <vector>

#include <gpu_voxels/vulkan/GpuVoxels.h>

#include "facade_support.h"

using namespace gpu_voxels::vulkan;
using gvl_test::snapshot;
using gvl_test::VulkanTest;
namespace fs = std::filesystem;

namespace
{
	constexpr Dim3ui kDims{ 40, 30, 20 };
	constexpr float kSide = 0.1f;
	constexpr std::size_t kHeader = 48;

	struct TempDir
	{
		fs::path path;
		TempDir()
		{
			std::random_device device;
			path = fs::temp_directory_path() / ("gvl_test_mapfile_" + std::to_string(device()) + "_" + std::to_string(device()));
			fs::create_directories(path);
		}
		~TempDir()
		{
			std::error_code ec;
			fs::remove_all(path, ec);
		}
		TempDir(const TempDir&) = delete;
		TempDir& operator=(const TempDir&) = delete;
	};

	std::vector<uint8_t> read_bytes(const fs::path& file)
	{
		std::ifstream in(file, std::ios::binary);
		return std::vector<uint8_t>(std::istreambuf_iterator<char>(in), std::istreambuf_iterator<char>());
	}

	void write_bytes(const fs::path& file, const std::vector<uint8_t>& bytes)
	{
		std::ofstream out(file, std::ios::binary | std::ios::trunc);
		out.write(reinterpret_cast<const char*>(bytes.data()), static_cast<std::streamsize>(bytes.size()));
	}

	/// FNV-1a 64 bit as the format describes it, written out here again.
	uint64_t fnv(const uint8_t* data, std::size_t size)
	{
		uint64_t hash = 14695981039346656037ull;
		for (std::size_t i = 0; i < size; ++i)
		{
			hash ^= data[i];
			hash *= 1099511628211ull;
		}
		return hash;
	}

	void put_u32(std::vector<uint8_t>& out, uint32_t v) { for (int i = 0; i < 4; ++i) out.push_back(static_cast<uint8_t>(v >> (8 * i))); }
	void put_u64(std::vector<uint8_t>& out, uint64_t v) { for (int i = 0; i < 8; ++i) out.push_back(static_cast<uint8_t>(v >> (8 * i))); }
	uint32_t get_u32(const std::vector<uint8_t>& b, std::size_t at) { uint32_t v; std::memcpy(&v, b.data() + at, 4); return v; }
	void set_u32(std::vector<uint8_t>& b, std::size_t at, uint32_t v) { std::memcpy(b.data() + at, &v, 4); }

	void reseal(std::vector<uint8_t>& bytes)
	{
		const uint64_t hash = fnv(bytes.data(), bytes.size() - 8);
		std::memcpy(bytes.data() + bytes.size() - 8, &hash, 8);
	}

	std::vector<uint8_t> header_of(uint32_t type, Dim3ui dims, float side, uint64_t count)
	{
		std::vector<uint8_t> bytes{ 'G', 'V', 'L', 'M', 'A', 'P', 0, 1 };
		put_u32(bytes, 1);
		put_u32(bytes, type);
		put_u32(bytes, dims.x);
		put_u32(bytes, dims.y);
		put_u32(bytes, dims.z);
		uint32_t side_bits;
		std::memcpy(&side_bits, &side, 4);
		put_u32(bytes, side_bits);
		put_u32(bytes, 0);
		put_u32(bytes, 0);
		put_u64(bytes, count);
		return bytes;
	}

	void seal(std::vector<uint8_t>& bytes) { put_u64(bytes, fnv(bytes.data(), bytes.size())); }

	std::vector<Coord3ui> random_coords(std::mt19937_64& rng, int n, Dim3ui dims)
	{
		std::vector<Coord3ui> coordinates;
		for (int i = 0; i < n; ++i)
			coordinates.push_back({ static_cast<uint32_t>(rng() % dims.x), static_cast<uint32_t>(rng() % dims.y), static_cast<uint32_t>(rng() % dims.z) });
		return coordinates;
	}

	std::vector<Point3f> centers(const std::vector<Coord3ui>& coordinates)
	{
		std::vector<Point3f> points;
		for (const auto& c : coordinates)
			points.push_back({ (static_cast<float>(c.x) + 0.5f) * kSide, (static_cast<float>(c.y) + 0.5f) * kSide, (static_cast<float>(c.z) + 0.5f) * kSide });
		return points;
	}

	/// Something in every kind of map or list that depends on `seed`.
	void populate(GpuVoxelsMap& m, uint64_t seed)
	{
		std::mt19937_64 rng(seed);
		switch (m.type())
		{
		case MT_BITVECTOR_VOXELMAP:
		case MT_BITVECTOR_VOXELLIST:
			for (const uint8_t meaning : { uint8_t(3), uint8_t(5), uint8_t(40), uint8_t(255) })
				m.insert_coordinate_list(random_coords(rng, 300, kDims), meaning);
			break;
		case MT_COUNTING_VOXELLIST:
		{
			const auto fixed = random_coords(rng, 20, kDims);
			for (int i = 0; i < 3; ++i)
				m.insert_coordinate_list(fixed, 0);
			m.insert_coordinate_list(random_coords(rng, 300, kDims), 0);
			for (int i = 0; i < 130; ++i)
				m.insert_coordinate_list(std::vector<Coord3ui>{ { 7, 7, 7 } }, 0); // 130 wraps to a negative int8 count
			break;
		}
		case MT_PROBAB_VOXELMAP:
			m.insert_coordinate_list(random_coords(rng, 300, kDims), eBVM_OCCUPIED);
			m.insert_coordinate_list(random_coords(rng, 100, kDims), eBVM_FREE);
			break;
		case MT_DISTANCE_VOXELMAP:
			m.insert_coordinate_list(random_coords(rng, 200, kDims), eBVM_OCCUPIED);
			break;
		case MT_BITVECTOR_OCTREE:
		case MT_PROBAB_OCTREE:
			m.insert_point_cloud(centers(random_coords(rng, 300, kDims)), eBVM_OCCUPIED);
			m.insert_point_cloud(centers(random_coords(rng, 80, kDims)), eBVM_FREE);
			break;
		default:
			ADD_FAILURE() << "type not built";
		}
	}

	/// The class-level read, which throws with the reason.
	void read_direct(GpuVoxelsMap& m, const fs::path& file)
	{
		if (auto* p = m.as<BitVoxelMap>()) read_map_file(file, *p);
		else if (auto* p = m.as<ProbVoxelMap>()) read_map_file(file, *p);
		else if (auto* p = m.as<DistanceVoxelMap>()) read_map_file(file, *p);
		else if (auto* p = m.as<BitVoxelList>()) read_map_file(file, *p);
		else if (auto* p = m.as<CountingVoxelList>()) read_map_file(file, *p);
		else ADD_FAILURE() << "no file format for this map";
	}

	const MapType kFileTypes[] = { MT_BITVECTOR_VOXELMAP, MT_PROBAB_VOXELMAP, MT_DISTANCE_VOXELMAP, MT_BITVECTOR_VOXELLIST, MT_COUNTING_VOXELLIST };
	const MapType kAllTypes[] = { MT_BITVECTOR_VOXELMAP, MT_PROBAB_VOXELMAP, MT_DISTANCE_VOXELMAP, MT_BITVECTOR_VOXELLIST, MT_COUNTING_VOXELLIST,
		MT_BITVECTOR_OCTREE, MT_PROBAB_OCTREE };

	class DiskTest : public VulkanTest
	{
	protected:
		std::unique_ptr<GpuVoxelsMap> make(MapType type, Dim3ui dims = kDims, float side = kSide) { return std::make_unique<GpuVoxelsMap>(ctx(), type, dims, side); }
		TempDir dir;
	};

	void expect_file_error(const std::function<void()>& call, const std::string& text)
	{
		try
		{
			call();
			ADD_FAILURE() << "no exception, expected one with '" << text << "'";
		}
		catch (const MapFileError& e)
		{
			EXPECT_NE(std::string(e.what()).find(text), std::string::npos) << "message was: " << e.what();
		}
	}
}

TEST_F(DiskTest, RoundTripRestoresTheWholeState)
{
	for (const MapType type : kAllTypes)
	{
		auto written = make(type);
		populate(*written, 501);
		const auto expected = snapshot(*written);
		ASSERT_GT(expected.size(), 10u) << map_type_name(type);
		const fs::path file = dir.path / (std::string(map_type_name(type)) + ".bin");
		ASSERT_TRUE(written->write_to_disk(file)) << map_type_name(type);
		EXPECT_EQ(snapshot(*written), expected) << map_type_name(type) << ": writing does not change the map";
		EXPECT_TRUE(fs::exists(file));

		auto fresh = make(type);
		ASSERT_TRUE(fresh->read_from_disk(file)) << map_type_name(type);
		EXPECT_EQ(snapshot(*fresh), expected) << map_type_name(type);

		// A read replaces the content, it does not merge.
		auto other = make(type);
		populate(*other, 502);
		ASSERT_NE(snapshot(*other), expected);
		ASSERT_TRUE(other->read_from_disk(file)) << map_type_name(type);
		EXPECT_EQ(snapshot(*other), expected) << map_type_name(type) << ": replaced";

		// Writing again over the file works, and an empty map round trips.
		ASSERT_TRUE(other->write_to_disk(file));
		auto empty = make(type);
		const auto nothing = snapshot(*empty);
		const fs::path empty_file = dir.path / (std::string("empty_") + map_type_name(type) + ".bin");
		ASSERT_TRUE(empty->write_to_disk(empty_file));
		EXPECT_TRUE(other->read_from_disk(empty_file));
		EXPECT_EQ(snapshot(*other), nothing) << map_type_name(type) << ": the empty map";
	}
}

TEST_F(DiskTest, BitMapBytesAreTheDocumentedFormat)
{
	auto map = make(MT_BITVECTOR_VOXELMAP, { 2, 1, 1 }, 0.5f);
	map->insert_coordinate_list(std::vector<Coord3ui>{ { 1, 0, 0 } }, 5);
	const fs::path file = dir.path / "bit_map.bin";
	ASSERT_TRUE(map->write_to_disk(file));

	std::vector<uint8_t> expected = header_of(MT_BITVECTOR_VOXELMAP, { 2, 1, 1 }, 0.5f, 2);
	for (int w = 0; w < 8; ++w)
		put_u32(expected, 0);       // voxel 0
	put_u32(expected, 1u << 5);     // voxel 1: meaning 5 is bit 5 of word 0
	for (int w = 1; w < 8; ++w)
		put_u32(expected, 0);
	seal(expected);
	EXPECT_EQ(read_bytes(file), expected);
	EXPECT_EQ(expected.size(), kHeader + 64 + 8);
}

TEST_F(DiskTest, ListBytesAreTheDocumentedFormat)
{
	{
		auto list = make(MT_BITVECTOR_VOXELLIST, { 4, 1, 1 }, 0.5f);
		list->insert_coordinate_list(std::vector<Coord3ui>{ { 3, 0, 0 } }, 2);
		list->insert_coordinate_list(std::vector<Coord3ui>{ { 1, 0, 0 } }, 70);
		const fs::path file = dir.path / "bit_list.bin";
		ASSERT_TRUE(list->write_to_disk(file));
		std::vector<uint8_t> expected = header_of(MT_BITVECTOR_VOXELLIST, { 4, 1, 1 }, 0.5f, 2);
		put_u32(expected, 1);
		put_u32(expected, 3);                                                          // ids, ascending
		for (const uint32_t v : { 1u, 0u, 0u, 3u, 0u, 0u })
			put_u32(expected, v);                                                      // coordinates
		for (const uint32_t v : { 0u, 0u, 64u, 0u, 0u, 0u, 0u, 0u })
			put_u32(expected, v);                                                      // entry 0: meaning 70 = word 2, bit 6
		for (const uint32_t v : { 4u, 0u, 0u, 0u, 0u, 0u, 0u, 0u })
			put_u32(expected, v);                                                      // entry 1: meaning 2
		seal(expected);
		EXPECT_EQ(read_bytes(file), expected);
	}
	{
		auto list = make(MT_COUNTING_VOXELLIST, { 4, 1, 1 }, 0.5f);
		list->insert_coordinate_list(std::vector<Coord3ui>{ { 3, 0, 0 } }, 0);
		list->insert_coordinate_list(std::vector<Coord3ui>{ { 3, 0, 0 } }, 0);
		list->insert_coordinate_list(std::vector<Coord3ui>{ { 1, 0, 0 } }, 0);
		const fs::path file = dir.path / "counting_list.bin";
		ASSERT_TRUE(list->write_to_disk(file));
		std::vector<uint8_t> expected = header_of(MT_COUNTING_VOXELLIST, { 4, 1, 1 }, 0.5f, 2);
		put_u32(expected, 1);
		put_u32(expected, 3);
		for (const uint32_t v : { 1u, 0u, 0u, 3u, 0u, 0u })
			put_u32(expected, v);
		expected.push_back(1);
		expected.push_back(2);                                                         // counts (int8)
		seal(expected);
		EXPECT_EQ(read_bytes(file), expected);
	}
	{
		// Prob. and distance map: the payload is what the class downloads.
		auto prob = make(MT_PROBAB_VOXELMAP, { 2, 1, 1 }, 0.5f);
		prob->insert_coordinate_list(std::vector<Coord3ui>{ { 1, 0, 0 } }, eBVM_OCCUPIED);
		const fs::path pfile = dir.path / "prob.bin";
		ASSERT_TRUE(prob->write_to_disk(pfile));
		const auto occupancy = prob->as<ProbVoxelMap>()->download_occupancy();
		std::vector<uint8_t> expected = header_of(MT_PROBAB_VOXELMAP, { 2, 1, 1 }, 0.5f, 2);
		for (const int8_t v : occupancy)
			expected.push_back(static_cast<uint8_t>(v));
		seal(expected);
		EXPECT_EQ(read_bytes(pfile), expected);
		EXPECT_EQ(occupancy[0], -128) << "an untouched voxel is unknown";

		auto distance = make(MT_DISTANCE_VOXELMAP, { 2, 1, 1 }, 0.5f);
		distance->insert_coordinate_list(std::vector<Coord3ui>{ { 1, 0, 0 } }, eBVM_OCCUPIED);
		const fs::path dfile = dir.path / "distance.bin";
		ASSERT_TRUE(distance->write_to_disk(dfile));
		expected = header_of(MT_DISTANCE_VOXELMAP, { 2, 1, 1 }, 0.5f, 2);
		for (const uint32_t w : distance->as<DistanceVoxelMap>()->download_words())
			put_u32(expected, w);
		seal(expected);
		EXPECT_EQ(read_bytes(dfile), expected);
	}
}

TEST_F(DiskTest, DamagedFilesAreRefusedAndChangeNothing)
{
	for (const MapType type : kFileTypes)
	{
		const bool is_list = type == MT_BITVECTOR_VOXELLIST || type == MT_COUNTING_VOXELLIST;
		auto source = make(type);
		populate(*source, 503);
		const fs::path good = dir.path / "good.bin";
		ASSERT_TRUE(source->write_to_disk(good));
		const std::vector<uint8_t> bytes = read_bytes(good);
		ASSERT_GT(bytes.size(), kHeader + 8 + 100);

		auto target = make(type);
		populate(*target, 504);
		const auto before = snapshot(*target);
		const fs::path bad = dir.path / "bad.bin";

		struct Damage
		{
			std::string name;
			std::function<void(std::vector<uint8_t>&)> apply;
			std::string text;
		};
		const std::string count_text = is_list ? "the header announces" : "corrupt header (the count of voxels";
		const std::vector<Damage> damages = {
			{ "empty", [](auto& b) { b.clear(); }, "too short" },
			{ "47 bytes", [](auto& b) { b.resize(47); }, "too short" },
			{ "header only", [](auto& b) { b.resize(kHeader); }, "too short" },
			{ "cut inside the payload", [](auto& b) { b.resize(b.size() / 2); }, "truncated file?" },
			{ "last byte missing", [](auto& b) { b.pop_back(); }, "truncated file?" },
			{ "byte appended", [](auto& b) { b.push_back(0); }, "data after the end?" },
			{ "magic", [](auto& b) { b[0] ^= 1; }, "wrong magic" },
			{ "magic version byte", [](auto& b) { b[7] = 2; }, "wrong magic" },
			{ "version", [](auto& b) { set_u32(b, 8, 2); }, "unsupported version 2" },
			{ "type", [](auto& b) { set_u32(b, 12, get_u32(b, 12) + 1); }, "the file holds the map type" },
			{ "dimension x", [](auto& b) { set_u32(b, 16, get_u32(b, 16) + 1); }, "the dimensions of the file" },
			{ "dimension y", [](auto& b) { set_u32(b, 20, get_u32(b, 20) + 1); }, "the dimensions of the file" },
			{ "dimension z", [](auto& b) { set_u32(b, 24, get_u32(b, 24) + 1); }, "the dimensions of the file" },
			{ "side length", [](auto& b) { set_u32(b, 28, get_u32(b, 28) ^ 1u); }, "the voxel side length of the file" },
			{ "reserved 1", [](auto& b) { set_u32(b, 32, 1); }, "reserved fields" },
			{ "reserved 2", [](auto& b) { set_u32(b, 36, 1); }, "reserved fields" },
			{ "count", [](auto& b) { set_u32(b, 40, get_u32(b, 40) + 1); }, count_text },
			{ "count high word", [](auto& b) { set_u32(b, 44, 1); }, is_list ? "corrupt header (the count of entries" : "corrupt header (the count of voxels" },
			{ "first payload byte", [](auto& b) { b[kHeader] ^= 0x10; }, "checksum mismatch" },
			{ "last payload byte", [](auto& b) { b[b.size() - 9] ^= 0x01; }, "checksum mismatch" },
			{ "checksum", [](auto& b) { b.back() ^= 0x80; }, "checksum mismatch" },
		};
		for (const auto& damage : damages)
		{
			std::vector<uint8_t> broken = bytes;
			damage.apply(broken);
			write_bytes(bad, broken);
			EXPECT_FALSE(target->read_from_disk(bad)) << map_type_name(type) << ": " << damage.name;
			EXPECT_EQ(snapshot(*target), before) << map_type_name(type) << ": " << damage.name << " changed the target";
			expect_file_error([&] { read_direct(*target, bad); }, damage.text);
			EXPECT_EQ(snapshot(*target), before);
		}

		// Valid checksum, but the content is no map or list of this size.
		struct Forged
		{
			std::string name;
			std::function<void(std::vector<uint8_t>&)> apply;
			std::string text;
		};
		std::vector<Forged> forged;
		if (is_list)
		{
			const uint32_t n = get_u32(bytes, 40);
			ASSERT_GT(n, 2u);
			const std::size_t ids = kHeader, coords = kHeader + 4u * n;
			forged.push_back({ "ids and coordinates of entries 0 and 1 swapped", [=](auto& b)
				{
					const uint32_t id0 = get_u32(b, ids), id1 = get_u32(b, ids + 4);
					set_u32(b, ids, id1);
					set_u32(b, ids + 4, id0);
					for (int a = 0; a < 3; ++a)
					{
						const uint32_t c0 = get_u32(b, coords + 4u * a), c1 = get_u32(b, coords + 12 + 4u * a);
						set_u32(b, coords + 4u * a, c1);
						set_u32(b, coords + 12 + 4u * a, c0);
					}
				}, "the ids are not strictly ascending at entry 1" });
			forged.push_back({ "id of entry 0 changed", [=](auto& b) { set_u32(b, ids, get_u32(b, ids) + 1); }, "the id of entry 0 is not the index of its coordinates" });
			forged.push_back({ "x of entry 0 outside", [=](auto& b) { set_u32(b, coords, kDims.x); }, "entry 0 lies outside the reference map" });
			forged.push_back({ "y of entry 1 outside", [=](auto& b) { set_u32(b, coords + 12 + 4, kDims.y); }, "entry 1 lies outside the reference map" });
			forged.push_back({ "z of entry 2 outside", [=](auto& b) { set_u32(b, coords + 24 + 8, kDims.z); }, "entry 2 lies outside the reference map" });
			forged.push_back({ "x of entry 2 far outside", [=](auto& b) { set_u32(b, coords + 24, 0xFFFFFFFFu); }, "entry 2 lies outside the reference map" });
			forged.push_back({ "equal ids", [=](auto& b)
				{
					for (std::size_t k = 0; k < 4; ++k)
						b[ids + 4 + k] = b[ids + k];
					for (std::size_t k = 0; k < 12; ++k)
						b[coords + 12 + k] = b[coords + k];
				}, "the ids are not strictly ascending at entry 1" });
		}
		if (type == MT_DISTANCE_VOXELMAP)
		{
			forged.push_back({ "bit 30 of voxel 0", [](auto& b) { set_u32(b, kHeader, get_u32(b, kHeader) | (1u << 30)); }, "voxel 0 has bits above bit 29" });
			forged.push_back({ "bit 31 of voxel 3", [](auto& b) { set_u32(b, kHeader + 12, get_u32(b, kHeader + 12) | (1u << 31)); }, "voxel 3 has bits above bit 29" });
		}
		for (const auto& f : forged)
		{
			std::vector<uint8_t> broken = bytes;
			f.apply(broken);
			reseal(broken);
			write_bytes(bad, broken);
			EXPECT_FALSE(target->read_from_disk(bad)) << map_type_name(type) << ": " << f.name;
			EXPECT_EQ(snapshot(*target), before) << map_type_name(type) << ": " << f.name << " changed the target";
			expect_file_error([&] { read_direct(*target, bad); }, f.text);
		}
	}
}

TEST_F(DiskTest, FilesOfAnotherTargetAreRefused)
{
	const fs::path file = dir.path / "file.bin";
	for (const MapType written_type : kFileTypes)
	{
		auto written = make(written_type);
		populate(*written, 505);
		ASSERT_TRUE(written->write_to_disk(file));
		for (const MapType target_type : kFileTypes)
		{
			if (target_type == written_type)
				continue;
			auto target = make(target_type);
			populate(*target, 506);
			const auto before = snapshot(*target);
			EXPECT_FALSE(target->read_from_disk(file)) << map_type_name(written_type) << " file into " << map_type_name(target_type);
			EXPECT_EQ(snapshot(*target), before);
			expect_file_error([&] { read_direct(*target, file); }, std::string("the target is ") + map_type_name(target_type));
		}
		// Other dimensions or voxel side length.
		for (const Dim3ui dims : { Dim3ui{ 41, 30, 20 }, Dim3ui{ 40, 29, 20 }, Dim3ui{ 40, 30, 21 } })
		{
			auto target = make(written_type, dims);
			const auto before = snapshot(*target);
			EXPECT_FALSE(target->read_from_disk(file)) << map_type_name(written_type) << " other dimensions";
			EXPECT_EQ(snapshot(*target), before);
			expect_file_error([&] { read_direct(*target, file); }, "the dimensions of the file (40 x 30 x 20) differ");
		}
		auto other_side = make(written_type, kDims, 0.2f);
		const auto before = snapshot(*other_side);
		EXPECT_FALSE(other_side->read_from_disk(file)) << map_type_name(written_type) << " other side length";
		EXPECT_EQ(snapshot(*other_side), before);
		expect_file_error([&] { read_direct(*other_side, file); }, "the voxel side length of the file");
	}
}

TEST_F(DiskTest, FilesThatCannotBeOpenedOrWritten)
{
	for (const MapType type : kAllTypes)
	{
		auto map = make(type);
		populate(*map, 507);
		const auto before = snapshot(*map);
		EXPECT_FALSE(map->read_from_disk(dir.path / "does_not_exist.bin")) << map_type_name(type);
		EXPECT_EQ(snapshot(*map), before);
		EXPECT_FALSE(map->write_to_disk(dir.path)) << map_type_name(type) << ": a directory is no file";
		EXPECT_FALSE(map->write_to_disk(dir.path / "no_such_directory" / "map.bin")) << map_type_name(type);
		EXPECT_FALSE(fs::exists(dir.path / "no_such_directory"));
		EXPECT_EQ(snapshot(*map), before);
	}
	auto map = make(MT_BITVECTOR_VOXELMAP);
	expect_file_error([&] { read_direct(*map, dir.path / "does_not_exist.bin"); }, "could not open the file");
	expect_file_error([&] { write_map_file(dir.path, *map->as<BitVoxelMap>()); }, "could not open the file for writing");
}

TEST_F(DiskTest, OctreeFiles)
{
	// Octrees use their own format: files of other maps and of the other kind are refused, damaged files change nothing.
	auto det = make(MT_BITVECTOR_OCTREE);
	populate(*det, 508);
	const fs::path det_file = dir.path / "det.bin";
	ASSERT_TRUE(det->write_to_disk(det_file));
	auto prob = make(MT_PROBAB_OCTREE);
	populate(*prob, 509);
	const auto prob_before = snapshot(*prob);
	EXPECT_FALSE(prob->read_from_disk(det_file)) << "a deterministic tree file into a probabilistic tree";
	EXPECT_EQ(snapshot(*prob), prob_before);

	auto bit_map = make(MT_BITVECTOR_VOXELMAP);
	populate(*bit_map, 510);
	const fs::path map_file = dir.path / "map.bin";
	ASSERT_TRUE(bit_map->write_to_disk(map_file));
	const auto det_before = snapshot(*det);
	EXPECT_FALSE(det->read_from_disk(map_file)) << "a bit map file into an octree";
	EXPECT_EQ(snapshot(*det), det_before);
	const auto map_before = snapshot(*bit_map);
	EXPECT_FALSE(bit_map->read_from_disk(det_file)) << "an octree file into a bit map";
	EXPECT_EQ(snapshot(*bit_map), map_before);

	auto bytes = read_bytes(det_file);
	ASSERT_GT(bytes.size(), 100u);
	const fs::path bad = dir.path / "bad_tree.bin";
	auto restored = make(MT_BITVECTOR_OCTREE);
	populate(*restored, 511);
	const auto restored_before = snapshot(*restored);
	for (const std::size_t cut : { std::size_t(0), std::size_t(10), std::size_t(31), bytes.size() / 2, bytes.size() - 1 })
	{
		write_bytes(bad, std::vector<uint8_t>(bytes.begin(), bytes.begin() + static_cast<std::ptrdiff_t>(cut)));
		EXPECT_FALSE(restored->read_from_disk(bad)) << "cut at " << cut;
		EXPECT_EQ(snapshot(*restored), restored_before) << "cut at " << cut;
	}
	auto flipped = bytes;
	flipped[bytes.size() / 2] ^= 0x04;
	write_bytes(bad, flipped);
	EXPECT_FALSE(restored->read_from_disk(bad));
	EXPECT_EQ(snapshot(*restored), restored_before);
	ASSERT_TRUE(restored->read_from_disk(det_file)) << "the good file still reads";
	EXPECT_EQ(snapshot(*restored), snapshot(*det));
}

TEST_F(DiskTest, UploadsRefuseTheWrongSize)
{
	const auto expect_vulkan_error = [](const std::function<void()>& call, const std::string& text)
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
	};
	{
		BitVoxelMap map(ctx(), { 2, 1, 1 }, 0.5f);
		std::vector<uint32_t> words(16, 0);
		words[8] = 0x20u;
		map.upload_words(words);
		EXPECT_EQ(map.download_words(), words);
		expect_vulkan_error([&] { map.upload_words(std::vector<uint32_t>(15, 1)); }, "BitVoxelMap::upload_words: expected 16 words, got 15");
		expect_vulkan_error([&] { map.upload_words(std::vector<uint32_t>(17, 1)); }, "BitVoxelMap::upload_words: expected 16 words, got 17");
		EXPECT_EQ(map.download_words(), words) << "refused uploads change nothing";
	}
	{
		ProbVoxelMap map(ctx(), { 2, 1, 1 }, 0.5f);
		map.upload_occupancy(std::vector<int8_t>{ -128, 5 });
		EXPECT_EQ(map.download_occupancy(), (std::vector<int8_t>{ -128, 5 }));
		map.upload_occupancy(std::vector<int8_t>{ 127, -1 });
		EXPECT_EQ(map.download_occupancy(), (std::vector<int8_t>{ 127, -1 })) << "negative values keep their sign";
		EXPECT_EQ(map.voxel_buffer().download_vector<uint32_t>(2), (std::vector<uint32_t>{ 127u, 0xFFFFFFFFu })) << "the device words are sign extended";
		expect_vulkan_error([&] { map.upload_occupancy(std::vector<int8_t>{ 1 }); }, "ProbVoxelMap::upload_occupancy: expected 2 values, got 1");
		expect_vulkan_error([&] { map.upload_occupancy(std::vector<int8_t>{ 1, 2, 3 }); }, "ProbVoxelMap::upload_occupancy: expected 2 values, got 3");
		EXPECT_EQ(map.download_occupancy(), (std::vector<int8_t>{ 127, -1 }));
	}
	{
		DistanceVoxelMap map(ctx(), { 2, 1, 1 }, 0.5f);
		const std::vector<uint32_t> words{ DistanceVoxelMap::pack({ 1, 2, 3 }), DistanceVoxelMap::pack({ 1023, 0, 1023 }) };
		map.upload_words(words);
		EXPECT_EQ(map.download_words(), words);
		expect_vulkan_error([&] { map.upload_words(std::vector<uint32_t>{ 1 }); }, "DistanceVoxelMap::upload_words: expected 2 words, got 1");
		expect_vulkan_error([&] { map.upload_words(std::vector<uint32_t>{ 1, 2, 3 }); }, "DistanceVoxelMap::upload_words: expected 2 words, got 3");
		expect_vulkan_error([&] { map.upload_words(std::vector<uint32_t>{ 1, 1u << 30 }); }, "a word has bits above bit 29");
		expect_vulkan_error([&] { map.upload_words(std::vector<uint32_t>{ 1u << 31, 1 }); }, "a word has bits above bit 29");
		EXPECT_EQ(map.download_words(), words);
	}
}
