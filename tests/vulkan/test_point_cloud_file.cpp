// Slice S7-2: the host readers (.xyz, .pcd, .binvox), the load options, the box of voxels and the façade calls insert_point_cloud_from_file /
// insert_box_into_map. The readers are checked against the model files of the repository with values that an independent Python decoder computed
// (count, sums, extremes, first and last point) and against files written by the tests byte by byte; the façade calls against the classes used directly.
#include "vulkan_fixture.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <limits>
#include <memory>
#include <string>
#include <tuple>
#include <vector>

#include <gpu_voxels/vulkan/GpuVoxels.h>
#include <gpu_voxels/vulkan/PointCloudFile.h>

using namespace gpu_voxels::vulkan;
using gvl_test::VulkanTest;
namespace fs = std::filesystem;

namespace
{
	const fs::path kModels = GVL_TEST_MODEL_DIR;

	struct Stats
	{
		size_t count;
		double sum[3];
		float min[3], max[3];
		Point3f first, last;
	};

	void expect_stats(const std::vector<Point3f>& pts, const Stats& s, double sum_tolerance)
	{
		ASSERT_EQ(pts.size(), s.count);
		double sum[3] = {};
		float mn[3] = { 1e30f, 1e30f, 1e30f }, mx[3] = { -1e30f, -1e30f, -1e30f };
		for (const Point3f& p : pts)
		{
			const float c[3] = { p.x, p.y, p.z };
			for (int a = 0; a < 3; ++a)
			{
				sum[a] += c[a];
				mn[a] = std::min(mn[a], c[a]);
				mx[a] = std::max(mx[a], c[a]);
			}
		}
		for (int a = 0; a < 3; ++a)
		{
			EXPECT_NEAR(sum[a], s.sum[a], sum_tolerance) << "sum " << a;
			EXPECT_NEAR(mn[a], s.min[a], 1e-5) << "min " << a;
			EXPECT_NEAR(mx[a], s.max[a], 1e-5) << "max " << a;
		}
		EXPECT_NEAR(pts.front().x, s.first.x, 1e-5);
		EXPECT_NEAR(pts.front().y, s.first.y, 1e-5);
		EXPECT_NEAR(pts.front().z, s.first.z, 1e-5);
		EXPECT_NEAR(pts.back().x, s.last.x, 1e-5);
		EXPECT_NEAR(pts.back().y, s.last.y, 1e-5);
		EXPECT_NEAR(pts.back().z, s.last.z, 1e-5);
	}

	/// A directory for the files of one test, removed at the end.
	struct TempDir
	{
		fs::path path;
		TempDir()
		{
			path = fs::temp_directory_path() / ("gvl_test_pcf_" + std::to_string(std::rand()) + "_" + std::to_string(reinterpret_cast<uintptr_t>(this)));
			fs::create_directories(path);
		}
		~TempDir() { std::error_code ec; fs::remove_all(path, ec); }
		fs::path write(const std::string& name, const std::string& bytes) const
		{
			const fs::path p = path / name;
			std::ofstream(p, std::ios::binary).write(bytes.data(), static_cast<std::streamsize>(bytes.size()));
			return p;
		}
	};

	template<class T>
	void put(std::string& bytes, T value) { bytes.append(reinterpret_cast<const char*>(&value), sizeof(T)); }

	bool eq(const Point3f& a, const Point3f& b) { return a.x == b.x && a.y == b.y && a.z == b.z; }

	template<class F>
	void expect_file_error(F&& f, const std::string& contains)
	{
		try
		{
			f();
			ADD_FAILURE() << "no exception, expected one with '" << contains << "'";
		}
		catch (const PointCloudFileError& e)
		{
			EXPECT_NE(std::string(e.what()).find(contains), std::string::npos) << e.what();
		}
	}
}

// ---------------------------------------------------------------- readers: the model files

TEST(PointCloudFiles, XyzModelFile)
{
	const fs::path file = kModels / "hollie" / "arm_0_link.xyz";
	if (!fs::exists(file))
		GTEST_SKIP() << "model file missing";
	expect_stats(read_xyz_file(file), { 962, { -0.00033423394, -0.00033407837, 28.060018392 }, { -0.0650003457f, -0.0600003481f, -0.0099999808f },
		{ 0.0649996555f, 0.0599996507f, 0.0600000203f }, { -0.0150003473f, -0.040000346f, -0.0099999808f }, { 0.0149996525f, 0.0599996507f, 0.0600000203f } }, 1e-3);
}

TEST(PointCloudFiles, PcdModelFiles)
{
	const fs::path binary = kModels / "hollie_from_pointcloud2.pcd", ascii = kModels / "robot4cmRes.pcd", intensity = kModels / "pointcloud_0002.pcd";
	if (!fs::exists(binary) || !fs::exists(ascii) || !fs::exists(intensity))
		GTEST_SKIP() << "model files missing";
	// binary, fields x y z rgba
	expect_stats(read_pcd_file(binary), { 3151, { 16396.00094986, -9589.16827583, 38.49295912 }, { 4.9240456f, -3.9466028f, -0.6224322f },
		{ 5.6750069f, -1.9466724f, 0.9467092f }, { 5.1761179f, -2.8084764f, 0.1338043f }, { 5.2185688f, -2.7808518f, 0.1440473f } }, 1e-2);
	// ascii, x y z
	expect_stats(read_pcd_file(ascii), { 139995, { 71073.7282117, 84072.5983358, 67426.5602252 }, { -0.04f, -0.04f, -0.04f }, { 1.08f, 1.16f, 1.48f },
		{ 0.32000002f, 0.11999997f, -0.04f }, { 0.56000012f, 0.72f, 1.48f } }, 0.1);
	// ascii with a fourth column (intensity)
	const auto points = read_pcd_file(intensity);
	ASSERT_EQ(points.size(), 262897u);
	EXPECT_NEAR(points.front().x, -0.45265159f, 1e-6);
	EXPECT_NEAR(points.front().y, 0.59387434f, 1e-6);
	EXPECT_NEAR(points.front().z, -0.088303901f, 1e-6);
}

TEST(PointCloudFiles, BinvoxModelFile)
{
	const fs::path file = kModels / "coordinate_system_100.binvox";
	if (!fs::exists(file))
		GTEST_SKIP() << "model file missing";
	expect_stats(read_binvox_file(file), { 11777, { 2932.733095, 2658.7757575, 3100.52586 }, { -0.05f, -0.05f, -0.05f }, { 1.317345f, 1.317345f, 1.317345f },
		{ -0.05f, 0.00581f, -0.022095f }, { 1.317345f, 0.033715f, 0.312765f } }, 0.05);
}

// ---------------------------------------------------------------- readers: files made here

TEST(PointCloudFiles, XyzSyntax)
{
	const TempDir dir;
	const fs::path file = dir.write("a.xyz",
		"x y z\n"                      // a header: skipped
		"# a comment\n"
		"\n"
		"1 2 3\n"
		"  4.5\t-5.5   6.5  \r\n"      // tabs, CRLF
		"+7 +8 +9\n"                   // plus signs
		"1e-1 2E+1 -3.5e0\n"           // exponents
		"10 11 12 0.5 0.5 0.5\n"       // further columns are not a second point
		"13 14\n"                      // too short
		"15 16 abc\n"                  // not a number
		"abc 17 18\n"
		"19 20 21");                   // no end of line
	const std::vector<Point3f> expected = { { 1, 2, 3 }, { 4.5f, -5.5f, 6.5f }, { 7, 8, 9 }, { 0.1f, 20.0f, -3.5f }, { 10, 11, 12 }, { 19, 20, 21 } };
	const auto points = read_xyz_file(file);
	ASSERT_EQ(points.size(), expected.size());
	for (size_t i = 0; i < points.size(); ++i)
		EXPECT_TRUE(eq(points[i], expected[i])) << i;
	EXPECT_TRUE(read_xyz_file(dir.write("empty.xyz", "")).empty());
	expect_file_error([&] { (void)read_xyz_file(dir.path / "missing.xyz"); }, "could not open");
}

TEST(PointCloudFiles, PcdSyntheticAscii)
{
	const TempDir dir;
	// fields in another order, a field with count 3 before z, the data lines hold more than the header says
	const fs::path file = dir.write("a.pcd",
		"# .PCD v0.7\nVERSION 0.7\nFIELDS intensity y normal z x\nSIZE 4 4 4 4 4\nTYPE F F F F F\nCOUNT 1 1 3 1 1\nWIDTH 2\nHEIGHT 1\nPOINTS 2\nDATA ascii\n"
		"9 2 0 0 0 3 1\n"
		"9 5 0 0 0 6 4\r\n"
		"9 8 0 0 0 9 7\n");
	const auto points = read_pcd_file(file);
	ASSERT_EQ(points.size(), 2u);
	EXPECT_TRUE(eq(points[0], { 1, 2, 3 }));
	EXPECT_TRUE(eq(points[1], { 4, 5, 6 }));
	// POINTS missing: WIDTH * HEIGHT; double precision fields; NaN is kept
	const fs::path doubles = dir.write("d.pcd", "FIELDS x y z\nSIZE 8 8 8\nTYPE F F F\nCOUNT 1 1 1\nWIDTH 2\nHEIGHT 2\nDATA ascii\n1.5 2.5 3.5\nnan 0 0\n0 0 0\n7 8 9\n");
	const auto d = read_pcd_file(doubles);
	ASSERT_EQ(d.size(), 4u);
	EXPECT_TRUE(eq(d[0], { 1.5f, 2.5f, 3.5f }));
	EXPECT_TRUE(std::isnan(d[1].x));
	EXPECT_TRUE(eq(d[3], { 7, 8, 9 }));
	// COUNT missing means 1 for every field
	EXPECT_EQ(read_pcd_file(dir.write("c.pcd", "FIELDS x y z\nSIZE 4 4 4\nTYPE F F F\nPOINTS 1\nDATA ascii\n1 2 3\n")).size(), 1u);

	const std::string head = "FIELDS x y z\nSIZE 4 4 4\nTYPE F F F\nCOUNT 1 1 1\n";
	expect_file_error([&] { (void)read_pcd_file(dir.write("e1.pcd", head + "POINTS 1\n1 2 3\n")); }, "DATA");
	expect_file_error([&] { (void)read_pcd_file(dir.write("e2.pcd", "SIZE 4 4 4\nTYPE F F F\nPOINTS 1\nDATA ascii\n1 2 3\n")); }, "has no FIELDS");
	expect_file_error([&] { (void)read_pcd_file(dir.write("e3.pcd", "FIELDS x y w\nSIZE 4 4 4\nTYPE F F F\nPOINTS 1\nDATA ascii\n1 2 3\n")); }, "field z");
	expect_file_error([&] { (void)read_pcd_file(dir.write("e4.pcd", "FIELDS x y z\nSIZE 4 4 4\nTYPE F F U\nPOINTS 1\nDATA ascii\n1 2 3\n")); }, "float");
	expect_file_error([&] { (void)read_pcd_file(dir.write("e5.pcd", "FIELDS x y z\nSIZE 2 4 4\nTYPE F F F\nPOINTS 1\nDATA ascii\n1 2 3\n")); }, "float");
	expect_file_error([&] { (void)read_pcd_file(dir.write("e6.pcd", "FIELDS x y z\nSIZE 4 4\nTYPE F F F\nPOINTS 1\nDATA ascii\n1 2 3\n")); }, "different numbers");
	expect_file_error([&] { (void)read_pcd_file(dir.write("e7.pcd", head + "POINTS 3\nDATA ascii\n1 2 3\n4 5 6\n")); }, "holds 2 points");
	expect_file_error([&] { (void)read_pcd_file(dir.write("e8.pcd", head + "POINTS 1\nDATA ascii\n1 2\n")); }, "too few columns");
	expect_file_error([&] { (void)read_pcd_file(dir.write("e9.pcd", head + "POINTS 1\nDATA ascii\n1 2 x\n")); }, "not a number");
	expect_file_error([&] { (void)read_pcd_file(dir.write("e10.pcd", head + "POINTS 1\nDATA binary_compressed\nxxxxxxxxxxxx")); }, "binary_compressed");
	expect_file_error([&] { (void)read_pcd_file(dir.write("e11.pcd", head + "DATA ascii\n1 2 3\n")); }, "POINTS");
	expect_file_error([&] { (void)read_pcd_file(dir.write("e12.pcd", "FIELDS x y z\nSIZE 4 4 4\nTYPE F F F\nCOUNT 1 0 1\nPOINTS 1\nDATA ascii\n1 2 3\n")); }, "COUNT");
	expect_file_error([&] { (void)read_pcd_file(dir.write("e13.pcd", "FIELDS x y z\nSIZE 4 4 4\nTYPE F F F\nCOUNT 2 1 1\nPOINTS 1\nDATA ascii\n1 2 3 4\n")); }, "float");
	// no HEIGHT: the cloud has one row
	EXPECT_EQ(read_pcd_file(dir.write("w.pcd", "FIELDS x y z\nSIZE 4 4 4\nTYPE F F F\nWIDTH 2\nDATA ascii\n1 2 3\n4 5 6\n")).size(), 2u);
	expect_file_error([&] { (void)read_pcd_file(dir.path / "missing.pcd"); }, "could not open");
}

TEST(PointCloudFiles, PcdSyntheticBinary)
{
	const TempDir dir;
	// x y z as float, then an int field, in front of them a field of 3 bytes (count 3, size 1)
	std::string bytes = "FIELDS pad x y z id\nSIZE 1 4 4 4 4\nTYPE U F F F U\nCOUNT 3 1 1 1 1\nWIDTH 3\nHEIGHT 1\nPOINTS 3\nDATA binary\n";
	const Point3f in[3] = { { 1, 2, 3 }, { -4.5f, 5.5f, -6.5f }, { 1e10f, 0, -1e-10f } };
	for (int i = 0; i < 3; ++i)
	{
		bytes += "abc";
		put(bytes, in[i].x);
		put(bytes, in[i].y);
		put(bytes, in[i].z);
		put<uint32_t>(bytes, 100 + i);
	}
	const auto points = read_pcd_file(dir.write("b.pcd", bytes));
	ASSERT_EQ(points.size(), 3u);
	for (int i = 0; i < 3; ++i)
		EXPECT_TRUE(eq(points[i], in[i])) << i;
	// doubles, header with CRLF
	std::string d = "FIELDS x y z\r\nSIZE 8 8 8\r\nTYPE F F F\r\nCOUNT 1 1 1\r\nPOINTS 1\r\nDATA binary\n";
	put(d, 0.25);
	put(d, -0.5);
	put(d, 1.0e300);
	const auto dp = read_pcd_file(dir.write("d.pcd", d));
	ASSERT_EQ(dp.size(), 1u);
	EXPECT_EQ(dp[0].x, 0.25f);
	EXPECT_EQ(dp[0].y, -0.5f);
	EXPECT_TRUE(std::isinf(dp[0].z) && dp[0].z > 0); // 1e300 does not fit a float
	std::string neg = "FIELDS x y z\nSIZE 8 8 8\nTYPE F F F\nPOINTS 1\nDATA binary\n";
	put(neg, 0.0);
	put(neg, -1.0e300);
	put(neg, 1.0e-300);
	const auto np = read_pcd_file(dir.write("n.pcd", neg));
	ASSERT_EQ(np.size(), 1u);
	EXPECT_TRUE(std::isinf(np[0].y) && np[0].y < 0);
	EXPECT_EQ(np[0].z, 0.0f); // too small for a float: 0, not an error
	// truncated: one byte short
	std::string cut = bytes;
	cut.pop_back();
	expect_file_error([&] { (void)read_pcd_file(dir.write("t.pcd", cut)); }, "fewer than the 3 points");
	// the count may not overflow the size check
	expect_file_error([&] { (void)read_pcd_file(dir.write("h.pcd", "FIELDS x y z\nSIZE 4 4 4\nTYPE F F F\nPOINTS 18446744073709551615\nDATA binary\nxxxx")); }, "fewer than");
}

namespace
{
	/// A binvox file: header, then (value, count) pairs.
	std::string binvox(const std::string& header, const std::vector<std::pair<int, int>>& runs)
	{
		std::string s = header + "data\n";
		for (const auto& [value, count] : runs)
		{
			s.push_back(static_cast<char>(value));
			s.push_back(static_cast<char>(count));
		}
		return s;
	}
}

TEST(PointCloudFiles, BinvoxSynthetic)
{
	const TempDir dir;
	// dim depth height width = 2 3 4: index = (x * 3 + z) * 4 + y; scale 8 / width 4 = 2 per voxel
	const std::string header = "#binvox 1\ndim 2 3 4\ntranslate 10 20 30\nscale 8\n";
	// voxels 0, 5, 6, 7 (a run of 3), 23 are filled
	const auto points = read_binvox_file(dir.write("a.binvox", binvox(header, { { 1, 1 }, { 0, 4 }, { 1, 3 }, { 0, 15 }, { 1, 1 } })));
	auto at = [](int x, int y, int z) { return Point3f{ 2.0f * x + 10, 2.0f * y + 20, 2.0f * z + 30 }; };
	const std::vector<Point3f> expected = { at(0, 0, 0), at(0, 1, 1), at(0, 2, 1), at(0, 3, 1), at(1, 3, 2) };
	ASSERT_EQ(points.size(), expected.size());
	for (size_t i = 0; i < points.size(); ++i)
		EXPECT_TRUE(eq(points[i], expected[i])) << i << ": " << points[i].x << " " << points[i].y << " " << points[i].z;
	// keywords in another order (scale before dim: the width of the dim line counts), an unknown keyword is skipped, value 2 is not filled
	const auto other = read_binvox_file(dir.write("b.binvox", binvox("#binvox 2\nscale 8\ntranslate 10 20 30\ndim 2 3 4\nextra dim 9 9 9\n",
		{ { 2, 1 }, { 0, 3 }, { 1, 1 }, { 0, 18 }, { 1, 1 } })));
	ASSERT_EQ(other.size(), 2u);
	EXPECT_TRUE(eq(other[0], at(0, 0, 1)));
	EXPECT_TRUE(eq(other[1], at(1, 3, 2)));
	// no scale and no translate: one voxel per unit... scale 1 / width
	const auto plain = read_binvox_file(dir.write("c.binvox", binvox("#binvox 1\ndim 1 1 4\n", { { 0, 3 }, { 1, 1 } })));
	ASSERT_EQ(plain.size(), 1u);
	EXPECT_TRUE(eq(plain[0], { 0.0f, 0.75f, 0.0f }));
	EXPECT_TRUE(read_binvox_file(dir.write("e.binvox", binvox("#binvox 1\ndim 1 1 1\n", { { 0, 1 } }))).empty());

	expect_file_error([&] { (void)read_binvox_file(dir.write("e1.binvox", binvox("#binvax 1\ndim 1 1 1\n", { { 1, 1 } }))); }, "#binvox");
	expect_file_error([&] { (void)read_binvox_file(dir.write("e2.binvox", binvox("#binvox 1\ntranslate 0 0 0\n", { { 1, 1 } }))); }, "no dim line");
	expect_file_error([&] { (void)read_binvox_file(dir.write("e3.binvox", "#binvox 1\ndim 1 1 1\n")); }, "data");
	expect_file_error([&] { (void)read_binvox_file(dir.write("e4.binvox", binvox("#binvox 1\ndim 2 2 2\n", { { 1, 4 } }))); }, "ends at voxel 4 of 8");
	expect_file_error([&] { (void)read_binvox_file(dir.write("e5.binvox", binvox("#binvox 1\ndim 2 2 2\n", { { 1, 5 }, { 0, 4 } }))); }, "beyond");
	expect_file_error([&] { (void)read_binvox_file(dir.write("e6.binvox", binvox("#binvox 1\ndim 0 2 2\n", {}))); }, "positive");
	expect_file_error([&] { (void)read_binvox_file(dir.write("e8.binvox", binvox("#binvox 1\ndim 2 2 0\n", {}))); }, "positive");
	expect_file_error([&] { (void)read_binvox_file(dir.write("e9.binvox", binvox("#binvox 1\ndim 2 0 2\n", {}))); }, "positive");
	expect_file_error([&] { (void)read_binvox_file(dir.write("e7.binvox", binvox("#binvox 1\ndim 2 two 2\n", {}))); }, "bad number");
	expect_file_error([&] { (void)read_binvox_file(dir.path / "missing.binvox"); }, "could not open");
}

TEST(PointCloudFiles, ByExtensionAndLoadOptions)
{
	const TempDir dir;
	const fs::path upper = dir.write("UP.XYZ", "1 2 3\n4 5 6\n");
	EXPECT_EQ(read_point_cloud_file(upper).size(), 2u); // the extension is not case sensitive
	expect_file_error([&] { (void)read_point_cloud_file(dir.write("a.txt", "1 2 3")); }, ".txt");
	expect_file_error([&] { (void)read_point_cloud_file(dir.write("noext", "1 2 3")); }, "noext");
	expect_file_error([&] { (void)read_point_cloud_file(dir.path / "missing.xyz"); }, "could not open");
	EXPECT_EQ(read_point_cloud_file(dir.write("p.PCD", "FIELDS x y z\nSIZE 4 4 4\nTYPE F F F\nPOINTS 1\nDATA ascii\n1 2 3\n")).size(), 1u);
	EXPECT_EQ(read_point_cloud_file(dir.write("v.BinVox", binvox("#binvox 1\ndim 1 1 1\n", { { 1, 1 } }))).size(), 1u);

	// shift: the minimum of every axis goes to 0, points that are not finite stay and do not count
	std::vector<Point3f> cloud = { { 3, -2, 10 }, { 5, 4, 12 }, { std::numeric_limits<float>::quiet_NaN(), -100, 0 }, { 4, 0, 11 }, { std::numeric_limits<float>::infinity(), 1, 1 } };
	shift_point_cloud_to_zero(cloud);
	EXPECT_TRUE(eq(cloud[0], { 0, 0, 0 }));
	EXPECT_TRUE(eq(cloud[1], { 2, 6, 2 }));
	EXPECT_TRUE(std::isnan(cloud[2].x));
	EXPECT_TRUE(eq(cloud[3], { 1, 2, 1 }));
	EXPECT_TRUE(std::isinf(cloud[4].x));
	std::vector<Point3f> none;
	shift_point_cloud_to_zero(none);
	EXPECT_TRUE(none.empty());
	std::vector<Point3f> only_nan = { { std::numeric_limits<float>::quiet_NaN(), 1, 1 } };
	shift_point_cloud_to_zero(only_nan);
	EXPECT_EQ(only_nan[0].y, 1.0f);

	// load: model_path / file, then shift, then scaling, then offset
	dir.write("m.xyz", "3 -2 10\n5 4 12\n");
	PointCloudLoadOptions options;
	options.shift_to_zero = true;
	options.scaling = 2.0f;
	options.offset = { 100, 200, 300 };
	auto loaded = load_point_cloud("m.xyz", dir.path, options);
	ASSERT_EQ(loaded.size(), 2u);
	EXPECT_TRUE(eq(loaded[0], { 100, 200, 300 }));
	EXPECT_TRUE(eq(loaded[1], { 104, 212, 304 }));
	options.shift_to_zero = false;
	loaded = load_point_cloud("m.xyz", dir.path, options);
	EXPECT_TRUE(eq(loaded[0], { 106, 196, 320 })); // 2 * p + offset
	EXPECT_TRUE(eq(loaded[1], { 110, 208, 324 }));
	// an absolute file name replaces the model path
	loaded = load_point_cloud(dir.path / "m.xyz", "/does/not/exist", {});
	EXPECT_TRUE(eq(loaded[1], { 5, 4, 12 }));
	// a file below a model path that is a directory of its own
	fs::create_directories(dir.path / "sub");
	dir.write("sub/n.xyz", "7 8 9\n");
	EXPECT_EQ(load_point_cloud("sub/n.xyz", dir.path).size(), 1u);
	expect_file_error([&] { (void)load_point_cloud("m.xyz", dir.path / "nowhere"); }, "could not open");
}

// ---------------------------------------------------------------- the box

namespace
{
	const Dim3ui kBig{ 1u << 20, 1u << 20, 1u << 20 };

	std::vector<Coord3ui> expected_box(int64_t lo[3], int64_t hi[3], int64_t step, const uint32_t limit[3])
	{
		std::vector<Coord3ui> out;
		for (int64_t x = lo[0]; x <= hi[0]; x += step)
			for (int64_t y = lo[1]; y <= hi[1]; y += step)
				for (int64_t z = lo[2]; z <= hi[2]; z += step)
					if (x < limit[0] && y < limit[1] && z < limit[2])
						out.push_back({ uint32_t(x), uint32_t(y), uint32_t(z) });
		return out;
	}

	bool same(const std::vector<Coord3ui>& a, const std::vector<Coord3ui>& b)
	{
		return a.size() == b.size() && std::equal(a.begin(), a.end(), b.begin(), [](const Coord3ui& p, const Coord3ui& q) { return p.x == q.x && p.y == q.y && p.z == q.z; });
	}
}

TEST(PointCloudFiles, BoxCoordinates)
{
	const uint32_t big[3] = { 1u << 20, 1u << 20, 1u << 20 };
	// side 0.5 is exact in float: voxels 2 .. 4, 4 .. 6, 0 .. 2, the upper corner voxel is included (the maximum is a multiple of the side)
	{
		int64_t lo[3] = { 2, 4, 0 }, hi[3] = { 4, 6, 2 };
		const BoxVoxels box = box_voxel_coordinates({ 1, 2, 0 }, { 2, 3, 1 }, 0.5f, 0.5f, kBig);
		EXPECT_EQ(box.total, 27u);
		EXPECT_TRUE(same(box.coordinates, expected_box(lo, hi, 1, big)));
		EXPECT_EQ(box.coordinates.front().x, 2u);
		EXPECT_EQ(box.coordinates[1].z, 1u); // z is the innermost axis
		EXPECT_EQ(box.coordinates[3].y, 5u);
		EXPECT_EQ(box.coordinates[9].x, 3u);
	}
	// the minimum is cut at 0 on every axis, the maximum rounds up: floor(-3 -> 0) .. ceil(1.3 / 0.5 = 2.6 -> 3)
	{
		int64_t lo[3] = { 0, 0, 0 }, hi[3] = { 3, 3, 3 };
		const BoxVoxels box = box_voxel_coordinates({ -3, -0.2f, -1e6f }, { 1.3f, 1.3f, 1.3f }, 0.5f, 0.5f, kBig);
		EXPECT_EQ(box.total, 64u);
		EXPECT_TRUE(same(box.coordinates, expected_box(lo, hi, 1, big)));
	}
	// the minimum rounds down
	{
		int64_t lo[3] = { 2, 2, 2 }, hi[3] = { 3, 3, 3 };
		EXPECT_TRUE(same(box_voxel_coordinates({ 1.2f, 1.2f, 1.2f }, { 1.4f, 1.4f, 1.4f }, 0.5f, 0.5f, kBig).coordinates, expected_box(lo, hi, 1, big)));
	}
	// step = round(spacing / side): 2 voxels, then 3; below 1 it is 1 (CUDA: 0 and an endless loop)
	{
		int64_t lo[3] = { 1, 1, 1 }, hi[3] = { 8, 8, 8 };
		const BoxVoxels two = box_voxel_coordinates({ 1, 1, 1 }, { 4, 4, 4 }, 2.0f, 1.0f, kBig); // 1 .. 4 voxels
		EXPECT_EQ(two.coordinates.size(), 8u);              // 1 and 3 per axis
		EXPECT_EQ(two.total, 8u);
		hi[0] = hi[1] = hi[2] = 4;
		EXPECT_TRUE(same(two.coordinates, expected_box(lo, hi, 2, big)));
		hi[0] = hi[1] = hi[2] = 8;
		EXPECT_TRUE(same(box_voxel_coordinates({ 1, 1, 1 }, { 8, 8, 8 }, 2.6f, 1.0f, kBig).coordinates, expected_box(lo, hi, 3, big))); // round(2.6) = 3
		EXPECT_TRUE(same(box_voxel_coordinates({ 1, 1, 1 }, { 8, 8, 8 }, 2.4f, 1.0f, kBig).coordinates, expected_box(lo, hi, 2, big)));
		EXPECT_TRUE(same(box_voxel_coordinates({ 1, 1, 1 }, { 8, 8, 8 }, 0.3f, 1.0f, kBig).coordinates, expected_box(lo, hi, 1, big)));
		EXPECT_TRUE(same(box_voxel_coordinates({ 1, 1, 1 }, { 8, 8, 8 }, 0.5f, 1.0f, kBig).coordinates, expected_box(lo, hi, 1, big))); // round(0.5) = 1
	}
	// the limit: voxels at or beyond it are neither generated nor lost from the total
	{
		const uint32_t lim[3] = { 4, 100, 3 };
		int64_t lo[3] = { 2, 4, 0 }, hi[3] = { 4, 6, 2 };
		const BoxVoxels box = box_voxel_coordinates({ 1, 2, 0 }, { 2, 3, 1 }, 0.5f, 0.5f, { 4, 100, 3 });
		EXPECT_EQ(box.total, 27u);
		EXPECT_TRUE(same(box.coordinates, expected_box(lo, hi, 1, lim)));
		EXPECT_EQ(box.coordinates.size(), 18u); // x = 2, 3 (4 is at the limit), y = 4 .. 6, z = 0 .. 2
	}
	{
		const BoxVoxels none = box_voxel_coordinates({ 5, 5, 5 }, { 6, 6, 6 }, 1.0f, 1.0f, { 5, 100, 100 }); // starts at x = 5 = limit
		EXPECT_TRUE(none.coordinates.empty());
		EXPECT_EQ(none.total, 8u);
		// stepping: the last voxel inside the limit counts only when the step hits it
		const BoxVoxels stepped = box_voxel_coordinates({ 0, 0, 0 }, { 9, 0, 0 }, 3.0f, 1.0f, { 7, 7, 7 }); // x = 0, 3, 6, 9: 9 is outside
		ASSERT_EQ(stepped.coordinates.size(), 3u);
		EXPECT_EQ(stepped.coordinates[2].x, 6u);
		EXPECT_EQ(stepped.total, 4u);
	}
	// a start at the limit with a step above 1 gives nothing; a limit that the step hits exactly does not belong to the box
	EXPECT_TRUE(box_voxel_coordinates({ 5, 0, 0 }, { 9, 0, 0 }, 3.0f, 1.0f, { 5, 7, 7 }).coordinates.empty());
	{
		const BoxVoxels hit = box_voxel_coordinates({ 0, 0, 0 }, { 9, 0, 0 }, 3.0f, 1.0f, { 6, 7, 7 }); // x = 0, 3, 6, 9 below 6
		ASSERT_EQ(hit.coordinates.size(), 2u);
		EXPECT_EQ(hit.coordinates[1].x, 3u);
	}
	// an empty box: the voxel minimum is one above the maximum
	EXPECT_THROW((void)box_voxel_coordinates({ 3, 0, 0 }, { 2, 0, 0 }, 1.0f, 1.0f, kBig), VulkanError);
	EXPECT_EQ(box_voxel_coordinates({ 3, 0, 0 }, { 3, 0, 0 }, 1.0f, 1.0f, kBig).coordinates.size(), 1u);
	auto box_error = [](Point3f lo, Point3f hi, float spacing, float side, const std::string& contains, Dim3ui limit = kBig) {
		try
		{
			(void)box_voxel_coordinates(lo, hi, spacing, side, limit);
			ADD_FAILURE() << "no exception, expected '" << contains << "'";
		}
		catch (const VulkanError& e)
		{
			EXPECT_NE(std::string(e.what()).find(contains), std::string::npos) << e.what();
		}
	};
	const float nan = std::numeric_limits<float>::quiet_NaN(), inf = std::numeric_limits<float>::infinity();
	box_error({ nan, 0, 0 }, { 1, 1, 1 }, 1, 1, "finite");
	box_error({ 0, 0, 0 }, { 1, inf, 1 }, 1, 1, "finite");
	box_error({ 0, 0, 0 }, { 1, 1, nan }, 1, 1, "finite");
	box_error({ 0, 0, 0 }, { 1, 1, 1 }, 1, 0, "side length");
	box_error({ 0, 0, 0 }, { 1, 1, 1 }, 1, -1, "side length");
	box_error({ 0, 0, 0 }, { 1, 1, 1 }, 1, nan, "side length");
	box_error({ 0, 0, 0 }, { 1, 1, 1 }, 1, inf, "side length");
	box_error({ 0, 0, 0 }, { 1, 1, 1 }, 0, 1, "spacing");
	box_error({ 0, 0, 0 }, { 1, 1, 1 }, -1, 1, "spacing");
	box_error({ 0, 0, 0 }, { 1, 1, 1 }, nan, 1, "spacing");
	box_error({ 0, 0, 0 }, { 1, 1, 1 }, inf, 1, "spacing");
	box_error({ 0, 0, 0 }, { 1e12f, 1, 1 }, 1, 1, "2^31");
	box_error({ 0, 0, 0 }, { -1e12f, 1, 1 }, 1, 1, "2^31");
	box_error({ 0, 0, 0 }, { 1, 1, 1 }, 1, 1e-12f, "2^31");
	box_error({ 1, 1, 1 }, { 0, 5, 5 }, 1, 1, "empty");
	// more voxels than one call may generate (checked before any is made); the clipping to the limit comes first
	box_error({ 0, 0, 0 }, { float(kMaxBoxVoxels), 0, 0 }, 1, 1, "at most", { 4000000000u, 4000000000u, 4000000000u });
	EXPECT_EQ(box_voxel_coordinates({ 0, 0, 0 }, { float(kMaxBoxVoxels), 0, 0 }, 1, 1, { 100, 1, 1 }).coordinates.size(), 100u);
	EXPECT_EQ(box_voxel_coordinates({ 0, 0, 0 }, { float(kMaxBoxVoxels), 0, 0 }, 1, 1, { 100, 1, 1 }).total, kMaxBoxVoxels + 1);
	// equal corners inside a voxel still give two voxels per axis: floor(2.5) = 2 .. ceil(2.5) = 3
	EXPECT_EQ(box_voxel_coordinates({ 2.5f, 2.5f, 2.5f }, { 2.5f, 2.5f, 2.5f }, 1.0f, 1.0f, kBig).coordinates.size(), 8u);
	// equal corners on a voxel border give one
	EXPECT_EQ(box_voxel_coordinates({ 2, 2, 2 }, { 2, 2, 2 }, 1.0f, 1.0f, kBig).coordinates.size(), 1u);
}

// ---------------------------------------------------------------- the façade calls

namespace
{
	constexpr Dim3ui kMapDims{ 40, 30, 20 };
	constexpr float kMapSide = 0.1f;

	class FileFacadeTest : public VulkanTest
	{
	protected:
		std::unique_ptr<GpuVoxels> make()
		{
			auto gvl = std::make_unique<GpuVoxels>(ctx());
			gvl->initialize(kMapDims, kMapSide);
			return gvl;
		}
	};

	std::vector<uint32_t> bit_words(const GpuVoxelsMapSharedPtr& m) { return m->as<BitVoxelMap>()->download_words(); }

	uint64_t count_voxels(const std::vector<uint32_t>& words)
	{
		uint64_t n = 0;
		for (size_t i = 0; i < words.size(); i += 8)
			n += std::any_of(words.begin() + i, words.begin() + i + 8, [](uint32_t w) { return w != 0; }) ? 1 : 0;
		return n;
	}
}

TEST_F(FileFacadeTest, InsertFromFile)
{
	const fs::path rel = fs::path("hollie") / "arm_0_link.xyz";
	if (!fs::exists(kModels / rel))
		GTEST_SKIP() << "model file missing";
	const auto gvl = make();
	const GpuVoxelsMapSharedPtr bit = gvl->add_map(MT_BITVECTOR_VOXELMAP, "bit");
	const GpuVoxelsMapSharedPtr tree = gvl->add_map(MT_BITVECTOR_OCTREE, "tree");
	const GpuVoxelsMapSharedPtr list = gvl->add_map(MT_BITVECTOR_VOXELLIST, "list");

	// the cloud is 0.13 x 0.12 x 0.07 m with z from -0.01: without the shift and an offset in z a part lies below the map, scaled by 30 a part beyond it
	for (const auto& [shift, offset, scaling, some_outside] : { std::tuple{ true, Point3f{ 1.0f, 1.2f, 0.5f }, 1.0f, false },
		std::tuple{ false, Point3f{ 2.0f, 1.5f, 0.0f }, 1.0f, true }, std::tuple{ true, Point3f{ 0.25f, 0.25f, 0.25f }, 30.0f, true } })
	{
		for (const char* name : { "bit", "tree", "list" })
			gvl->clear_map(name);
		PointCloudLoadOptions options;
		options.shift_to_zero = shift;
		options.offset = offset;
		options.scaling = scaling;
		const std::vector<Point3f> points = load_point_cloud(rel, kModels, options);

		BitVoxelMap bit_twin(ctx(), kMapDims, kMapSide);
		Octree tree_twin(ctx(), OctreeKind::Deterministic, kMapSide);
		BitVoxelList list_twin(ctx(), kMapDims, kMapSide);
		const uint32_t outside_bit = bit_twin.insert_point_cloud(points, 7);
		const uint32_t outside_tree = tree_twin.insert_point_cloud(points, eBVM_OCCUPIED);
		const uint32_t outside_list = list_twin.insert_point_cloud(points, 7);

		uint32_t got = 12345;
		EXPECT_TRUE(gvl->insert_point_cloud_from_file("bit", rel, kModels, 7, shift, offset, scaling, &got));
		EXPECT_EQ(got, outside_bit);
		EXPECT_EQ(bit_words(bit), bit_twin.download_words());
		EXPECT_GT(count_voxels(bit_words(bit)), 2u);
		EXPECT_EQ(outside_bit > 0, some_outside);

		EXPECT_TRUE(gvl->insert_point_cloud_from_file("tree", rel, kModels, eBVM_OCCUPIED, shift, offset, scaling, &got));
		EXPECT_EQ(got, outside_tree);
		EXPECT_EQ(tree->as<Octree>()->dump(), tree_twin.dump());

		EXPECT_TRUE(gvl->insert_point_cloud_from_file("list", rel, kModels, 7, shift, offset, scaling, &got));
		EXPECT_EQ(got, outside_list);
		EXPECT_EQ(list->as<BitVoxelList>()->download_ids(), list_twin.download_ids());
		EXPECT_EQ(list->as<BitVoxelList>()->download_voxel_words(), list_twin.download_voxel_words());
	}

	// an absolute path ignores the model path; the defaults leave the points as they are; no counter wanted
	gvl->clear_map("bit");
	EXPECT_TRUE(gvl->insert_point_cloud_from_file("bit", kModels / rel, "/nowhere", 7));
	BitVoxelMap plain(ctx(), kMapDims, kMapSide);
	plain.insert_point_cloud(read_xyz_file(kModels / rel), 7);
	EXPECT_EQ(bit_words(bit), plain.download_words());
}

TEST_F(FileFacadeTest, InsertFromFileFailures)
{
	const TempDir dir;
	const auto gvl = make();
	const GpuVoxelsMapSharedPtr bit = gvl->add_map(MT_BITVECTOR_VOXELMAP, "bit");
	dir.write("ok.xyz", "0.15 0.15 0.15\n");
	dir.write("bad.pcd", "FIELDS x y z\nSIZE 4 4 4\nTYPE F F F\nPOINTS 5\nDATA ascii\n1 2 3\n");
	dir.write("bad.binvox", "#binvox 1\ndim 2 2 2\ndata\n");
	dir.write("what.txt", "1 2 3");
	const std::vector<uint32_t> fresh = bit_words(bit);
	uint32_t got = 777;
	EXPECT_FALSE(gvl->insert_point_cloud_from_file("nope", "ok.xyz", dir.path, 7, false, { 0, 0, 0 }, 1.0f, &got));
	EXPECT_FALSE(gvl->insert_point_cloud_from_file("bit", "missing.xyz", dir.path, 7, false, { 0, 0, 0 }, 1.0f, &got));
	EXPECT_FALSE(gvl->insert_point_cloud_from_file("bit", "bad.pcd", dir.path, 7));
	EXPECT_FALSE(gvl->insert_point_cloud_from_file("bit", "bad.binvox", dir.path, 7));
	EXPECT_FALSE(gvl->insert_point_cloud_from_file("bit", "what.txt", dir.path, 7));
	EXPECT_EQ(got, 777u); // a failure does not touch the counter
	EXPECT_EQ(bit_words(bit), fresh);
	EXPECT_TRUE(gvl->insert_point_cloud_from_file("bit", "ok.xyz", dir.path, 7, false, { 0, 0, 0 }, 1.0f, &got));
	EXPECT_EQ(got, 0u);
	EXPECT_EQ(count_voxels(bit_words(bit)), 1u);
	// the meaning rules of the map are not file errors: they still throw
	const GpuVoxelsMapSharedPtr tree = gvl->add_map(MT_BITVECTOR_OCTREE, "tree");
	EXPECT_THROW((void)gvl->insert_point_cloud_from_file("tree", "ok.xyz", dir.path, 5), VulkanError);
}

TEST_F(FileFacadeTest, InsertBox)
{
	const auto gvl = make();
	const GpuVoxelsMapSharedPtr bit = gvl->add_map(MT_BITVECTOR_VOXELMAP, "bit");
	const GpuVoxelsMapSharedPtr tree = gvl->add_map(MT_BITVECTOR_OCTREE, "tree");
	const GpuVoxelsMapSharedPtr list = gvl->add_map(MT_BITVECTOR_VOXELLIST, "list");

	// voxels 5 .. 10 on every axis: floor(5.5) .. ceil(9.5), the voxel at the upper corner is in
	std::vector<Coord3ui> coords;
	std::vector<uint32_t> expected(size_t(kMapDims.voxel_count()) * 8, 0);
	for (uint32_t x = 5; x <= 10; ++x)
		for (uint32_t y = 5; y <= 10; ++y)
			for (uint32_t z = 5; z <= 10; ++z)
			{
				coords.push_back({ x, y, z });
				expected[((uint64_t(z) * kMapDims.y + y) * kMapDims.x + x) * 8] |= 1u << 7;
			}
	for (const uint16_t ppv : { uint16_t(1), uint16_t(2), uint16_t(7) }) // only the step depends on it: always 1
	{
		gvl->clear_map("bit");
		uint32_t outside = 99;
		EXPECT_TRUE(gvl->insert_box_into_map({ 0.55f, 0.55f, 0.55f }, { 0.95f, 0.95f, 0.95f }, "bit", 7, ppv, &outside));
		EXPECT_EQ(outside, 0u);
		EXPECT_EQ(bit_words(bit), expected) << ppv;
	}
	EXPECT_TRUE(gvl->insert_box_into_map({ 0.55f, 0.55f, 0.55f }, { 0.95f, 0.95f, 0.95f }, "bit", 7)); // defaults, no counter
	EXPECT_EQ(bit_words(bit), expected);

	// the octree and a list get the same coordinates
	EXPECT_TRUE(gvl->insert_box_into_map({ 0.55f, 0.55f, 0.55f }, { 0.95f, 0.95f, 0.95f }, "tree", eBVM_OCCUPIED));
	Octree tree_twin(ctx(), OctreeKind::Deterministic, kMapSide);
	tree_twin.insert_coordinate_list(coords, eBVM_OCCUPIED);
	EXPECT_EQ(tree->as<Octree>()->dump(), tree_twin.dump());
	EXPECT_TRUE(gvl->insert_box_into_map({ 0.55f, 0.55f, 0.55f }, { 0.95f, 0.95f, 0.95f }, "list", 7));
	BitVoxelList list_twin(ctx(), kMapDims, kMapSide);
	list_twin.insert_coordinate_list(coords, 7);
	EXPECT_EQ(list->as<BitVoxelList>()->download_ids(), list_twin.download_ids());
	EXPECT_EQ(list->as<BitVoxelList>()->download_voxel_words(), list_twin.download_voxel_words());

	// partly outside the map: x = 35 .. 46 (12 voxels, 35 .. 39 inside), y and z = 10 .. 16 (7 voxels): 7 * 49 voxels are outside
	gvl->clear_map("bit");
	uint32_t outside = 0;
	EXPECT_TRUE(gvl->insert_box_into_map({ 3.55f, 1.05f, 1.05f }, { 4.55f, 1.55f, 1.55f }, "bit", 7, 1, &outside));
	EXPECT_EQ(outside, 7u * 49u);
	std::vector<uint32_t> edge(size_t(kMapDims.voxel_count()) * 8, 0);
	for (uint32_t x = 35; x < 40; ++x)
		for (uint32_t y = 10; y <= 16; ++y)
			for (uint32_t z = 10; z <= 16; ++z)
				edge[((uint64_t(z) * kMapDims.y + y) * kMapDims.x + x) * 8] |= 1u << 7;
	EXPECT_EQ(bit_words(bit), edge);
	// the octree is bigger than the map: nothing is outside there
	EXPECT_TRUE(gvl->insert_box_into_map({ 3.55f, 1.05f, 1.05f }, { 4.55f, 1.55f, 1.55f }, "tree", eBVM_OCCUPIED, 1, &outside));
	EXPECT_EQ(outside, 0u);
	// a box wholly outside inserts nothing and counts everything: voxels 50 .. 53 on x, 0 on y and z
	gvl->clear_map("bit");
	EXPECT_TRUE(gvl->insert_box_into_map({ 5.05f, 0.0f, 0.0f }, { 5.25f, 0.0f, 0.0f }, "bit", 7, 1, &outside));
	EXPECT_EQ(outside, 4u);
	EXPECT_EQ(count_voxels(bit_words(bit)), 0u);

	EXPECT_FALSE(gvl->insert_box_into_map({ 0, 0, 0 }, { 1, 1, 1 }, "nope", 7));
	try
	{
		(void)gvl->insert_box_into_map({ 0, 0, 0 }, { 1, 1, 1 }, "bit", 7, 0);
		ADD_FAILURE() << "no exception";
	}
	catch (const VulkanError& e)
	{
		EXPECT_NE(std::string(e.what()).find("points_per_voxel"), std::string::npos) << e.what();
	}
	EXPECT_THROW((void)gvl->insert_box_into_map({ 1, 1, 1 }, { 0.5f, 2, 2 }, "bit", 7), VulkanError); // minimum above maximum
	EXPECT_THROW((void)gvl->insert_box_into_map({ 0, 0, 0 }, { 1, 1, 1 }, "tree", 5), VulkanError);   // the meaning rules of the map
}
