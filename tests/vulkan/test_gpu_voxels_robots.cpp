// Slice S7-3: DH robots in the façade (add_robot from files and from a MetaPointCloud, the wrappers, insert_robot_into_map and the
// self collision aware insertion). The chain itself is tested in test_kinematic_chain.cpp; here the façade is checked against the
// independent double precision forward kinematics (tests/reference), against the classes used directly (a twin chain and a twin map built
// in this file) and against the host reference of the bit map.
#include "vulkan_fixture.h"

#include <bitset>
#include <cmath>
#include <filesystem>
#include <functional>
#include <memory>
#include <numbers>
#include <random>
#include <string>
#include <vector>

#include <gpu_voxels/vulkan/GpuVoxels.h>

#include "../reference/bit_voxel_map_reference.h"
#include "../reference/kinematics_reference.h"
#include "facade_support.h"

using namespace gpu_voxels::vulkan;
using gvl_test::snapshot;
using gvl_test::VulkanTest;
namespace ref = gvl_reference;
namespace fs = std::filesystem;

namespace
{
	constexpr float kHalfPi = static_cast<float>(std::numbers::pi / 2);
	constexpr Dim3ui kDims{ 40, 40, 40 };
	constexpr float kSide = 0.1f;
	const fs::path kModels = GVL_TEST_MODEL_DIR;

	/// A robot as the façade takes it: link i has `names[i]` and `dh[i]`; sub cloud k belongs to the link `cloud_links[k]`.
	struct Robot
	{
		std::vector<std::string> names;
		std::vector<DHParameters> dh;
		std::vector<std::string> cloud_links;
		std::vector<std::vector<Point3f>> clouds;
	};

	std::vector<Point3f> random_cloud(std::mt19937& rng, int n, float range)
	{
		std::uniform_real_distribution<float> coord(-range, range);
		std::vector<Point3f> cloud(n);
		for (auto& p : cloud)
			p = { coord(rng), coord(rng), coord(rng) };
		return cloud;
	}

	/// Five links with random DH parameters (link 3 prismatic); link 2 has no cloud.
	Robot random_robot(uint64_t seed)
	{
		std::mt19937 rng(static_cast<uint32_t>(seed));
		std::uniform_real_distribution<float> angle(-3.1f, 3.1f), length(-0.6f, 0.6f);
		Robot r;
		for (int i = 0; i < 5; ++i)
		{
			r.names.push_back("link" + std::to_string(i));
			r.dh.push_back({ length(rng), angle(rng), length(rng), angle(rng), angle(rng), i == 3 ? DHJointType::Prismatic : DHJointType::Revolute });
			if (i != 2)
			{
				r.cloud_links.push_back(r.names.back());
				r.clouds.push_back(random_cloud(rng, 40 + 10 * i, 0.5f));
			}
		}
		return r;
	}

	/// Four links, quarter turns and lengths in voxels, clouds in the middle of voxels: the points stay at least 0.02 m from a voxel
	/// boundary in any configuration, so the float / double difference of the transformation cannot change a voxel (as in the S2 test).
	/// Link 2 has no cloud.
	Robot grid_robot(uint64_t seed)
	{
		std::mt19937 rng(static_cast<uint32_t>(seed));
		std::uniform_int_distribution<int> cell(-3, 3);
		std::uniform_real_distribution<float> jitter(-0.03f, 0.03f);
		auto cloud = [&] {
			std::vector<Point3f> points;
			for (int i = 0; i < 12; ++i)
				points.push_back({ 0.1f * static_cast<float>(cell(rng)) + 0.05f + jitter(rng), 0.1f * static_cast<float>(cell(rng)) + 0.05f + jitter(rng),
					0.1f * static_cast<float>(cell(rng)) + 0.05f + jitter(rng) });
			return points;
		};
		Robot r;
		r.names = { "l0", "l1", "l2", "l3" };
		r.dh = { { 0.2f, 0.0f, 0.1f, kHalfPi, 0.0f, DHJointType::Revolute }, { 0.0f, 0.0f, 0.3f, 0.0f, 0.0f, DHJointType::Revolute },
			{ 0.0f, 0.0f, 0.2f, -kHalfPi, 0.0f, DHJointType::Revolute }, { 0.1f, 0.0f, 0.0f, 0.0f, 0.0f, DHJointType::Revolute } };
		for (const char* link : { "l0", "l1", "l3" })
		{
			r.cloud_links.push_back(link);
			r.clouds.push_back(cloud());
		}
		return r;
	}

	std::vector<LinkSpec> specs(const Robot& r)
	{
		std::vector<LinkSpec> links;
		for (std::size_t i = 0; i < r.names.size(); ++i)
			links.push_back({ r.names[i], r.dh[i], std::nullopt });
		for (std::size_t k = 0; k < r.clouds.size(); ++k)
			for (auto& link : links)
				if (link.name == r.cloud_links[k])
					link.cloud = r.clouds[k];
		return links;
	}

	std::size_t link_of_cloud(const Robot& r, std::size_t k)
	{
		for (std::size_t i = 0; i < r.names.size(); ++i)
			if (r.names[i] == r.cloud_links[k])
				return i;
		ADD_FAILURE() << "no link for cloud " << k;
		return 0;
	}

	ref::DH to_reference(DHConvention c, const DHParameters& p)
	{
		return { p.d, p.theta, p.a, p.alpha, p.value, p.joint_type == DHJointType::Prismatic, c == DHConvention::Craigs };
	}

	ref::Mat4 to_ref(const Matrix4f& f)
	{
		ref::Mat4 m;
		for (int r = 0; r < 4; ++r)
			for (int c = 0; c < 4; ++c)
				m.at(r, c) = f(r, c);
		return m;
	}

	Matrix4f translation(float x, float y, float z)
	{
		Matrix4f m = Matrix4f::identity();
		m(0, 3) = x;
		m(1, 3) = y;
		m(2, 3) = z;
		return m;
	}

	std::vector<ref::Mat4> forward_kinematics(DHConvention c, const Robot& r, const Matrix4f& base)
	{
		std::vector<ref::DH> links;
		for (const auto& p : r.dh)
			links.push_back(to_reference(c, p));
		return ref::forward_kinematics(to_ref(base), links);
	}

	/// The clouds in world coordinates by the double precision reference, rounded to float.
	std::vector<std::vector<Point3f>> world_clouds(DHConvention c, const Robot& r, const Matrix4f& base)
	{
		const auto transforms = forward_kinematics(c, r, base);
		std::vector<std::vector<Point3f>> world;
		for (std::size_t k = 0; k < r.clouds.size(); ++k)
		{
			std::vector<Point3f> points;
			for (const auto& p : r.clouds[k])
			{
				const auto q = transforms[link_of_cloud(r, k)].apply(p);
				points.push_back({ static_cast<float>(q[0]), static_cast<float>(q[1]), static_cast<float>(q[2]) });
			}
			world.push_back(std::move(points));
		}
		return world;
	}

	::testing::AssertionResult matrices_near(const Matrix4f& got, const ref::Mat4& expected, double tolerance)
	{
		for (int r = 0; r < 4; ++r)
			for (int c = 0; c < 4; ++c)
				if (std::abs(got(r, c) - expected.at(r, c)) > tolerance)
					return ::testing::AssertionFailure() << "element (" << r << "," << c << "): got " << got(r, c) << " expected " << expected.at(r, c);
		return ::testing::AssertionSuccess();
	}

	::testing::AssertionResult clouds_near(const MetaPointCloud& got, const std::vector<std::vector<Point3f>>& expected, double tolerance)
	{
		if (got.num_clouds() != expected.size())
			return ::testing::AssertionFailure() << got.num_clouds() << " sub clouds, expected " << expected.size();
		for (uint32_t k = 0; k < got.num_clouds(); ++k)
		{
			const auto points = got.download_cloud(k);
			if (points.size() != expected[k].size())
				return ::testing::AssertionFailure() << "sub cloud " << k << " has " << points.size() << " points, expected " << expected[k].size();
			for (std::size_t i = 0; i < points.size(); ++i)
				if (std::abs(points[i].x - expected[k][i].x) > tolerance || std::abs(points[i].y - expected[k][i].y) > tolerance ||
					std::abs(points[i].z - expected[k][i].z) > tolerance)
					return ::testing::AssertionFailure() << "sub cloud " << k << " point " << i << " differs";
		}
		return ::testing::AssertionSuccess();
	}

	::testing::AssertionResult clouds_equal(const MetaPointCloud& got, const std::vector<std::vector<Point3f>>& expected)
	{
		return clouds_near(got, expected, 0.0);
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

	std::vector<uint32_t> words_of(const std::bitset<256>& bits)
	{
		std::vector<uint32_t> words(8, 0);
		for (uint32_t b = 0; b < 256; ++b)
			if (bits.test(b))
				words[b / 32] |= 1u << (b % 32);
		return words;
	}

	class RobotFacadeTest : public VulkanTest
	{
	protected:
		std::unique_ptr<GpuVoxels> make()
		{
			auto gvl = std::make_unique<GpuVoxels>(ctx());
			gvl->initialize(kDims, kSide);
			return gvl;
		}
		MetaPointCloud meta(const Robot& r) { return MetaPointCloud(ctx(), std::span<const std::vector<Point3f>>(r.clouds)); }
		bool add(GpuVoxels& gvl, const std::string& name, DHConvention convention, const Robot& r)
		{
			const MetaPointCloud clouds = meta(r);
			return gvl.add_robot(name, convention, r.names, r.dh, clouds, r.cloud_links);
		}
		/// The same robot as a chain built directly (twin of the façade's robot).
		KinematicChain twin(DHConvention convention, const Robot& r, const Matrix4f& base) { return KinematicChain(ctx(), convention, specs(r), base); }
	};

	/// Sets the joint values of the robot (also in the description of the reference).
	std::map<std::string, float> set_joints(Robot& r, const std::function<float(std::size_t)>& value)
	{
		std::map<std::string, float> joints;
		for (std::size_t i = 0; i < r.names.size(); ++i)
		{
			r.dh[i].value = value(i);
			joints[r.names[i]] = r.dh[i].value;
		}
		return joints;
	}
}

TEST_F(RobotFacadeTest, FromMetaPointCloudMatchesReference)
{
	std::mt19937 rng(71);
	std::uniform_real_distribution<float> angle(-2.5f, 2.5f);
	for (DHConvention convention : { DHConvention::Classic, DHConvention::Craigs })
	{
		Robot robot = random_robot(convention == DHConvention::Classic ? 1 : 2);
		const auto gvl = make();
		ASSERT_TRUE(add(*gvl, "robot", convention, robot));

		const KinematicChainSharedPtr chain = gvl->get_robot("robot");
		ASSERT_TRUE(chain);
		EXPECT_EQ(chain->num_links(), 5u);
		EXPECT_EQ(chain->link_names(), robot.names);
		EXPECT_EQ(chain->cloud_index("link0"), std::optional<uint32_t>(0));
		EXPECT_EQ(chain->cloud_index("link1"), std::optional<uint32_t>(1));
		EXPECT_EQ(chain->cloud_index("link2"), std::nullopt);
		EXPECT_EQ(chain->cloud_index("link3"), std::optional<uint32_t>(2));
		EXPECT_EQ(chain->cloud_index("link4"), std::optional<uint32_t>(3));
		EXPECT_TRUE(clouds_equal(chain->link_clouds(), robot.clouds)) << "the clouds are copied as they are";
		EXPECT_EQ(gvl->get_robot("robot"), chain) << "the same robot every time";

		const Matrix4f base = [] { Matrix4f m = translation(0.3f, -0.2f, 0.5f); m(0, 0) = std::cos(0.7f); m(0, 1) = -std::sin(0.7f); m(1, 0) = std::sin(0.7f); m(1, 1) = std::cos(0.7f); return m; }();
		EXPECT_TRUE(gvl->set_robot_base_transformation("robot", base));

		for (int round = 0; round < 3; ++round)
		{
			std::map<std::string, float> joints;
			for (std::size_t i = 0; i < robot.names.size(); ++i)
				if (round == 0 || i % (round + 1) == 0)
				{
					robot.dh[i].value = angle(rng);
					joints[robot.names[i]] = robot.dh[i].value;
				}
			EXPECT_TRUE(gvl->set_robot_configuration("robot", joints));

			std::map<std::string, float> read{ { "not a joint", 7.0f }, { "link0", 99.0f } };
			EXPECT_TRUE(gvl->get_robot_configuration("robot", read));
			EXPECT_EQ(read.size(), robot.names.size() + 1) << "the map is extended";
			EXPECT_EQ(read.at("not a joint"), 7.0f);
			for (std::size_t i = 0; i < robot.names.size(); ++i)
				EXPECT_EQ(read.at(robot.names[i]), robot.dh[i].value) << "joint " << i << " round " << round;

			const auto expected = forward_kinematics(convention, robot, base);
			for (std::size_t i = 0; i <= robot.names.size(); ++i)
			{
				Matrix4f got;
				EXPECT_TRUE(gvl->get_robot_transformation("robot", i, got));
				EXPECT_TRUE(matrices_near(got, expected[i], 3e-5)) << "transform " << i << " round " << round;
			}
			EXPECT_TRUE(clouds_near(chain->transformed_clouds(), world_clouds(convention, robot, base), 2e-4)) << "round " << round;
		}

		Matrix4f untouched = translation(9, 9, 9);
		expect_error([&] { (void)gvl->get_robot_transformation("robot", 6, untouched); }, "index out of range");
		EXPECT_EQ(untouched, translation(9, 9, 9));
		Matrix4f got_base;
		EXPECT_TRUE(gvl->get_robot_transformation("robot", 0, got_base));
		EXPECT_EQ(got_base, base) << "transform 0 is the base itself";
	}
}

TEST_F(RobotFacadeTest, FromFilesMatchesReference)
{
	const std::vector<std::string> files{ "hollie/arm_0_link.xyz", "hollie/arm_1_link.xyz", "hollie/arm_2_link.xyz" };
	for (const auto& f : files)
		if (!fs::exists(kModels / f))
			GTEST_SKIP() << "model file missing: " << (kModels / f);

	// A link per file (named like the file), links without geometry in between.
	Robot robot;
	robot.names = { "base", files[0], files[1], "tool", files[2] };
	robot.dh = { { 0.3f, 0.0f, 0.0f, kHalfPi, 0.0f, DHJointType::Revolute }, { 0.0f, 0.2f, 0.4f, 0.0f, 0.0f, DHJointType::Revolute },
		{ 0.0f, -0.4f, 0.35f, 0.0f, 0.0f, DHJointType::Revolute }, { 0.1f, 0.0f, 0.0f, 0.0f, 0.0f, DHJointType::Prismatic },
		{ 0.0f, 0.3f, 0.2f, 0.5f, 0.0f, DHJointType::Revolute } };
	robot.cloud_links = files;
	for (const auto& f : files)
		robot.clouds.push_back(read_xyz_file(kModels / f));
	for (const auto& cloud : robot.clouds)
		ASSERT_GT(cloud.size(), 100u);

	for (DHConvention convention : { DHConvention::Classic, DHConvention::Craigs })
	{
		const auto gvl = make();
		ASSERT_TRUE(gvl->add_robot("hollie", convention, robot.names, robot.dh, files, kModels));
		const KinematicChainSharedPtr chain = gvl->get_robot("hollie");
		ASSERT_TRUE(chain);
		EXPECT_EQ(chain->cloud_index("base"), std::nullopt);
		EXPECT_EQ(chain->cloud_index(files[0]), std::optional<uint32_t>(0));
		EXPECT_EQ(chain->cloud_index(files[1]), std::optional<uint32_t>(1));
		EXPECT_EQ(chain->cloud_index("tool"), std::nullopt);
		EXPECT_EQ(chain->cloud_index(files[2]), std::optional<uint32_t>(2));
		EXPECT_TRUE(clouds_equal(chain->link_clouds(), robot.clouds)) << "the files are read as they are (no shift, no scaling)";

		const std::map<std::string, float> joints = set_joints(robot, [](std::size_t i) { return 0.4f - 0.3f * static_cast<float>(i); });
		EXPECT_TRUE(gvl->set_robot_configuration("hollie", joints));
		const Matrix4f base = translation(1.0f, 2.0f, 0.5f);
		EXPECT_TRUE(gvl->set_robot_base_transformation("hollie", base));
		EXPECT_TRUE(clouds_near(chain->transformed_clouds(), world_clouds(convention, robot, base), 3e-4));
		const auto expected = forward_kinematics(convention, robot, base);
		for (std::size_t i = 0; i <= robot.names.size(); ++i)
		{
			Matrix4f got;
			EXPECT_TRUE(gvl->get_robot_transformation("hollie", i, got));
			EXPECT_TRUE(matrices_near(got, expected[i], 3e-5)) << "transform " << i;
		}
		set_joints(robot, [](std::size_t) { return 0.0f; });
	}
}

TEST_F(RobotFacadeTest, AddRules)
{
	const Robot robot = grid_robot(5);
	const auto gvl = make();
	ASSERT_TRUE(add(*gvl, "r", DHConvention::Classic, robot));
	const KinematicChainSharedPtr first = gvl->get_robot("r");

	// A name that exists: false, the first robot stays (also when the second one has other links).
	Robot other = random_robot(3);
	EXPECT_FALSE(add(*gvl, "r", DHConvention::Craigs, other));
	EXPECT_FALSE(gvl->add_robot("r", DHConvention::Craigs, other.names, other.dh, std::vector<std::string>{}, kModels));
	EXPECT_EQ(gvl->get_robot("r"), first);
	EXPECT_EQ(first->num_links(), 4u);
	EXPECT_TRUE(add(*gvl, "r2", DHConvention::Craigs, other)) << "another name is fine";
	EXPECT_NE(gvl->get_robot("r2"), first);
	EXPECT_EQ(gvl->get_robot("r2")->num_links(), 5u);

	// Everything on an unknown robot is false and touches nothing.
	EXPECT_FALSE(gvl->get_robot("nope"));
	std::map<std::string, float> joints{ { "x", 1.0f } };
	Matrix4f matrix = translation(4, 5, 6);
	uint32_t outside = 12345;
	EXPECT_FALSE(gvl->set_robot_configuration("nope", { { "l0", 1.0f } }));
	EXPECT_FALSE(gvl->get_robot_configuration("nope", joints));
	EXPECT_EQ(joints.size(), 1u);
	EXPECT_FALSE(gvl->set_robot_base_transformation("nope", translation(1, 1, 1)));
	EXPECT_FALSE(gvl->get_robot_transformation("nope", 0, matrix));
	EXPECT_EQ(matrix, translation(4, 5, 6));
	EXPECT_FALSE(gvl->update_robot_part("nope", "l0", robot.clouds[0]));
	ASSERT_TRUE(gvl->add_map(MT_BITVECTOR_VOXELMAP, "map"));
	EXPECT_FALSE(gvl->insert_robot_into_map("nope", "map", 6, &outside));
	EXPECT_EQ(outside, 12345u);
	EXPECT_FALSE(gvl->insert_robot_into_map_self_coll_aware("nope", "map").has_value());
	// Known robot, unknown map.
	EXPECT_FALSE(gvl->insert_robot_into_map("r", "no map", 6, &outside));
	EXPECT_EQ(outside, 12345u);
	EXPECT_FALSE(gvl->insert_robot_into_map_self_coll_aware("r", "no map").has_value());
	EXPECT_EQ(snapshot(*gvl->get_map("map")), snapshot(BitVoxelMap(ctx(), kDims, kSide))) << "the map is still empty";

	// Files that cannot be read: false, nothing is registered, the name stays free.
	Robot on_file;
	on_file.names = { "missing/cloud.xyz" };
	on_file.dh = { DHParameters{} };
	EXPECT_FALSE(gvl->add_robot("f", DHConvention::Classic, on_file.names, on_file.dh, on_file.names, kModels));
	on_file.names = { "hollie/arm_0_link.abc" };
	EXPECT_FALSE(gvl->add_robot("f", DHConvention::Classic, on_file.names, on_file.dh, on_file.names, kModels)) << "unknown extension";
	EXPECT_FALSE(gvl->get_robot("f"));
	EXPECT_TRUE(add(*gvl, "f", DHConvention::Classic, robot));

	// Arguments that make no robot throw, also when a file would not be readable (the checks come first), and register nothing.
	const std::vector<std::string> three{ "a", "b", "c" };
	const std::vector<DHParameters> two(2), three_dh(3);
	const std::vector<std::string> none;
	expect_error([&] { (void)gvl->add_robot("e", DHConvention::Classic, three, two, none, kModels); }, "3 link names but 2 sets of DH parameters");
	expect_error([&] { (void)gvl->add_robot("e", DHConvention::Classic, std::vector<std::string>{ "a", "b" }, three_dh, none, kModels); },
		"2 link names but 3 sets of DH parameters");
	expect_error([&] { (void)gvl->add_robot("e", DHConvention::Classic, std::vector<std::string>{ "a", "b", "a" }, three_dh, none, kModels); }, "add_robot: duplicate link name 'a'");
	expect_error([&] { (void)gvl->add_robot("e", DHConvention::Classic, three, three_dh, std::vector<std::string>{ "zzz/not there.xyz" }, kModels); },
		"'zzz/not there.xyz' is no link name");
	expect_error([&] { (void)gvl->add_robot("e", DHConvention::Classic, three, three_dh, std::vector<std::string>{ "c", "a" }, kModels); },
		"the cloud of link 'a' is not listed in link order");
	expect_error([&] { (void)gvl->add_robot("e", DHConvention::Classic, three, three_dh, std::vector<std::string>{ "b", "b" }, kModels); },
		"the cloud of link 'b' is not listed in link order");
	{
		const MetaPointCloud two_clouds = meta(grid_robot(6));
		expect_error([&] { (void)gvl->add_robot("e", DHConvention::Classic, three, three_dh, two_clouds, std::vector<std::string>{ "a" }); },
			"1 cloud names for 3 sub clouds");
		expect_error([&] { (void)gvl->add_robot("e", DHConvention::Classic, three, three_dh, two_clouds, std::vector<std::string>{ "a", "b", "c", "c" }); },
			"4 cloud names for 3 sub clouds");
		expect_error([&] { (void)gvl->add_robot("e", DHConvention::Classic, three, two, two_clouds, std::vector<std::string>{ "a", "b", "c" }); },
			"3 link names but 2 sets of DH parameters");
		expect_error([&] { (void)gvl->add_robot("e", DHConvention::Classic, three, three_dh, two_clouds, std::vector<std::string>{ "a", "x", "c" }); },
			"'x' is no link name");
		expect_error([&] { (void)gvl->add_robot("e", DHConvention::Classic, three, three_dh, two_clouds, std::vector<std::string>{ "a", "c", "b" }); },
			"the cloud of link 'b' is not listed in link order");
	}
	EXPECT_FALSE(gvl->get_robot("e"));

	// No links, no clouds: a robot that is its base.
	EXPECT_TRUE(gvl->add_robot("empty", DHConvention::Classic, none, std::vector<DHParameters>{}, none, kModels));
	EXPECT_EQ(gvl->get_robot("empty")->num_links(), 0u);
	EXPECT_TRUE(gvl->set_robot_base_transformation("empty", translation(1, 2, 3)));
	Matrix4f base;
	EXPECT_TRUE(gvl->get_robot_transformation("empty", 0, base));
	EXPECT_EQ(base, translation(1, 2, 3));
	outside = 12345;
	EXPECT_TRUE(gvl->insert_robot_into_map("empty", "map", 6, &outside));
	EXPECT_EQ(outside, 0u);
	EXPECT_EQ(snapshot(*gvl->get_map("map")), snapshot(BitVoxelMap(ctx(), kDims, kSide))) << "nothing inserted";
	// Links without any cloud: transformations only.
	Robot bare;
	bare.names = { "a", "b" };
	bare.dh = { { 0.5f, 0.0f, 0.0f, 0.0f, 0.0f, DHJointType::Revolute }, { 0.25f, 0.0f, 0.0f, 0.0f, 0.0f, DHJointType::Revolute } };
	EXPECT_TRUE(gvl->add_robot("bare", DHConvention::Classic, bare.names, bare.dh, none, kModels));
	EXPECT_TRUE(gvl->get_robot_transformation("bare", 2, base));
	EXPECT_NEAR(base(2, 3), 0.75f, 1e-6f);
}

TEST_F(RobotFacadeTest, NoRobotBeforeTheContextExists)
{
	const GpuVoxelsSharedPtr instance = GpuVoxels::get_instance();
	ASSERT_FALSE(instance->is_initialized()) << "another test holds the singleton";
	const std::vector<std::string> none;
	expect_error([&] { (void)instance->add_robot("r", DHConvention::Classic, none, std::vector<DHParameters>{}, none, kModels); }, "before initialize()");
	const MetaPointCloud clouds = meta(grid_robot(1));
	expect_error([&] { (void)instance->add_robot("r", DHConvention::Classic, none, std::vector<DHParameters>{}, clouds, none); }, "before initialize()");
	EXPECT_FALSE(instance->get_robot("r"));
}

// An instance that owns its context destroys its robots before the context (a robot holds device buffers).
TEST_F(RobotFacadeTest, OwningInstanceReleasesItsRobotsBeforeItsContext)
{
	const Robot robot = grid_robot(2);
	{
		const GpuVoxelsSharedPtr instance = GpuVoxels::get_instance();
		ASSERT_FALSE(instance->is_initialized()) << "another test holds the singleton";
		instance->initialize(kDims, kSide);
		const MetaPointCloud clouds(instance->context(), std::span<const std::vector<Point3f>>(robot.clouds));
		ASSERT_TRUE(instance->add_robot("r", DHConvention::Classic, robot.names, robot.dh, clouds, robot.cloud_links));
		ASSERT_TRUE(instance->set_robot_base_transformation("r", translation(2.0f, 2.0f, 2.0f)));
		const auto result = instance->get_robot("r")->transformed_clouds().download();
		EXPECT_EQ(result.size(), 36u);
	}
}

TEST_F(RobotFacadeTest, UpdateRobotPart)
{
	Robot robot = grid_robot(7);
	const auto gvl = make();
	ASSERT_TRUE(add(*gvl, "r", DHConvention::Classic, robot));
	const KinematicChainSharedPtr chain = gvl->get_robot("r");
	const Matrix4f base = translation(2.0f, 2.0f, 2.0f);
	ASSERT_TRUE(gvl->set_robot_base_transformation("r", base));
	ASSERT_TRUE(clouds_near(chain->transformed_clouds(), world_clouds(DHConvention::Classic, robot, base), 2e-4));

	// The tool of link l3 is another shape: the next evaluation uses it, the other links stay.
	robot.clouds[2] = std::vector<Point3f>(robot.clouds[2].size(), Point3f{ 0.5f, -0.25f, 0.125f });
	ASSERT_TRUE(gvl->update_robot_part("r", "l3", robot.clouds[2]));
	EXPECT_TRUE(clouds_equal(chain->link_clouds(), robot.clouds));
	EXPECT_TRUE(clouds_near(chain->transformed_clouds(), world_clouds(DHConvention::Classic, robot, base), 2e-4));

	// The same call after a move of the robot.
	const std::map<std::string, float> joints = set_joints(robot, [](std::size_t i) { return kHalfPi * static_cast<float>(i % 3); });
	ASSERT_TRUE(gvl->set_robot_configuration("r", joints));
	robot.clouds[0].assign(robot.clouds[0].size(), Point3f{ 0.0f, 0.25f, 0.0f });
	ASSERT_TRUE(gvl->update_robot_part("r", "l0", robot.clouds[0]));
	EXPECT_TRUE(clouds_near(chain->transformed_clouds(), world_clouds(DHConvention::Classic, robot, base), 2e-4));

	// Refused updates change nothing.
	expect_error([&] { (void)gvl->update_robot_part("r", "l2", robot.clouds[0]); }, "link 'l2' has no cloud");
	expect_error([&] { (void)gvl->update_robot_part("r", "no link", robot.clouds[0]); }, "link 'no link' has no cloud");
	expect_error([&] { (void)gvl->update_robot_part("r", "l1", std::vector<Point3f>(3)); }, "the number of points must not change");
	EXPECT_TRUE(clouds_equal(chain->link_clouds(), robot.clouds));
	EXPECT_TRUE(clouds_near(chain->transformed_clouds(), world_clouds(DHConvention::Classic, robot, base), 2e-4));
}

namespace
{
	struct TypeCase
	{
		MapType type;
		uint8_t meaning;
	};
	const TypeCase kTypes[] = {
		{ MT_BITVECTOR_VOXELMAP, 6 },
		{ MT_PROBAB_VOXELMAP, eBVM_OCCUPIED },
		{ MT_DISTANCE_VOXELMAP, eBVM_OCCUPIED },
		{ MT_BITVECTOR_VOXELLIST, 6 },
		{ MT_COUNTING_VOXELLIST, eBVM_OCCUPIED },
		{ MT_BITVECTOR_OCTREE, eBVM_OCCUPIED },
		{ MT_PROBAB_OCTREE, eBVM_OCCUPIED },
	};

	struct Direct
	{
		uint32_t outside = 0;
		std::vector<uint64_t> state;
	};

	/// The class of `type` built directly, with the clouds of the twin chain inserted if `chain` is given.
	Direct insert_direct(Context& c, MapType type, KinematicChain* chain, uint8_t meaning)
	{
		auto go = [&](auto& m) {
			Direct d;
			if (chain != nullptr)
				d.outside = m.insert_meta_point_cloud(chain->transformed_clouds(), meaning);
			d.state = snapshot(m);
			return d;
		};
		switch (type)
		{
		case MT_BITVECTOR_VOXELMAP: { BitVoxelMap m(c, kDims, kSide); return go(m); }
		case MT_PROBAB_VOXELMAP: { ProbVoxelMap m(c, kDims, kSide); return go(m); }
		case MT_DISTANCE_VOXELMAP: { DistanceVoxelMap m(c, kDims, kSide); return go(m); }
		case MT_BITVECTOR_VOXELLIST: { BitVoxelList m(c, kDims, kSide); return go(m); }
		case MT_COUNTING_VOXELLIST: { CountingVoxelList m(c, kDims, kSide); return go(m); }
		case MT_BITVECTOR_OCTREE: { Octree m(c, OctreeKind::Deterministic, kSide); return go(m); }
		case MT_PROBAB_OCTREE: { Octree m(c, OctreeKind::Probabilistic, kSide); return go(m); }
		default: break;
		}
		ADD_FAILURE() << "type not built";
		return {};
	}
}

TEST_F(RobotFacadeTest, InsertRobotIntoMapIsTheClassWithTheTransformedClouds)
{
	std::mt19937 rng(72);
	std::uniform_int_distribution<int> quarter(0, 3);
	const Matrix4f bases[] = { translation(2.0f, 2.0f, 2.0f), translation(3.9f, 2.0f, 2.0f), translation(0.2f, 2.0f, 0.1f) };

	Robot robot = grid_robot(8);
	const auto gvl = make();
	ASSERT_TRUE(add(*gvl, "r", DHConvention::Classic, robot));
	std::vector<GpuVoxelsMapSharedPtr> maps;
	for (const auto& t : kTypes)
		maps.push_back(gvl->add_map(t.type, std::string("map") + std::to_string(maps.size())));

	bool saw_outside = false;
	for (const Matrix4f& base : bases)
		for (int configuration = 0; configuration < 3; ++configuration)
		{
			const auto joints = set_joints(robot, [&](std::size_t) { return kHalfPi * static_cast<float>(quarter(rng)); });
			ASSERT_TRUE(gvl->set_robot_configuration("r", joints));
			ASSERT_TRUE(gvl->set_robot_base_transformation("r", base));

			// What the host expects for the bit map: the reference kinematics, the points rounded to float, inserted one by one.
			const auto world = world_clouds(DHConvention::Classic, robot, base);
			ref::BitVoxelMapReference reference(kDims.x, kDims.y, kDims.z, kSide);
			uint32_t expected_outside = 0;
			for (const auto& points : world)
				expected_outside += reference.insert_points(points, 6);

			for (std::size_t i = 0; i < std::size(kTypes); ++i)
			{
				const auto& t = kTypes[i];
				maps[i]->clear_map();
				KinematicChain chain = twin(DHConvention::Classic, robot, base);
				const Direct expected = insert_direct(ctx(), t.type, &chain, t.meaning);

				uint32_t outside = 12345;
				ASSERT_TRUE(gvl->insert_robot_into_map("r", "map" + std::to_string(i), t.meaning, &outside)) << map_type_name(t.type);
				EXPECT_EQ(outside, expected.outside) << map_type_name(t.type) << " configuration " << configuration;
				EXPECT_EQ(snapshot(*maps[i]), expected.state) << map_type_name(t.type) << " configuration " << configuration;
				if (t.type == MT_BITVECTOR_VOXELMAP)
				{
					EXPECT_EQ(outside, expected_outside);
					EXPECT_TRUE(maps[i]->as<BitVoxelMap>()->download_words() == reference.to_words(0, maps[i]->as<BitVoxelMap>()->voxel_count()))
						<< "the voxels differ from the host reference";
				}
				saw_outside |= expected.outside > 0;
			}
			// Without the counter the call is the same.
			EXPECT_TRUE(gvl->insert_robot_into_map("r", "map0", 6));
		}
	EXPECT_TRUE(saw_outside) << "some sampled poses should reach out of the maps";
	for (std::size_t i = 0; i < std::size(kTypes); ++i)
		EXPECT_FALSE(snapshot(*maps[i]).empty());
}

TEST_F(RobotFacadeTest, SelfCollisionAwareInsertionMatchesReferenceAndClass)
{
	std::mt19937 rng(73);
	std::uniform_int_distribution<int> quarter(0, 3);
	Robot robot = grid_robot(9);
	const Matrix4f base = translation(2.0f, 2.0f, 2.0f);
	const auto gvl = make();
	ASSERT_TRUE(add(*gvl, "r", DHConvention::Classic, robot));
	ASSERT_TRUE(gvl->set_robot_base_transformation("r", base));
	const auto map = gvl->add_map(MT_BITVECTOR_VOXELMAP, "map");
	ASSERT_TRUE(map);

	// Mode 0: the defaults. Mode 1: own meanings, every link only collides with its neighbours. Mode 2: own meanings, nobody collides with anybody.
	const std::vector<uint8_t> meanings{ 10, 11, 12 };
	std::vector<std::bitset<256>> ref_masks[3];
	std::vector<BitVector256> masks[3];
	for (int mode = 1; mode <= 2; ++mode)
		for (std::size_t k = 0; k < 3; ++k)
		{
			std::bitset<256> mask;
			if (mode == 1 && k + 1 < 3)
				mask.set(meanings[k + 1]);
			if (mode == 1 && k > 0)
				mask.set(meanings[k - 1]);
			ref_masks[mode].push_back(mask);
			const auto words = words_of(mask);
			masks[mode].push_back({ words[0], words[1], words[2], words[3], words[4], words[5], words[6], words[7] });
		}

	bool saw_collision = false, saw_free = false;
	for (int configuration = 0; configuration < 10; ++configuration)
	{
		const auto joints = set_joints(robot, [&](std::size_t) { return kHalfPi * static_cast<float>(quarter(rng)); });
		ASSERT_TRUE(gvl->set_robot_configuration("r", joints));
		const auto world = world_clouds(DHConvention::Classic, robot, base);
		for (const int mode : { 0, 1, 2 })
		{
			map->clear_map();
			ref::BitVoxelMapReference reference(kDims.x, kDims.y, kDims.z, kSide);
			const auto expected = mode != 0 ? reference.insert_meta_clouds_self_collision(world, meanings, ref_masks[mode]) : reference.insert_meta_clouds_self_collision(world);

			const auto result = mode != 0 ? gvl->insert_robot_into_map_self_coll_aware("r", "map", meanings, masks[mode]) : gvl->insert_robot_into_map_self_coll_aware("r", "map");
			ASSERT_TRUE(result.has_value());
			EXPECT_EQ(result->colliding, expected.colliding) << "configuration " << configuration << " mode " << mode;
			EXPECT_EQ(std::vector<uint32_t>(result->colliding_meanings.begin(), result->colliding_meanings.end()), words_of(expected.meanings));
			EXPECT_EQ(result->points_outside, expected.outside);
			const auto words = map->as<BitVoxelMap>()->download_words();
			EXPECT_TRUE(words == reference.to_words(0, map->as<BitVoxelMap>()->voxel_count())) << "voxels differ, configuration " << configuration << " mode " << mode;

			// The class used directly with a twin chain gives the same result.
			KinematicChain chain = twin(DHConvention::Classic, robot, base);
			BitVoxelMap direct(ctx(), kDims, kSide);
			const auto direct_result = mode != 0 ? direct.insert_meta_point_cloud_with_self_collision_check(chain.transformed_clouds(), meanings, masks[mode])
											  : direct.insert_meta_point_cloud_with_self_collision_check(chain.transformed_clouds());
			EXPECT_EQ(result->colliding, direct_result.colliding);
			EXPECT_EQ(result->colliding_meanings, direct_result.colliding_meanings);
			EXPECT_EQ(result->points_outside, direct_result.points_outside);
			EXPECT_TRUE(words == direct.download_words());
			if (mode == 2)
				EXPECT_FALSE(result->colliding) << "nobody is a collision partner";
			if (mode == 0)
				(expected.colliding ? saw_collision : saw_free) = true;
		}
	}
	EXPECT_TRUE(saw_collision && saw_free) << "the sampled poses should include colliding and free ones";

	// Part of the robot outside the map is counted.
	ASSERT_TRUE(gvl->set_robot_base_transformation("r", translation(3.9f, 2.0f, 2.0f)));
	map->clear_map();
	const auto outside_result = gvl->insert_robot_into_map_self_coll_aware("r", "map");
	ASSERT_TRUE(outside_result.has_value());
	ref::BitVoxelMapReference reference(kDims.x, kDims.y, kDims.z, kSide);
	EXPECT_EQ(outside_result->points_outside, reference.insert_meta_clouds_self_collision(world_clouds(DHConvention::Classic, robot, translation(3.9f, 2.0f, 2.0f))).outside);
	EXPECT_GT(outside_result->points_outside, 0u);

	// Argument errors of the map class come through.
	expect_error([&] { (void)gvl->insert_robot_into_map_self_coll_aware("r", "map", std::vector<uint8_t>{ 10, 11 }); }, "expected one meaning per sub cloud");
}

TEST_F(RobotFacadeTest, SelfCollisionAwareInsertionNeedsABitMap)
{
	Robot robot = grid_robot(10);
	const auto gvl = make();
	ASSERT_TRUE(add(*gvl, "r", DHConvention::Classic, robot));
	ASSERT_TRUE(gvl->set_robot_base_transformation("r", translation(2.0f, 2.0f, 2.0f)));
	for (const auto& t : kTypes)
	{
		if (t.type == MT_BITVECTOR_VOXELMAP)
			continue;
		const std::string name = map_type_name(t.type);
		const auto map = gvl->add_map(t.type, name);
		const auto before = snapshot(*map);
		expect_error([&] { (void)gvl->insert_robot_into_map_self_coll_aware("r", name); },
			std::string("'insert_robot_into_map_self_coll_aware' is not supported by the map type ") + name);
		EXPECT_EQ(snapshot(*map), before) << name << " stays as it was";
	}
}
