// Slice S2b: DH kinematic chain whose link clouds are moved on the GPU, against an independent double precision
// forward kinematics (tests/reference/kinematics_reference.h).
#include "vulkan_fixture.h"

#include <array>
#include <cmath>
#include <numbers>
#include <random>
#include <vector>

#include <gpu_voxels/vulkan/BitVoxelMap.h>
#include <gpu_voxels/vulkan/KinematicChain.h>

#include "../reference/bit_voxel_map_reference.h"
#include "../reference/kinematics_reference.h"

using namespace gpu_voxels::vulkan;
using gvl_test::VulkanTest;
namespace ref = gvl_reference;

namespace
{
	constexpr double kPi = std::numbers::pi;

	ref::DH to_reference(DHConvention c, const DHParameters& p)
	{
		return { p.d, p.theta, p.a, p.alpha, p.value, p.joint_type == DHJointType::Prismatic, c == DHConvention::Craigs };
	}

	::testing::AssertionResult matrices_near(const Matrix4f& got, const ref::Mat4& expected, double tolerance)
	{
		for (int r = 0; r < 4; ++r)
			for (int c = 0; c < 4; ++c)
				if (std::abs(got(r, c) - expected.at(r, c)) > tolerance)
					return ::testing::AssertionFailure() << "element (" << r << "," << c << "): got " << got(r, c) << " expected " << expected.at(r, c);
		return ::testing::AssertionSuccess();
	}

	ref::Mat4 to_ref(const Matrix4f& f)
	{
		ref::Mat4 m;
		for (int r = 0; r < 4; ++r)
			for (int c = 0; c < 4; ++c)
				m.at(r, c) = f(r, c);
		return m;
	}

	std::vector<Point3f> random_cloud(std::mt19937& rng, int n, float range)
	{
		std::uniform_real_distribution<float> coord(-range, range);
		std::vector<Point3f> cloud(n);
		for (auto& p : cloud)
			p = { coord(rng), coord(rng), coord(rng) };
		return cloud;
	}

	DHParameters random_dh(std::mt19937& rng, bool prismatic)
	{
		std::uniform_real_distribution<float> angle(-3.1f, 3.1f), length(-0.6f, 0.6f);
		return { length(rng), angle(rng), length(rng), angle(rng), angle(rng), prismatic ? DHJointType::Prismatic : DHJointType::Revolute };
	}
}

TEST_F(VulkanTest, DhMatricesMatchIndependentComposition)
{
	std::mt19937 rng(30);
	for (DHConvention convention : { DHConvention::Classic, DHConvention::Craigs })
		for (bool prismatic : { false, true })
			for (int i = 0; i < 50; ++i)
			{
				const DHParameters p = random_dh(rng, prismatic);
				EXPECT_TRUE(matrices_near(dh_matrix(convention, p), ref::dh_matrix(to_reference(convention, p)), 2e-6))
					<< "convention " << static_cast<int>(convention) << " prismatic " << prismatic << " sample " << i;
			}
}

TEST_F(VulkanTest, ChainTransformsAndCloudsMatchReference)
{
	std::mt19937 rng(31);
	for (DHConvention convention : { DHConvention::Classic, DHConvention::Craigs })
	{
		std::vector<LinkSpec> links;
		std::vector<ref::DH> ref_links;
		for (int i = 0; i < 6; ++i)
		{
			LinkSpec link{ "link" + std::to_string(i), random_dh(rng, i == 4), std::nullopt };
			if (i != 3) // link 3 has no point cloud
				link.cloud = random_cloud(rng, 50 + 20 * i, 0.5f);
			links.push_back(link);
			ref_links.push_back(to_reference(convention, link.dh));
		}
		ref::Mat4 base = ref::translation(0.3, -0.2, 0.5) * ref::rotation_z(0.7);
		Matrix4f base_f;
		for (int r = 0; r < 4; ++r)
			for (int c = 0; c < 4; ++c)
				base_f(r, c) = static_cast<float>(base.at(r, c));
		base = to_ref(base_f);

		KinematicChain chain(ctx(), convention, links, base_f);
		EXPECT_EQ(chain.num_links(), 6u);
		EXPECT_EQ(chain.cloud_index("link2"), std::optional<uint32_t>(2));
		EXPECT_EQ(chain.cloud_index("link3"), std::nullopt);
		EXPECT_EQ(chain.cloud_index("link4"), std::optional<uint32_t>(3)); // clouds are numbered over links that have one
		EXPECT_EQ(chain.cloud_index("nope"), std::nullopt);

		for (int round = 0; round < 3; ++round)
		{
			// Joint values: change some, keep others.
			std::map<std::string, float> joints;
			std::uniform_real_distribution<float> angle(-2.5f, 2.5f);
			for (int i = 0; i < 6; ++i)
				if (round == 0 || i % (round + 1) == 0)
				{
					joints["link" + std::to_string(i)] = angle(rng);
					ref_links[i].value = joints["link" + std::to_string(i)];
				}
			chain.set_configuration(joints);
			const auto expected_transforms = ref::forward_kinematics(base, ref_links);
			for (std::size_t i = 0; i <= 6; ++i)
				EXPECT_TRUE(matrices_near(chain.transform(i), expected_transforms[i], 3e-5)) << "transform " << i << " round " << round;

			// Clouds: link i's cloud is moved with transform(i).
			const MetaPointCloud& moved = chain.transformed_clouds();
			ASSERT_EQ(moved.num_clouds(), 5u);
			uint32_t cloud = 0;
			for (int i = 0; i < 6; ++i)
			{
				if (!links[i].cloud)
					continue;
				const auto got = moved.download_cloud(cloud);
				ASSERT_EQ(got.size(), links[i].cloud->size());
				for (std::size_t k = 0; k < got.size(); ++k)
				{
					const auto e = expected_transforms[i].apply((*links[i].cloud)[k]);
					ASSERT_NEAR(got[k].x, e[0], 2e-4) << "link " << i << " point " << k << " round " << round;
					ASSERT_NEAR(got[k].y, e[1], 2e-4);
					ASSERT_NEAR(got[k].z, e[2], 2e-4);
				}
				++cloud;
			}
		}
	}
}

TEST_F(VulkanTest, ChainBaseTransformationConfigurationAndUpdates)
{
	std::mt19937 rng(32);
	std::vector<LinkSpec> links{
		{ "a", { 0.1f, 0.0f, 0.2f, 0.0f, 0.0f, DHJointType::Revolute }, random_cloud(rng, 20, 0.3f) },
		{ "b", { 0.0f, 0.0f, 0.3f, 0.0f, 0.0f, DHJointType::Prismatic }, random_cloud(rng, 20, 0.3f) } };
	KinematicChain chain(ctx(), DHConvention::Classic, links);

	auto first = chain.transformed_clouds().download();
	auto again = chain.transformed_clouds().download();
	for (std::size_t i = 0; i < first.size(); ++i)
		EXPECT_TRUE(first[i].x == again[i].x && first[i].y == again[i].y && first[i].z == again[i].z) << "no change => same result";

	// Prismatic joint moves link b along z; link a (frame 0 = base) does not move.
	chain.set_configuration({ { "b", 0.5f }, { "unknown", 1.0f } });
	EXPECT_EQ(chain.configuration().at("b"), 0.5f);
	EXPECT_EQ(chain.configuration().count("unknown"), 0u);
	auto moved = chain.transformed_clouds().download();
	for (std::size_t i = 0; i < 20; ++i)
		EXPECT_EQ(moved[i].z, first[i].z) << "link a stays at the base";
	// transform(2) (end effector) changed by the prismatic value along the local z axis: d + value
	EXPECT_NEAR(chain.transform(2)(2, 3) - dh_matrix(DHConvention::Classic, links[0].dh)(2, 3), 0.5f, 1e-6);

	// Base transformation shifts everything.
	Matrix4f shifted = Matrix4f::identity();
	shifted(0, 3) = 10.0f;
	chain.set_base_transformation(shifted);
	const auto after_base = chain.transformed_clouds().download();
	EXPECT_NEAR(after_base[0].x, moved[0].x + 10.0f, 1e-5);

	// A replaced link cloud is used in the next evaluation.
	std::vector<Point3f> replacement(20, Point3f{ 1.0f, 2.0f, 3.0f });
	chain.update_link_cloud("a", replacement);
	const auto replaced = chain.link_clouds().download_cloud(0);
	EXPECT_EQ(replaced[7].y, 2.0f);
	EXPECT_NEAR(chain.transformed_clouds().download_cloud(0)[7].x, 11.0f, 1e-5);
}

TEST_F(VulkanTest, ChainRejectsInvalidArguments)
{
	std::vector<LinkSpec> links{ { "a", {}, std::vector<Point3f>(3) }, { "b", {}, std::nullopt } };
	KinematicChain chain(ctx(), DHConvention::Craigs, links);
	EXPECT_THROW((void)chain.transform(3), VulkanError);
	EXPECT_NO_THROW((void)chain.transform(2));
	EXPECT_THROW(chain.update_link_cloud("b", std::vector<Point3f>(3)), VulkanError); // no cloud
	EXPECT_THROW(chain.update_link_cloud("a", std::vector<Point3f>(2)), VulkanError); // size change
	std::vector<LinkSpec> duplicate{ { "x", {}, std::nullopt }, { "x", {}, std::nullopt } };
	EXPECT_THROW(KinematicChain(ctx(), DHConvention::Classic, duplicate), VulkanError);
}

// Robot -> voxel map with self collision detection. Angles are multiples of 90 degrees and all lengths multiples of
// the voxel size, so the transformed points stay far (>= 0.02 m) from voxel boundaries and the float / double
// difference of the transformation cannot change the voxel.
TEST_F(VulkanTest, RobotInsertionWithSelfCollisionMatchesReference)
{
	std::mt19937 rng(33);
	const float side = 0.1f;
	constexpr Dim3ui dims{ 40, 40, 40 };
	std::uniform_int_distribution<int> cell(-3, 3);
	std::uniform_real_distribution<float> jitter(-0.03f, 0.03f);
	auto link_cloud = [&] {
		std::vector<Point3f> cloud;
		for (int i = 0; i < 12; ++i)
			cloud.push_back({ 0.1f * static_cast<float>(cell(rng)) + 0.05f + jitter(rng), 0.1f * static_cast<float>(cell(rng)) + 0.05f + jitter(rng),
				0.1f * static_cast<float>(cell(rng)) + 0.05f + jitter(rng) });
		return cloud;
	};

	const float half_pi = static_cast<float>(kPi / 2);
	std::vector<LinkSpec> links{
		{ "l0", { 0.2f, 0.0f, 0.1f, half_pi, 0.0f, DHJointType::Revolute }, link_cloud() },
		{ "l1", { 0.0f, 0.0f, 0.3f, 0.0f, 0.0f, DHJointType::Revolute }, link_cloud() },
		{ "l2", { 0.0f, 0.0f, 0.2f, -half_pi, 0.0f, DHJointType::Revolute }, link_cloud() },
		{ "l3", { 0.1f, 0.0f, 0.0f, 0.0f, 0.0f, DHJointType::Revolute }, link_cloud() } };
	Matrix4f base = Matrix4f::identity();
	base(0, 3) = 2.0f; base(1, 3) = 2.0f; base(2, 3) = 2.0f;
	KinematicChain chain(ctx(), DHConvention::Classic, links, base);

	std::uniform_int_distribution<int> quarter(0, 3);
	bool saw_collision = false, saw_free = false;
	for (int configuration = 0; configuration < 8; ++configuration)
	{
		std::map<std::string, float> joints;
		std::vector<ref::DH> ref_links;
		for (std::size_t i = 0; i < links.size(); ++i)
		{
			links[i].dh.value = half_pi * static_cast<float>(quarter(rng));
			joints[links[i].name] = links[i].dh.value;
			ref_links.push_back(to_reference(DHConvention::Classic, links[i].dh));
		}
		chain.set_configuration(joints);
		const auto transforms = ref::forward_kinematics(to_ref(base), ref_links);

		// Reference: double precision FK, then insert the (float rounded) world points on the CPU.
		std::vector<std::vector<Point3f>> world;
		for (std::size_t i = 0; i < links.size(); ++i)
		{
			std::vector<Point3f> points;
			for (const auto& p : *links[i].cloud)
			{
				const auto q = transforms[i].apply(p);
				points.push_back({ static_cast<float>(q[0]), static_cast<float>(q[1]), static_cast<float>(q[2]) });
			}
			world.push_back(points);
		}
		ref::BitVoxelMapReference reference(dims.x, dims.y, dims.z, side);
		const auto expected = reference.insert_meta_clouds_self_collision(world);

		BitVoxelMap map(ctx(), dims, side);
		const auto result = map.insert_meta_point_cloud_with_self_collision_check(chain.transformed_clouds());
		EXPECT_EQ(result.colliding, expected.colliding) << "configuration " << configuration;
		EXPECT_EQ(result.points_outside, expected.outside);

		std::vector<uint32_t> expected_words(8, 0);
		for (uint32_t b = 0; b < 256; ++b)
			if (expected.meanings.test(b))
				expected_words[b / 32] |= 1u << (b % 32);
		EXPECT_EQ(std::vector<uint32_t>(result.colliding_meanings.begin(), result.colliding_meanings.end()), expected_words);
		const auto gpu = map.download_words();
		const auto cpu = reference.to_words(0, map.voxel_count());
		EXPECT_TRUE(gpu == cpu) << "voxel maps differ in configuration " << configuration;
		(expected.colliding ? saw_collision : saw_free) = true;
	}
	EXPECT_TRUE(saw_collision && saw_free) << "the sampled configurations should include colliding and free ones";
}
