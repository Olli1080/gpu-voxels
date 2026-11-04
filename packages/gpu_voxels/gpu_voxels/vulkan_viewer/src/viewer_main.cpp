// -- BEGIN LICENSE BLOCK ----------------------------------------------
// This file is part of the GPU Voxels Software Library.
// Licensed under the CDDL (see LICENSE.txt in the top directory).
// -- END LICENSE BLOCK ------------------------------------------------

/*!\file
 * \brief  gvl_viewer: shows a demo scene or a point cloud file in an in-process Vulkan window (or renders it to a PNG without a window).
 *
 *   gvl_viewer [--points <file.xyz|.pcd|.binvox>] [--side <metres>] [--prob] [--octree [--level <N>]] [--facade]
 *              [--png <out.png> [--size <W>x<H>]] [--frames <N>] [--no-vsync] [--validation]
 *
 * --facade shows a GpuVoxels scene through the visualization hooks (visualize_map, visualize_primitives_array): a bit map, an octree, moving spheres and
 * cuboids and a three link robot as layers; the keys 1 .. 5 show / hide them.
 *
 * Mouse: left button rotates, right / middle button pans, wheel zooms. Keys: S toggles the shading, R resets the camera, ESC closes.
 */
#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <memory>
#include <random>
#include <string>
#include <vector>

#include <gpu_voxels/vulkan/BitVoxelMap.h>
#include <gpu_voxels/vulkan/GpuVoxels.h>
#include <gpu_voxels/vulkan/ImageFile.h>
#include <gpu_voxels/vulkan/Octree.h>
#include <gpu_voxels/vulkan/PointCloudFile.h>
#include <gpu_voxels/vulkan/ProbVoxelMap.h>
#include <gpu_voxels/vulkan/Viewer.h>
#include <gpu_voxels/vulkan/ViewerScene.h>
#include <gpu_voxels/vulkan_viewer/ViewerWindow.h>

using namespace gpu_voxels::vulkan;

namespace
{
	struct Arguments
	{
		std::string points_file;
		float side = 0.05f;
		bool prob = false;
		bool facade = false;   ///< layers of a GpuVoxels through ViewerScene and the visualize_* calls
		bool octree = false;   ///< show the points as an octree (deterministic, with --prob probabilistic) instead of a voxel map
		uint32_t level = 0;    ///< --octree: nodes of this level are drawn as one cube
		std::string png;
		uint32_t width = 1280, height = 720;
		uint64_t frames = 0; ///< 0 = until the window is closed
		bool vsync = true;
		bool validation = false;
	};

	Arguments parse(int argc, char** argv)
	{
		Arguments a;
		for (int i = 1; i < argc; ++i)
		{
			const std::string arg = argv[i];
			const auto value = [&]() -> std::string
			{
				if (i + 1 >= argc)
					throw std::runtime_error("missing value for " + arg);
				return argv[++i];
			};
			if (arg == "--points") a.points_file = value();
			else if (arg == "--octree") a.octree = true;
			else if (arg == "--facade") a.facade = true;
			else if (arg == "--level") a.level = static_cast<uint32_t>(std::stoul(value()));
			else if (arg == "--side") a.side = std::stof(value());
			else if (arg == "--prob") a.prob = true;
			else if (arg == "--png") a.png = value();
			else if (arg == "--size")
			{
				const std::string size = value();
				const auto x = size.find('x');
				if (x == std::string::npos)
					throw std::runtime_error("--size expects <W>x<H>");
				a.width = static_cast<uint32_t>(std::stoul(size.substr(0, x)));
				a.height = static_cast<uint32_t>(std::stoul(size.substr(x + 1)));
			}
			else if (arg == "--frames") a.frames = std::stoull(value());
			else if (arg == "--no-vsync") a.vsync = false;
			else if (arg == "--validation") a.validation = true;
			else throw std::runtime_error("unknown argument " + arg);
		}
		if (!(a.side > 0.0f))
			throw std::runtime_error("--side must be positive");
		return a;
	}

	/// The points of the scene in metres: a file, or a demo (ground plate, a sphere, a column and scattered noise).
	std::vector<Point3f> scene_points(const Arguments& a)
	{
		if (!a.points_file.empty())
		{
			auto points = read_point_cloud_file(a.points_file);
			if (points.empty())
				throw std::runtime_error("the point cloud file has no points");
			shift_point_cloud_to_zero(points);
			return points;
		}
		std::vector<Point3f> points;
		const float step = a.side;
		for (float x = 0.0f; x < 4.0f; x += step)
			for (float y = 0.0f; y < 4.0f; y += step)
				points.push_back({ x, y, 0.0f });
		for (float x = 0.0f; x < 4.0f; x += step)
			for (float y = 0.0f; y < 4.0f; y += step)
				for (float z = step; z < 1.6f; z += step)
				{
					const float dx = x - 2.0f, dy = y - 1.5f, dz = z - 0.75f;
					const float r2 = dx * dx + dy * dy + dz * dz;
					if (r2 < 0.5f * 0.5f && r2 > 0.46f * 0.46f)
						points.push_back({ x, y, z });
					if (std::abs(x - 3.4f) < step && std::abs(y - 3.2f) < step)
						points.push_back({ x, y, z });
				}
		std::mt19937 rng(1);
		std::uniform_real_distribution<float> dist(0.0f, 4.0f);
		for (int i = 0; i < 500; ++i)
			points.push_back({ dist(rng), dist(rng), dist(rng) * 0.4f });
		return points;
	}

	struct Scene
	{
		std::unique_ptr<BitVoxelMap> bit_map;
		std::unique_ptr<ProbVoxelMap> prob_map;
		std::unique_ptr<Octree> octree;
		OctreeDisplay octree_display;
		std::array<float, 3> box_min{}, box_max{};
	};

	Scene build_scene(Context& context, const Arguments& a)
	{
		const auto points = scene_points(a);
		Point3f max{ 0.0f, 0.0f, 0.0f };
		for (const Point3f& p : points)
		{
			max.x = std::max(max.x, p.x);
			max.y = std::max(max.y, p.y);
			max.z = std::max(max.z, p.z);
		}
		const auto cells = [&](float extent) { return static_cast<uint32_t>(std::floor(extent / a.side)) + 2; };
		const Dim3ui dims{ cells(max.x), cells(max.y), cells(max.z) };
		Scene scene;
		scene.box_max = { static_cast<float>(dims.x) * a.side, static_cast<float>(dims.y) * a.side, static_cast<float>(dims.z) * a.side };
		if (a.octree)
		{
			scene.octree = std::make_unique<Octree>(context, a.prob ? OctreeKind::Probabilistic : OctreeKind::Deterministic, a.side);
			scene.octree->insert_point_cloud(points, eBVM_OCCUPIED);
			scene.octree_display.min_level = a.level;
		}
		else if (a.prob)
		{
			scene.prob_map = std::make_unique<ProbVoxelMap>(context, dims, a.side);
			scene.prob_map->insert_point_cloud(points, eBVM_OCCUPIED);
		}
		else
		{
			scene.bit_map = std::make_unique<BitVoxelMap>(context, dims, a.side);
			scene.bit_map->insert_point_cloud(points, 1);
		}
		return scene;
	}

	void extract(CubeExtractor& extractor, const Scene& scene, CubeBuffer& cubes)
	{
		if (scene.octree)
			extractor.extract(*scene.octree, scene.octree_display, cubes);
		else if (scene.prob_map)
			extractor.extract(*scene.prob_map, ProbDisplay{}, cubes);
		else
			extractor.extract(*scene.bit_map, BitPalette::defaults(), cubes);
	}

	/// The façade demo: the scene points as a bit map, a box as an octree, spheres that move and cuboids; shown through ViewerScene and the visualize calls.
	int run_facade(const Arguments& a)
	{
		const bool window_mode = a.png.empty();
		ContextCreateInfo base;
		base.enable_validation = a.validation;
		base.enable_graphics = true;
		Context context(window_mode ? ViewerWindow::context_requirements(base) : base);
		const auto cells = [&](float metres) { return static_cast<uint32_t>(std::lround(metres / a.side)) + 2; };
		GpuVoxels gvl(context);
		gvl.initialize({ cells(4.0f), cells(4.0f), cells(2.0f) }, a.side);
		gvl.add_map(MT_BITVECTOR_VOXELMAP, "environment");
		gvl.add_map(MT_BITVECTOR_OCTREE, "octree");
		gvl.insert_point_cloud_into_map(scene_points(a), "environment", 1);
		gvl.insert_box_into_map({ 0.2f, 0.2f, 0.0f }, { 0.8f, 0.8f, 0.5f }, "octree", eBVM_OCCUPIED);
		gvl.add_primitives(ePRIM_SPHERE, "spheres");
		gvl.add_primitives(ePRIM_CUBOID, "cuboids");
		gvl.modify_primitives("cuboids", std::vector<Point3f>{ { 3.0f, 0.6f, 0.2f }, { 3.4f, 0.6f, 0.2f }, { 3.2f, 0.6f, 0.6f } }, 0.4f);
		const auto move_spheres = [&](double angle)
		{
			std::vector<Point3f> centres;
			for (int i = 0; i < 4; ++i)
			{
				const double phase = angle + i * 1.5707963;
				centres.push_back({ static_cast<float>(2.0 + 1.6 * std::cos(phase)), static_cast<float>(2.0 + 1.6 * std::sin(phase)), 1.0f + 0.2f * i });
			}
			gvl.modify_primitives("spheres", centres, 0.4f);
		};
		move_spheres(0.0);

		// a robot with three links (the first has no geometry): clouds along the local x axis of the two arms, which swing with the frames
		{
			std::vector<Point3f> arm;
			for (int i = 0; i <= 30; ++i)
				arm.push_back({ 0.03f * static_cast<float>(i), 0.0f, 0.0f });
			std::vector<Point3f> arm2;
			for (int i = 0; i <= 20; ++i)
				arm2.push_back({ 0.03f * static_cast<float>(i), 0.0f, 0.0f });
			const std::vector<std::vector<Point3f>> sub_clouds{ arm, arm2 };
			const MetaPointCloud clouds(context, std::span<const std::vector<Point3f>>(sub_clouds));
			std::vector<DHParameters> dh(3);
			dh[0].d = 0.7f;
			dh[1].a = 0.9f;
			dh[2].a = 0.6f;
			gvl.add_robot("arm", DHConvention::Classic, std::vector<std::string>{ "base", "upper", "lower" }, dh, clouds, std::vector<std::string>{ "upper", "lower" });
			Matrix4f base;
			base(0, 3) = 1.2f;
			base(1, 3) = 3.0f;
			gvl.set_robot_base_transformation("arm", base);
		}

		ViewerScene scene(gvl, 1u << 21);
		for (const char* name : { "environment", "octree" })
			gvl.visualize_map(name);
		gvl.visualize_primitives_array("spheres");
		gvl.visualize_primitives_array("cuboids");
		RobotDisplay robot_style;
		robot_style.snap_to_voxels = false;   // the points as small cubes, not the voxels of the 5 cm grid
		robot_style.point_size = 0.07f;
		robot_style.colours = { pack_colour(240, 200, 40), pack_colour(230, 60, 200) };
		scene.show_robot("arm", robot_style);
		const std::size_t robot_layer = scene.layer_count() - 1;
		const auto layers = scene.layers();
		for (std::size_t i = 0; i < layers.size(); ++i)
			std::cout << "layer " << i + 1 << ": " << layers[i].name << (layers[i].kind == ViewerScene::LayerKind::Primitives ? " (primitives)" : layers[i].kind == ViewerScene::LayerKind::Robot ? " (robot)" : "") << ", " << layers[i].cubes << " cubes\n";

		OrbitCamera camera;
		camera.frame({ 0.0f, 0.0f, 0.0f }, scene.world_size());
		if (!window_mode)
		{
			VoxelRenderer renderer(context, OffscreenTarget::kColourFormat);
			OffscreenTarget target(context, a.width, a.height);
			const auto buffers = scene.buffers();
			target.render(renderer, camera.camera(), buffers);
			write_png(a.png, a.width, a.height, target.read_rgba());
			std::cout << "wrote " << a.png << "\n";
			return 0;
		}
		ViewerWindowOptions options;
		options.width = a.width;
		options.height = a.height;
		options.vsync = a.vsync;
		ViewerWindow window(context, options);
		window.camera() = camera;
		window.set_home_camera();
		window.set_key_handler([&](int key)
		{
			if (key >= '1' && key <= '9')
				scene.toggle_visible(static_cast<std::size_t>(key - '1'));
		});
		while (!window.should_close() && (a.frames == 0 || window.frames_presented() < a.frames))
		{
			window.poll_events();
			move_spheres(static_cast<double>(window.frames_presented()) * 0.02);
			gvl.visualize_primitives_array("spheres", true);   // extracts the new positions
			const float swing = static_cast<float>(window.frames_presented()) * 0.02f;
			gvl.set_robot_configuration("arm", { { "upper", 0.8f * std::sin(swing) }, { "lower", 1.0f * std::sin(1.7f * swing) } });
			scene.refresh_layer(robot_layer);   // the robot is read in its new pose
			const auto buffers = scene.buffers();
			window.draw(buffers);
		}
		std::cout << "frames: " << window.frames_presented() << "\n";
		return 0;
	}

	int run(const Arguments& a)
	{
		if (a.facade)
			return run_facade(a);
		if (!a.png.empty())
		{
			// no window: render into an image
			ContextCreateInfo info;
			info.enable_graphics = true;
			info.enable_validation = a.validation;
			Context context(info);
			Scene scene = build_scene(context, a);
			CubeExtractor extractor(context);
			CubeBuffer cubes(context, 1u << 22);
			extract(extractor, scene, cubes);
			std::cout << "cubes: " << cubes.count() << (cubes.dropped() ? " (" + std::to_string(cubes.dropped()) + " dropped)" : "") << "\n";

			OrbitCamera camera;
			camera.frame(scene.box_min, scene.box_max);
			VoxelRenderer renderer(context, OffscreenTarget::kColourFormat);
			OffscreenTarget target(context, a.width, a.height);
			const CubeBuffer* buffers[] = { &cubes };
			target.render(renderer, camera.camera(), buffers);
			write_png(a.png, a.width, a.height, target.read_rgba());
			std::cout << "wrote " << a.png << "\n";
			return 0;
		}

		ContextCreateInfo base;
		base.enable_validation = a.validation;
		Context context(ViewerWindow::context_requirements(base));
		std::cout << "device: " << context.info().name << "\n";
		Scene scene = build_scene(context, a);
		CubeExtractor extractor(context);
		CubeBuffer cubes(context, 1u << 22);
		extract(extractor, scene, cubes);
		std::cout << "cubes: " << cubes.count() << (cubes.dropped() ? " (" + std::to_string(cubes.dropped()) + " dropped)" : "") << "\n";

		ViewerWindowOptions options;
		options.width = a.width;
		options.height = a.height;
		options.vsync = a.vsync;
		ViewerWindow window(context, options);
		window.camera().frame(scene.box_min, scene.box_max);
		window.set_home_camera();
		const CubeBuffer* buffers[] = { &cubes };
		while (!window.should_close() && (a.frames == 0 || window.frames_presented() < a.frames))
		{
			window.poll_events();
			window.draw(buffers);
		}
		std::cout << "frames: " << window.frames_presented() << "\n";
		return 0;
	}
}

int main(int argc, char** argv)
{
	try
	{
		return run(parse(argc, argv));
	}
	catch (const std::exception& e)
	{
		std::cerr << "gvl_viewer: " << e.what() << "\n";
		return 1;
	}
}
