// -- BEGIN LICENSE BLOCK ----------------------------------------------
// This file is part of the GPU Voxels Software Library.
// Licensed under the CDDL (see LICENSE.txt in the top directory).
// -- END LICENSE BLOCK ------------------------------------------------

#include <gpu_voxels/vulkan/GpuVoxels.h>

#include <algorithm>
#include <cmath>
#include <limits>
#include <mutex>
#include <string>
#include <type_traits>

#include <gpu_voxels/logging/logging_vulkan.h>

namespace gpu_voxels::vulkan
{
	namespace
	{
		[[noreturn]] void unsupported(MapType type, const char* operation)
		{
			throw VulkanError(std::string("the operation '") + operation + "' is not supported by the map type " + map_type_name(type));
		}

		template<class T, class P>
		constexpr bool is_map_v = std::is_same_v<T, typename P::element_type>;

		/// For every cloud the index of the link it belongs to (the link whose name is the cloud name); checks the arguments of add_robot.
		std::vector<std::size_t> links_of_clouds(std::span<const std::string> link_names, std::size_t num_dh_parameters,
			std::span<const std::string> cloud_names)
		{
			if (link_names.size() != num_dh_parameters)
				throw VulkanError("add_robot: " + std::to_string(link_names.size()) + " link names but " + std::to_string(num_dh_parameters) + " sets of DH parameters");
			for (std::size_t i = 0; i < link_names.size(); ++i)
				for (std::size_t j = i + 1; j < link_names.size(); ++j)
					if (link_names[i] == link_names[j])
						throw VulkanError("add_robot: duplicate link name '" + link_names[i] + "'");

			std::vector<std::size_t> links;
			for (const std::string& cloud : cloud_names)
			{
				const auto it = std::find(link_names.begin(), link_names.end(), cloud);
				if (it == link_names.end())
					throw VulkanError("add_robot: '" + cloud + "' is no link name, so no joint would move this cloud");
				const auto link = static_cast<std::size_t>(it - link_names.begin());
				if (!links.empty() && link <= links.back())
					throw VulkanError("add_robot: the cloud of link '" + cloud + "' is not listed in link order");
				links.push_back(link);
			}
			return links;
		}

		std::vector<LinkSpec> link_specs(std::span<const std::string> link_names, std::span<const DHParameters> dh_parameters,
			std::span<const std::size_t> link_of_cloud, std::vector<std::vector<Point3f>>& clouds)
		{
			std::vector<LinkSpec> links;
			for (std::size_t i = 0; i < link_names.size(); ++i)
				links.push_back({ link_names[i], dh_parameters[i], std::nullopt });
			for (std::size_t k = 0; k < link_of_cloud.size(); ++k)
				links[link_of_cloud[k]].cloud = std::move(clouds[k]);
			return links;
		}
	}

	const char* map_type_name(MapType type)
	{
		switch (type)
		{
		case MT_BITVECTOR_VOXELMAP: return "MT_BITVECTOR_VOXELMAP";
		case MT_BITVECTOR_VOXELLIST: return "MT_BITVECTOR_VOXELLIST";
		case MT_BITVECTOR_OCTREE: return "MT_BITVECTOR_OCTREE";
		case MT_BITVECTOR_MORTON_VOXELLIST: return "MT_BITVECTOR_MORTON_VOXELLIST";
		case MT_PROBAB_VOXELMAP: return "MT_PROBAB_VOXELMAP";
		case MT_PROBAB_VOXELLIST: return "MT_PROBAB_VOXELLIST";
		case MT_PROBAB_OCTREE: return "MT_PROBAB_OCTREE";
		case MT_PROBAB_MORTON_VOXELLIST: return "MT_PROBAB_MORTON_VOXELLIST";
		case MT_COUNTING_VOXELLIST: return "MT_COUNTING_VOXELLIST";
		case MT_DISTANCE_VOXELMAP: return "MT_DISTANCE_VOXELMAP";
		}
		return "unknown map type";
	}

	// ---------------------------------------------------------------- GpuVoxelsMap

	GpuVoxelsMap::GpuVoxelsMap(Context& context, MapType type, Dim3ui dimensions, float voxel_side_length) : type_(type)
	{
		switch (type)
		{
		case MT_BITVECTOR_VOXELMAP:
			storage_.emplace<std::unique_ptr<BitVoxelMap>>(std::make_unique<BitVoxelMap>(context, dimensions, voxel_side_length));
			break;
		case MT_PROBAB_VOXELMAP:
			storage_.emplace<std::unique_ptr<ProbVoxelMap>>(std::make_unique<ProbVoxelMap>(context, dimensions, voxel_side_length));
			break;
		case MT_DISTANCE_VOXELMAP:
			storage_.emplace<std::unique_ptr<DistanceVoxelMap>>(std::make_unique<DistanceVoxelMap>(context, dimensions, voxel_side_length));
			break;
		case MT_BITVECTOR_VOXELLIST:
			storage_.emplace<std::unique_ptr<BitVoxelList>>(std::make_unique<BitVoxelList>(context, dimensions, voxel_side_length));
			break;
		case MT_COUNTING_VOXELLIST:
			storage_.emplace<std::unique_ptr<CountingVoxelList>>(std::make_unique<CountingVoxelList>(context, dimensions, voxel_side_length));
			break;
		case MT_BITVECTOR_OCTREE:
			storage_.emplace<std::unique_ptr<Octree>>(std::make_unique<Octree>(context, OctreeKind::Deterministic, voxel_side_length));
			break;
		case MT_PROBAB_OCTREE:
			storage_.emplace<std::unique_ptr<Octree>>(std::make_unique<Octree>(context, OctreeKind::Probabilistic, voxel_side_length));
			break;
		case MT_PROBAB_VOXELLIST:
		case MT_BITVECTOR_MORTON_VOXELLIST:
		case MT_PROBAB_MORTON_VOXELLIST:
			throw VulkanError(std::string("the map type ") + map_type_name(type) + " is not implemented");
		default:
			throw VulkanError("unknown map type " + std::to_string(static_cast<unsigned>(type)));
		}
	}

	GpuVoxelsMap::~GpuVoxelsMap() = default;

	uint32_t GpuVoxelsMap::insert_point_cloud(std::span<const Point3f> points, uint8_t meaning)
	{
		return std::visit([&](auto& map) -> uint32_t { return map->insert_point_cloud(points, meaning); }, storage_);
	}

	uint32_t GpuVoxelsMap::insert_coordinate_list(std::span<const Coord3ui> coordinates, uint8_t meaning)
	{
		return std::visit([&](auto& map) -> uint32_t { return map->insert_coordinate_list(coordinates, meaning); }, storage_);
	}

	uint32_t GpuVoxelsMap::insert_meta_point_cloud(const MetaPointCloud& cloud, uint8_t meaning)
	{
		return std::visit([&](auto& map) -> uint32_t { return map->insert_meta_point_cloud(cloud, meaning); }, storage_);
	}

	uint32_t GpuVoxelsMap::insert_meta_point_cloud(const MetaPointCloud& cloud, std::span<const uint8_t> meanings)
	{
		return std::visit([&](auto& map) -> uint32_t
		{
			if constexpr (is_map_v<DistanceVoxelMap, std::remove_cvref_t<decltype(map)>>)
			{
				// the distance map knows one meaning only (occupied): per sub cloud meanings are checked, then it is one insert
				if (meanings.size() != cloud.num_clouds())
					throw VulkanError("one meaning per sub cloud is required: " + std::to_string(meanings.size()) + " meanings for "
						+ std::to_string(cloud.num_clouds()) + " sub clouds");
				for (const uint8_t meaning : meanings)
					if (meaning != eBVM_OCCUPIED)
						throw VulkanError(std::string("the map type ") + map_type_name(MT_DISTANCE_VOXELMAP) + " only supports the meaning eBVM_OCCUPIED");
				return map->insert_meta_point_cloud(cloud, eBVM_OCCUPIED);
			}
			else
				return map->insert_meta_point_cloud(cloud, meanings);
		}, storage_);
	}

	void GpuVoxelsMap::clear_map()
	{
		std::visit([&](auto& map)
		{
			if constexpr (std::is_same_v<std::remove_cvref_t<decltype(*map)>, BitVoxelList>
				|| std::is_same_v<std::remove_cvref_t<decltype(*map)>, CountingVoxelList>
				|| std::is_same_v<std::remove_cvref_t<decltype(*map)>, Octree>)
				map->clear();
			else
				map->clear_map();
		}, storage_);
	}

	void GpuVoxelsMap::clear_bit_voxel_meaning(uint8_t meaning)
	{
		std::visit([&](auto& map)
		{
			using T = std::remove_cvref_t<decltype(*map)>;
			if constexpr (std::is_same_v<T, BitVoxelMap> || std::is_same_v<T, ProbVoxelMap>)
				map->clear_bit_voxel_meaning(meaning);
			else if constexpr (std::is_same_v<T, Octree>)
			{
				if (meaning != eBVM_OCCUPIED)
					throw VulkanError(std::string("the map type ") + map_type_name(type_) + " only supports the meaning eBVM_OCCUPIED");
				map->clear();
			}
			else
				unsupported(type_, "clear_bit_voxel_meaning");
		}, storage_);
	}

	bool GpuVoxelsMap::needs_rebuild() const
	{
		const Octree* tree = as<Octree>();
		return tree != nullptr && tree->needs_rebuild();
	}

	bool GpuVoxelsMap::rebuild()
	{
		Octree* tree = as<Octree>();
		if (tree == nullptr)
			return false;
		tree->rebuild();
		return true;
	}

	bool GpuVoxelsMap::rebuild_if_needed()
	{
		if (!needs_rebuild())
			return false;
		rebuild();
		return true;
	}

	uint64_t GpuVoxelsMap::memory_usage_bytes() const
	{
		return std::visit([&](const auto& map) -> uint64_t
		{
			if constexpr (std::is_same_v<std::remove_cvref_t<decltype(*map)>, Octree>)
				return map->memory_usage().bytes;
			else
				return map->memory_usage_bytes();
		}, storage_);
	}

	Dim3ui GpuVoxelsMap::dimensions() const
	{
		return std::visit([&](const auto& map) -> Dim3ui
		{
			using T = std::remove_cvref_t<decltype(*map)>;
			if constexpr (std::is_same_v<T, Octree>)
				return { Octree::kCoordinateLimit, Octree::kCoordinateLimit, Octree::kCoordinateLimit };
			else if constexpr (std::is_same_v<T, BitVoxelList> || std::is_same_v<T, CountingVoxelList>)
				return map->reference_dimensions();
			else
				return map->dimensions();
		}, storage_);
	}

	Point3f GpuVoxelsMap::metric_dimensions() const
	{
		const Dim3ui dim = dimensions();
		const float side = voxel_side_length();
		return { static_cast<float>(dim.x) * side, static_cast<float>(dim.y) * side, static_cast<float>(dim.z) * side };
	}

	float GpuVoxelsMap::voxel_side_length() const
	{
		return std::visit([&](const auto& map) -> float { return map->voxel_side_length(); }, storage_);
	}

	// ---------------------------------------------------------------- GpuVoxels

	namespace
	{
		std::mutex g_instance_mutex;
		std::weak_ptr<GpuVoxels> g_instance;
	}

	GpuVoxels::GpuVoxels() = default;

	GpuVoxels::GpuVoxels(Context& context) : context_(&context) {}

	GpuVoxels::~GpuVoxels() = default;

	GpuVoxelsSharedPtr GpuVoxels::get_instance()
	{
		const std::lock_guard lock(g_instance_mutex);
		GpuVoxelsSharedPtr instance = g_instance.lock();
		if (!instance)
		{
			instance.reset(new GpuVoxels());
			g_instance = instance;
		}
		return instance;
	}

	void GpuVoxels::initialize(Dim3ui dimensions, float voxel_side_length)
	{
		if (dimensions.x == 0 || dimensions.y == 0 || dimensions.z == 0)
			throw VulkanError("the dimensions of the maps must not be empty");
		if (!(voxel_side_length > 0.0f) || !std::isfinite(voxel_side_length))
			throw VulkanError("the voxel side length must be positive and finite");
		if (!maps_.empty())
			throw VulkanError("initialize() cannot change the size while maps exist: delete them first");

		if (context_ == nullptr)
		{
			owned_context_ = std::make_unique<Context>();
			context_ = owned_context_.get();
		}
		dimensions_ = dimensions;
		voxel_side_length_ = voxel_side_length;
		initialized_ = true;
	}

	Context& GpuVoxels::context() const
	{
		if (context_ == nullptr)
			throw VulkanError("the Vulkan context does not exist before initialize()");
		return *context_;
	}

	GpuVoxelsMapSharedPtr GpuVoxels::add_map(MapType type, const std::string& name)
	{
		if (!initialized_)
			throw VulkanError("initialize() has to be called before maps are added");
		if (maps_.contains(name))
		{
			LOGGING_ERROR_C(VulkanLog, GpuVoxels, "Map with name '" << name << "' already exists." << endl);
			return {};
		}
		auto map = std::make_shared<GpuVoxelsMap>(*context_, type, dimensions_, voxel_side_length_);
		maps_.emplace(name, map);
		return map;
	}

	bool GpuVoxels::del_map(const std::string& name)
	{
		if (maps_.erase(name) == 0)
		{
			LOGGING_ERROR_C(VulkanLog, GpuVoxels, "Map with name '" << name << "' not found." << endl);
			return false;
		}
		return true;
	}

	GpuVoxelsMapSharedPtr GpuVoxels::get_map(const std::string& name) const
	{
		const auto it = maps_.find(name);
		if (it == maps_.end())
		{
			LOGGING_ERROR_C(VulkanLog, GpuVoxels, "Map with name '" << name << "' not found." << endl);
			return {};
		}
		return it->second;
	}

	bool GpuVoxels::clear_map(const std::string& name)
	{
		const GpuVoxelsMapSharedPtr map = get_map(name);
		if (!map)
			return false;
		map->clear_map();
		return true;
	}

	bool GpuVoxels::clear_map(const std::string& name, uint8_t voxel_meaning)
	{
		const GpuVoxelsMapSharedPtr map = get_map(name);
		if (!map)
			return false;
		map->clear_bit_voxel_meaning(voxel_meaning);
		return true;
	}

	bool GpuVoxels::insert_point_cloud_into_map(std::span<const Point3f> points, const std::string& map_name, uint8_t voxel_meaning,
		uint32_t* outside)
	{
		const GpuVoxelsMapSharedPtr map = get_map(map_name);
		if (!map)
			return false;
		const uint32_t dropped = map->insert_point_cloud(points, voxel_meaning);
		if (outside != nullptr)
			*outside = dropped;
		return true;
	}

	bool GpuVoxels::insert_meta_point_cloud_into_map(const MetaPointCloud& cloud, const std::string& map_name,
		std::span<const uint8_t> voxel_meanings, uint32_t* outside)
	{
		const GpuVoxelsMapSharedPtr map = get_map(map_name);
		if (!map)
			return false;
		const uint32_t dropped = map->insert_meta_point_cloud(cloud, voxel_meanings);
		if (outside != nullptr)
			*outside = dropped;
		return true;
	}

	bool GpuVoxels::insert_meta_point_cloud_into_map(const MetaPointCloud& cloud, const std::string& map_name, uint8_t voxel_meaning,
		uint32_t* outside)
	{
		const GpuVoxelsMapSharedPtr map = get_map(map_name);
		if (!map)
			return false;
		const uint32_t dropped = map->insert_meta_point_cloud(cloud, voxel_meaning);
		if (outside != nullptr)
			*outside = dropped;
		return true;
	}

	bool GpuVoxels::insert_point_cloud_from_file(const std::string& map_name, const std::filesystem::path& file, const std::filesystem::path& model_path,
		uint8_t voxel_meaning, bool shift_to_zero, Point3f offset, float scaling, uint32_t* outside)
	{
		const GpuVoxelsMapSharedPtr map = get_map(map_name);
		if (!map)
			return false;
		std::vector<Point3f> points;
		try
		{
			points = load_point_cloud(file, model_path, PointCloudLoadOptions{ shift_to_zero, offset, scaling });
		}
		catch (const PointCloudFileError& e)
		{
			LOGGING_ERROR_C(VulkanLog, GpuVoxels, e.what() << endl);
			return false;
		}
		const uint32_t dropped = map->insert_point_cloud(points, voxel_meaning);
		if (outside != nullptr)
			*outside = dropped;
		return true;
	}

	bool GpuVoxels::insert_box_into_map(Point3f corner_min, Point3f corner_max, const std::string& map_name, uint8_t voxel_meaning,
		uint16_t points_per_voxel, uint32_t* outside)
	{
		const GpuVoxelsMapSharedPtr map = get_map(map_name);
		if (!map)
			return false;
		if (points_per_voxel == 0)
			throw VulkanError("points_per_voxel must be at least 1");
		const BoxVoxels box = box_voxel_coordinates(corner_min, corner_max, voxel_side_length_ / static_cast<float>(points_per_voxel),
			voxel_side_length_, map->dimensions());
		const uint32_t dropped = map->insert_coordinate_list(box.coordinates, voxel_meaning);
		if (outside != nullptr)
			*outside = static_cast<uint32_t>(std::min<uint64_t>(box.total - box.coordinates.size() + dropped, std::numeric_limits<uint32_t>::max()));
		return true;
	}

	bool GpuVoxels::add_robot(const std::string& robot_name, DHConvention convention, std::span<const std::string> link_names,
		std::span<const DHParameters> dh_parameters, std::span<const std::string> files, const std::filesystem::path& model_path)
	{
		if (robots_.contains(robot_name))
		{
			LOGGING_ERROR_C(VulkanLog, GpuVoxels, "Robot with name '" << robot_name << "' already exists." << endl);
			return false;
		}
		Context& ctx = context();
		const std::vector<std::size_t> link_of_cloud = links_of_clouds(link_names, dh_parameters.size(), files);
		std::vector<std::vector<Point3f>> clouds;
		try
		{
			for (const std::string& file : files)
				clouds.push_back(load_point_cloud(file, model_path, PointCloudLoadOptions{}));
		}
		catch (const PointCloudFileError& e)
		{
			LOGGING_ERROR_C(VulkanLog, GpuVoxels, e.what() << endl);
			return false;
		}
		const std::vector<LinkSpec> links = link_specs(link_names, dh_parameters, link_of_cloud, clouds);
		robots_.emplace(robot_name, std::make_shared<KinematicChain>(ctx, convention, links));
		return true;
	}

	bool GpuVoxels::add_robot(const std::string& robot_name, DHConvention convention, std::span<const std::string> link_names,
		std::span<const DHParameters> dh_parameters, const MetaPointCloud& clouds, std::span<const std::string> cloud_names)
	{
		if (robots_.contains(robot_name))
		{
			LOGGING_ERROR_C(VulkanLog, GpuVoxels, "Robot with name '" << robot_name << "' already exists." << endl);
			return false;
		}
		Context& ctx = context();
		if (cloud_names.size() != clouds.num_clouds())
			throw VulkanError("add_robot: " + std::to_string(cloud_names.size()) + " cloud names for " + std::to_string(clouds.num_clouds()) + " sub clouds");
		const std::vector<std::size_t> link_of_cloud = links_of_clouds(link_names, dh_parameters.size(), cloud_names);
		std::vector<std::vector<Point3f>> points;
		for (uint32_t k = 0; k < clouds.num_clouds(); ++k)
			points.push_back(clouds.download_cloud(k));
		const std::vector<LinkSpec> links = link_specs(link_names, dh_parameters, link_of_cloud, points);
		robots_.emplace(robot_name, std::make_shared<KinematicChain>(ctx, convention, links));
		return true;
	}

	KinematicChainSharedPtr GpuVoxels::get_robot(const std::string& robot_name) const
	{
		const auto it = robots_.find(robot_name);
		if (it == robots_.end())
		{
			LOGGING_ERROR_C(VulkanLog, GpuVoxels, "Robot with name '" << robot_name << "' not found." << endl);
			return {};
		}
		return it->second;
	}

	bool GpuVoxels::set_robot_configuration(const std::string& robot_name, const std::map<std::string, float>& joint_values)
	{
		const KinematicChainSharedPtr robot = get_robot(robot_name);
		if (!robot)
			return false;
		robot->set_configuration(joint_values);
		return true;
	}

	bool GpuVoxels::get_robot_configuration(const std::string& robot_name, std::map<std::string, float>& joint_values) const
	{
		const KinematicChainSharedPtr robot = get_robot(robot_name);
		if (!robot)
			return false;
		for (const auto& [name, value] : robot->configuration())
			joint_values[name] = value;
		return true;
	}

	bool GpuVoxels::set_robot_base_transformation(const std::string& robot_name, const Matrix4f& transformation)
	{
		const KinematicChainSharedPtr robot = get_robot(robot_name);
		if (!robot)
			return false;
		robot->set_base_transformation(transformation);
		return true;
	}

	bool GpuVoxels::get_robot_transformation(const std::string& robot_name, std::size_t idx, Matrix4f& transformation) const
	{
		const KinematicChainSharedPtr robot = get_robot(robot_name);
		if (!robot)
			return false;
		transformation = robot->transform(idx);
		return true;
	}

	bool GpuVoxels::update_robot_part(const std::string& robot_name, const std::string& link_name, std::span<const Point3f> points)
	{
		const KinematicChainSharedPtr robot = get_robot(robot_name);
		if (!robot)
			return false;
		robot->update_link_cloud(link_name, points);
		return true;
	}

	bool GpuVoxels::insert_robot_into_map(const std::string& robot_name, const std::string& map_name, uint8_t voxel_meaning, uint32_t* outside)
	{
		const KinematicChainSharedPtr robot = get_robot(robot_name);
		if (!robot)
			return false;
		const GpuVoxelsMapSharedPtr map = get_map(map_name);
		if (!map)
			return false;
		const uint32_t dropped = map->insert_meta_point_cloud(robot->transformed_clouds(), voxel_meaning);
		if (outside != nullptr)
			*outside = dropped;
		return true;
	}

	std::optional<BitVoxelMap::SelfCollisionResult> GpuVoxels::insert_robot_into_map_self_coll_aware(const std::string& robot_name,
		const std::string& map_name, std::span<const uint8_t> voxel_meanings, std::span<const BitVector256> collision_masks)
	{
		const KinematicChainSharedPtr robot = get_robot(robot_name);
		if (!robot)
			return std::nullopt;
		const GpuVoxelsMapSharedPtr map = get_map(map_name);
		if (!map)
			return std::nullopt;
		BitVoxelMap* bit_map = map->as<BitVoxelMap>();
		if (bit_map == nullptr)
			unsupported(map->type(), "insert_robot_into_map_self_coll_aware");
		return bit_map->insert_meta_point_cloud_with_self_collision_check(robot->transformed_clouds(), voxel_meanings, collision_masks);
	}

	bool GpuVoxels::visualize_map(const std::string& map_name, bool) const
	{
		if (!maps_.contains(map_name))
			LOGGING_ERROR_C(VulkanLog, GpuVoxels, "Map with name '" << map_name << "' not found." << endl);
		return false;
	}
}
