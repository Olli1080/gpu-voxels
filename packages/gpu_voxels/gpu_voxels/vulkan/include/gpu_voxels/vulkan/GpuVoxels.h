// -- BEGIN LICENSE BLOCK ----------------------------------------------
// This file is part of the GPU Voxels Software Library.
// Licensed under the CDDL (see LICENSE.txt in the top directory).
// -- END LICENSE BLOCK ------------------------------------------------

/*!\file
 * \brief  Façade of the Vulkan backend: named maps behind one runtime type (slice S7-1 of the migration plan).
 *
 * Counterpart of gpu_voxels::GpuVoxels / GpuVoxelsMap for maps, lists and octrees, point cloud files, boxes (S7-2) and DH
 * robots (S7-3), merge and disk I/O of maps and lists (S7-4), primitive arrays (S7-5). URDF robots and the viewer follow later
 * (Agent/Analysis/2026-11-02_s7_facade_checklist.md). The
 * decisions D1 .. D6 of that note apply: a thin handle over the concrete classes, checked as<T>(), unsupported
 * operations throw, no locking, Vulkan value types in the signatures, namespace gpu_voxels::vulkan.
 */
#ifndef GPU_VOXELS_VULKAN_GPU_VOXELS_H_INCLUDED
#define GPU_VOXELS_VULKAN_GPU_VOXELS_H_INCLUDED

#include <cstdint>
#include <filesystem>
#include <functional>
#include <map>
#include <memory>
#include <optional>
#include <span>
#include <string>
#include <variant>
#include <vector>

#include <gpu_voxels/vulkan/BitVoxelMap.h>
#include <gpu_voxels/vulkan/Context.h>
#include <gpu_voxels/vulkan/DistanceVoxelMap.h>
#include <gpu_voxels/vulkan/KinematicChain.h>
#include <gpu_voxels/vulkan/MapFile.h>
#include <gpu_voxels/vulkan/MetaPointCloud.h>
#include <gpu_voxels/vulkan/Octree.h>
#include <gpu_voxels/vulkan/OctreeStorage.h>
#include <gpu_voxels/vulkan/PointCloudFile.h>
#include <gpu_voxels/vulkan/PrimitiveArray.h>
#include <gpu_voxels/vulkan/ProbVoxelMap.h>
#include <gpu_voxels/vulkan/VoxelList.h>
#include <gpu_voxels/vulkan/vulkan_types.h>

namespace gpu_voxels::vulkan
{
	/// Name of a map type for messages ("MT_BITVECTOR_VOXELMAP", ...).
	[[nodiscard]] const char* map_type_name(MapType type);

	/// One map, list or octree behind its runtime type (CUDA GpuVoxelsMap). Owns exactly one of the concrete classes; the
	/// members below dispatch on it, `as<T>()` gives the concrete class for everything else (collisions, queries, ...).
	/// Handles must not outlive the GpuVoxels (the Context) that made them.
	class GpuVoxelsMap
	{
	public:
		/// Creates the class of `type` with the dimensions and voxel side length of the façade. Octrees ignore the
		/// dimensions (they always cover 2^14 voxels per axis); distance maps allow at most 1023 voxels per axis.
		/// \throws VulkanError for MT_PROBAB_VOXELLIST and the two Morton list types (CUDA: "not implemented"), an unknown
		///         type, or whatever the concrete constructor throws
		GpuVoxelsMap(Context& context, MapType type, Dim3ui dimensions, float voxel_side_length);
		~GpuVoxelsMap();
		GpuVoxelsMap(const GpuVoxelsMap&) = delete;
		GpuVoxelsMap& operator=(const GpuVoxelsMap&) = delete;

		[[nodiscard]] MapType type() const { return type_; }

		/// The concrete class, or nullptr if the map is another one (CUDA: an unchecked static_cast). Both octree kinds are
		/// `Octree`; ask `type()` or `Octree::kind()` to tell them apart.
		template<class T>
		[[nodiscard]] T* as() noexcept
		{
			auto* held = std::get_if<std::unique_ptr<T>>(&storage_);
			return held ? held->get() : nullptr;
		}
		template<class T>
		[[nodiscard]] const T* as() const noexcept
		{
			const auto* held = std::get_if<std::unique_ptr<T>>(&storage_);
			return held ? held->get() : nullptr;
		}
		template<class T>
		[[nodiscard]] bool is() const noexcept { return as<T>() != nullptr; }

		/// The inserts return the number of points / coordinates that were not inserted (outside the map or the tree).
		/// Meanings follow the concrete class: counting lists ignore them, prob. maps turn them into updates, distance
		/// maps and octrees accept occupied (octrees also free) and throw for others.
		uint32_t insert_point_cloud(std::span<const Point3f> points, uint8_t meaning);
		uint32_t insert_coordinate_list(std::span<const Coord3ui> coordinates, uint8_t meaning);
		uint32_t insert_meta_point_cloud(const MetaPointCloud& cloud, uint8_t meaning);
		/// One meaning per sub cloud. \throws VulkanError for a wrong number of meanings; distance maps (occupied only) throw
		/// for any other meaning before anything is inserted
		uint32_t insert_meta_point_cloud(const MetaPointCloud& cloud, std::span<const uint8_t> meanings);

		/// Removes all data (maps: unknown / free, lists and octrees: empty).
		void clear_map();
		/// Bit maps clear the bit; prob. maps and octrees only support eBVM_OCCUPIED (clears the map, CUDA: logs an error for the
		/// rest, here: throws); distance maps and lists do not support it (CUDA: no-op / logs). \throws VulkanError
		void clear_bit_voxel_meaning(uint8_t meaning);

		/// Octrees only; every other type is never fragmented (CUDA logs "not supported" and returns false).
		[[nodiscard]] bool needs_rebuild() const;
		/// \return true if a rebuild was done (octrees; false for every other type)
		bool rebuild();
		/// CUDA rebuildIfNeeded: rebuilds when needed. \return true if it rebuilt
		bool rebuild_if_needed();

		/// CUDA merge: copies the data of `other` into this map, moved by `voxel_offset`. The combinations (the rest throws, CUDA logs and
		/// returns false): bit / prob. / distance map <- bit list or counting list (the coordinates are inserted as occupied; no offset and no
		/// `new_meaning`, CUDA has no offset there and ignores the meaning); bit list <- bit list, counting list (bit vectors OR-ed into existing
		/// entries; `new_meaning` replaces the bits of the new entries) or bit map (the occupied voxels, no offset); counting list <- counting list
		/// (counts added) or bit list (count 0 for new entries, as CUDA), no `new_meaning`. Octrees merge nothing (CUDA: stub).
		/// \return number of voxels / entries that were not merged because they lie outside this map or reference map (after the offset)
		/// \throws VulkanError for an unsupported combination or argument, a different voxel side length (lists: or reference dimensions),
		///         `&other == this`
		uint32_t merge(const GpuVoxelsMap& other, Offset3i voxel_offset = {}, std::optional<uint8_t> new_meaning = std::nullopt);
		/// CUDA merge with a metric offset: `voxel_offset = floor(metric_offset / side)` for list targets (CUDA mapToVoxelsSigned) and the truncation
		/// toward zero for maps (CUDA cast). \throws VulkanError for an offset that is not finite or beyond +-2^31 voxels, and as above
		uint32_t merge_metric(const GpuVoxelsMap& other, Point3f metric_offset, std::optional<uint8_t> new_meaning = std::nullopt);

		/// CUDA writeToDisk / readFromDisk: the native formats of MapFile.h (octrees: OctreeNativeStorage). A file that cannot be written, opened or
		/// used (wrong type, dimensions, side length, length or checksum) gives `false` and an error log (CUDA the same, but reads partly); a read
		/// replaces the whole content and leaves the map unchanged when it fails. Octrees restore the voxel side length of the file.
		bool write_to_disk(const std::filesystem::path& file);
		bool read_from_disk(const std::filesystem::path& file);

		[[nodiscard]] uint64_t memory_usage_bytes() const;
		/// Voxels per axis. Maps: their dimensions; lists: the dimensions of the reference map (CUDA returns
		/// `(number of entries, 1, 1)`, use `as<BitVoxelList>()->size()` for that); octrees: 2^14 per axis.
		[[nodiscard]] Dim3ui dimensions() const;
		/// `dimensions() * voxel_side_length()` per axis in metres.
		[[nodiscard]] Point3f metric_dimensions() const;
		[[nodiscard]] float voxel_side_length() const;

	private:
		using Storage = std::variant<std::unique_ptr<BitVoxelMap>, std::unique_ptr<ProbVoxelMap>, std::unique_ptr<DistanceVoxelMap>,
			std::unique_ptr<BitVoxelList>, std::unique_ptr<CountingVoxelList>, std::unique_ptr<Octree>>;

		MapType type_;
		Storage storage_;
	};

	using GpuVoxelsMapSharedPtr = std::shared_ptr<GpuVoxelsMap>;
	/// The Vulkan robot is the DH chain itself (there is no URDF robot yet, so no common interface class). Like the maps,
	/// a handle must not outlive the GpuVoxels (the Context) that made it.
	using KinematicChainSharedPtr = std::shared_ptr<KinematicChain>;

	class GpuVoxels;
	using GpuVoxelsSharedPtr = std::shared_ptr<GpuVoxels>;

	/// The high level API (CUDA gpu_voxels::GpuVoxels): named maps of one common size and voxel side length. Not thread safe
	/// (one thread per instance, D3).
	class GpuVoxels
	{
	public:
		/// Uses an existing Context (not owned, must outlive this object). Call initialize() before adding maps.
		explicit GpuVoxels(Context& context);
		~GpuVoxels();
		GpuVoxels(const GpuVoxels&) = delete;
		GpuVoxels& operator=(const GpuVoxels&) = delete;

		/// Singleton like CUDA: the same object while someone holds it. It owns its Context, which is created in initialize()
		/// with the default ContextCreateInfo.
		[[nodiscard]] static GpuVoxelsSharedPtr get_instance();

		/// Sets the size and voxel side length of all maps added later; an owning instance creates its Context here.
		/// \throws VulkanError for an empty dimension, a side length that is not positive and finite, maps that already exist
		///         (their size would no longer match), or a failing Context creation (then the instance stays uninitialised)
		void initialize(Dim3ui dimensions, float voxel_side_length);
		[[nodiscard]] bool is_initialized() const { return context_ != nullptr && initialized_; }
		/// \throws VulkanError before the context exists
		[[nodiscard]] Context& context() const;

		/// \return the new map; empty (and an error log) if the name is already used
		/// \throws VulkanError before initialize(), for the types that are not built and for failing constructors (no map is added then)
		GpuVoxelsMapSharedPtr add_map(MapType type, const std::string& name);
		/// \return false (and an error log) for an unknown name
		bool del_map(const std::string& name);
		/// \return empty (and an error log) for an unknown name
		[[nodiscard]] GpuVoxelsMapSharedPtr get_map(const std::string& name) const;
		/// \return false for an unknown name. \throws VulkanError as GpuVoxelsMap::clear_map / clear_bit_voxel_meaning
		bool clear_map(const std::string& name);
		bool clear_map(const std::string& name, uint8_t voxel_meaning);

		/// The inserts return false for an unknown map. `outside` (optional) receives the number of points that were not inserted.
		bool insert_point_cloud_into_map(std::span<const Point3f> points, const std::string& map_name, uint8_t voxel_meaning,
			uint32_t* outside = nullptr);
		bool insert_meta_point_cloud_into_map(const MetaPointCloud& cloud, const std::string& map_name,
			std::span<const uint8_t> voxel_meanings, uint32_t* outside = nullptr);
		bool insert_meta_point_cloud_into_map(const MetaPointCloud& cloud, const std::string& map_name, uint8_t voxel_meaning,
			uint32_t* outside = nullptr);

		/// CUDA insertPointCloudFromFile: loads `model_path / file` (.xyz, .pcd, .binvox, see PointCloudFile.h), shifts it so its minimum is 0
		/// if asked, applies `p = scaling * p + offset` and inserts the points with the meaning.
		/// \return false (and an error log) for an unknown map or a file that cannot be read (unknown extension, missing, malformed);
		///         `outside` receives the number of points that were not inserted
		bool insert_point_cloud_from_file(const std::string& map_name, const std::filesystem::path& file, const std::filesystem::path& model_path,
			uint8_t voxel_meaning, bool shift_to_zero = false, Point3f offset = { 0.0f, 0.0f, 0.0f }, float scaling = 1.0f, uint32_t* outside = nullptr);
		/// CUDA insertBoxIntoMap: the voxels from floor(min / side) to ceil(max / side) inclusive (see box_voxel_coordinates) inserted as a
		/// coordinate list with the meaning. `points_per_voxel` only sets the step (CUDA: round(1 / points_per_voxel) voxels, so it makes no
		/// difference above 2). Voxels beyond the map are counted in `outside` without being generated.
		/// \return false (and an error log) for an unknown map. \throws VulkanError for points_per_voxel 0 and the cases of box_voxel_coordinates
		bool insert_box_into_map(Point3f corner_min, Point3f corner_max, const std::string& map_name, uint8_t voxel_meaning,
			uint16_t points_per_voxel = 1, uint32_t* outside = nullptr);

		/// CUDA addRobot (DH, files): link i is `link_names[i]` with `dh_parameters[i]`. `files` are the model files of the links
		/// that have geometry; as in CUDA a file belongs to the link whose name is the file string, so every entry of `files` must be
		/// a link name. Differences: the files must be listed in link order (sub cloud k of the robot = k-th link with a file; CUDA:
		/// k-th file), a file that is no link name throws (CUDA keeps it as a cloud that no joint moves). Files are read with
		/// load_point_cloud (no shift, scaling 1) from `model_path`; links without a file only add their DH transformation.
		/// \return false (and an error log) if the name is used or a file cannot be read (CUDA: a file error leaves a robot without
		///         clouds and still returns true); nothing is added then
		/// \throws VulkanError for different numbers of links and parameters, duplicate link names, an unknown or misplaced file,
		///         before the context exists
		bool add_robot(const std::string& robot_name, DHConvention convention, std::span<const std::string> link_names,
			std::span<const DHParameters> dh_parameters, std::span<const std::string> files, const std::filesystem::path& model_path);
		/// CUDA addRobot (DH, MetaPointCloud): the clouds are copied. `cloud_names[k]` is the link that sub cloud k belongs to (CUDA keeps
		/// the names in the MetaPointCloud); same rules as above (link names, link order). \return false for a used name.
		/// \throws VulkanError as above, and for a number of names that differs from the number of sub clouds
		bool add_robot(const std::string& robot_name, DHConvention convention, std::span<const std::string> link_names,
			std::span<const DHParameters> dh_parameters, const MetaPointCloud& clouds, std::span<const std::string> cloud_names);
		/// \return the robot; empty (and an error log) for an unknown name
		[[nodiscard]] KinematicChainSharedPtr get_robot(const std::string& robot_name) const;

		/// Wrappers of the KinematicChain members, all `false` (and an error log) for an unknown robot.
		/// update_robot_part: the number of points must not change and the link must have a cloud (\throws VulkanError otherwise).
		bool set_robot_configuration(const std::string& robot_name, const std::map<std::string, float>& joint_values);
		/// Extends `joint_values` with every joint of the robot (CUDA getConfiguration).
		bool get_robot_configuration(const std::string& robot_name, std::map<std::string, float>& joint_values) const;
		bool set_robot_base_transformation(const std::string& robot_name, const Matrix4f& transformation);
		/// idx 0 is the base, num_links() the end effector. \throws VulkanError for a larger index (CUDA: identity and an error log)
		bool get_robot_transformation(const std::string& robot_name, std::size_t idx, Matrix4f& transformation) const;
		bool update_robot_part(const std::string& robot_name, const std::string& link_name, std::span<const Point3f> points);

		/// CUDA insertRobotIntoMap: all link clouds in the current pose with one meaning. `false` for an unknown robot or map;
		/// `outside` receives the number of points that were not inserted.
		bool insert_robot_into_map(const std::string& robot_name, const std::string& map_name, uint8_t voxel_meaning, uint32_t* outside = nullptr);
		/// CUDA insertRobotIntoMapSelfCollAware, bit maps only: BitVoxelMap::insert_meta_point_cloud_with_self_collision_check with
		/// the empty defaults of CUDA (meaning of sub cloud k = eBVM_SWEPT_VOLUME_START + k, every sub cloud collides with all but itself
		/// and eBVM_COLLISION). Returns the whole result (CUDA: only `colliding`, so an unknown name and "no collision" are both false);
		/// nullopt (and an error log) for an unknown robot or map. \throws VulkanError for every other map type (CUDA logs and returns
		/// true) and for the argument errors of the map class
		std::optional<BitVoxelMap::SelfCollisionResult> insert_robot_into_map_self_coll_aware(const std::string& robot_name, const std::string& map_name,
			std::span<const uint8_t> voxel_meanings = {}, std::span<const BitVector256> collision_masks = {});

		/// CUDA addPrimitives: a new, empty host container of `type` (see PrimitiveArray.h) with the dimensions and voxel side length of
		/// this instance; the names are separate from the map names. \return false (and an error log) if the name is used
		/// \throws VulkanError before initialize() and for a type that is neither sphere nor cuboid
		bool add_primitives(PrimitiveType type, const std::string& array_name);
		/// \return false (and an error log) for an unknown name. Handles from get_primitives() stay valid.
		bool del_primitives(const std::string& array_name);
		/// CUDA modifyPrimitives: PrimitiveArray::set_points of the named array, all four forms. \return false (and an error log) for an
		/// unknown name. \throws VulkanError as PrimitiveArray::set_points (the array is unchanged then)
		bool modify_primitives(const std::string& array_name, std::span<const Point4f> metric_positions);
		bool modify_primitives(const std::string& array_name, std::span<const Coord4i> voxel_positions);
		bool modify_primitives(const std::string& array_name, std::span<const Point3f> metric_positions, float diameter);
		bool modify_primitives(const std::string& array_name, std::span<const Coord3i> voxel_positions, uint32_t diameter);
		/// The array itself (CUDA has no getter; without a viewer it is the only way to look at the content). Empty (and an error log) for an unknown name.
		[[nodiscard]] PrimitiveArraySharedPtr get_primitives(const std::string& array_name) const;

		/// Hand over a map / primitive array to the viewer: false without a viewer (like CUDA without GPUVOXEL_VISUALIZE), and false (and an error log) for an
		/// unknown name. A viewer registers itself with set_visualization_hooks (the ViewerScene of ViewerScene.h does it); the hook then decides what
		/// `force_repaint` means and what the result is.
		bool visualize_map(const std::string& map_name, bool force_repaint = true) const;
		bool visualize_primitives_array(const std::string& array_name, bool force_repaint = true) const;
		using VisualizeHook = std::function<bool(const std::string& name, bool force_repaint)>;
		/// The functions behind visualize_map / visualize_primitives_array; an empty function removes the hook. Not owned: whoever sets them must remove them
		/// before it dies (ViewerScene does).
		void set_visualization_hooks(VisualizeHook map_hook, VisualizeHook primitives_hook);

		[[nodiscard]] Dim3ui dimensions() const { return dimensions_; }
		[[nodiscard]] float voxel_side_length() const { return voxel_side_length_; }

	private:
		GpuVoxels();

		std::unique_ptr<Context> owned_context_;
		Context* context_ = nullptr;
		bool initialized_ = false;
		Dim3ui dimensions_{ 0, 0, 0 };
		float voxel_side_length_ = 0.0f;
		std::map<std::string, GpuVoxelsMapSharedPtr> maps_; ///< destroyed before the owned context
		std::map<std::string, KinematicChainSharedPtr> robots_; ///< likewise
		std::map<std::string, PrimitiveArraySharedPtr> primitives_; ///< host only
		VisualizeHook visualize_map_hook_, visualize_primitives_hook_;
	};
}

#endif
