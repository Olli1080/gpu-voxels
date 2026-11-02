// -- BEGIN LICENSE BLOCK ----------------------------------------------
// This file is part of the GPU Voxels Software Library.
// Licensed under the CDDL (see LICENSE.txt in the top directory).
// -- END LICENSE BLOCK ------------------------------------------------

/*!\file
 * \brief  Façade of the Vulkan backend: named maps behind one runtime type (slice S7-1 of the migration plan).
 *
 * Counterpart of gpu_voxels::GpuVoxels / GpuVoxelsMap for maps, lists and octrees. Robots, point cloud files, boxes,
 * primitives, merge and disk I/O follow in S7-2 .. S7-5 (Agent/Analysis/2026-11-02_s7_facade_checklist.md). The
 * decisions D1 .. D6 of that note apply: a thin handle over the concrete classes, checked as<T>(), unsupported
 * operations throw, no locking, Vulkan value types in the signatures, namespace gpu_voxels::vulkan.
 */
#ifndef GPU_VOXELS_VULKAN_GPU_VOXELS_H_INCLUDED
#define GPU_VOXELS_VULKAN_GPU_VOXELS_H_INCLUDED

#include <cstdint>
#include <filesystem>
#include <map>
#include <memory>
#include <span>
#include <string>
#include <variant>

#include <gpu_voxels/vulkan/BitVoxelMap.h>
#include <gpu_voxels/vulkan/Context.h>
#include <gpu_voxels/vulkan/DistanceVoxelMap.h>
#include <gpu_voxels/vulkan/MetaPointCloud.h>
#include <gpu_voxels/vulkan/Octree.h>
#include <gpu_voxels/vulkan/PointCloudFile.h>
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
		/// eturn false (and an error log) for an unknown map or a file that cannot be read (unknown extension, missing, malformed);
		///         `outside` receives the number of points that were not inserted
		bool insert_point_cloud_from_file(const std::string& map_name, const std::filesystem::path& file, const std::filesystem::path& model_path,
			uint8_t voxel_meaning, bool shift_to_zero = false, Point3f offset = { 0.0f, 0.0f, 0.0f }, float scaling = 1.0f, uint32_t* outside = nullptr);
		/// CUDA insertBoxIntoMap: the voxels from floor(min / side) to ceil(max / side) inclusive (see box_voxel_coordinates) inserted as a
		/// coordinate list with the meaning. `points_per_voxel` only sets the step (CUDA: round(1 / points_per_voxel) voxels, so it makes no
		/// difference above 2). Voxels beyond the map are counted in `outside` without being generated.
		/// eturn false (and an error log) for an unknown map. 	hrows VulkanError for points_per_voxel 0 and the cases of box_voxel_coordinates
		bool insert_box_into_map(Point3f corner_min, Point3f corner_max, const std::string& map_name, uint8_t voxel_meaning,
			uint16_t points_per_voxel = 1, uint32_t* outside = nullptr);

		/// There is no viewer on Vulkan (decision Q6): always false, like CUDA without GPUVOXEL_VISUALIZE.
		bool visualize_map(const std::string& map_name, bool force_repaint = true) const;

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
	};
}

#endif
