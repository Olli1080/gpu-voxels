// -- BEGIN LICENSE BLOCK ----------------------------------------------
// This file is part of the GPU Voxels Software Library.
// Licensed under the CDDL (see LICENSE.txt in the top directory).
// -- END LICENSE BLOCK ------------------------------------------------

/*!\file
 * \brief  The viewer side of the façade: named layers (maps, bit lists, octrees, primitive arrays of one GpuVoxels) with a CubeBuffer each, shown or hidden
 *         one by one. It is what a window draws (ViewerWindow::draw(scene.buffers())) or an OffscreenTarget renders. Registers itself as the visualization
 *         hook of the GpuVoxels, so code written for the CUDA library (`gvl->visualizeMap("name")`) shows the map without further changes.
 *
 * \code
 *   GpuVoxels gvl(context);  gvl.initialize(...);  gvl.add_map(MT_BITVECTOR_VOXELMAP, "map");  ...
 *   ViewerScene scene(gvl);
 *   gvl.visualize_map("map");                  // or scene.show_map("map", style)
 *   while (!window.should_close()) { window.poll_events(); scene.refresh(); window.draw(scene.buffers()); }
 * \endcode
 * Layers are drawn in the order they were added. The scene holds the maps (shared pointers); it must not outlive the GpuVoxels and its Context. Not thread safe.
 */
#ifndef GPU_VOXELS_VULKAN_VIEWER_SCENE_H_INCLUDED
#define GPU_VOXELS_VULKAN_VIEWER_SCENE_H_INCLUDED

#include <array>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include <gpu_voxels/vulkan/GpuVoxels.h>
#include <gpu_voxels/vulkan/Viewer.h>

namespace gpu_voxels::vulkan
{
	/// How a map is drawn: the part that fits its type is used (bit map / bit list: palette; prob. map: prob; octree: octree).
	struct MapStyle
	{
		BitPalette palette = BitPalette::defaults();
		ProbDisplay prob;
		OctreeDisplay octree;
	};

	class ViewerScene
	{
	public:
		/// \param capacity_per_layer cubes one layer can hold (32 bytes each)
		explicit ViewerScene(GpuVoxels& voxels, uint32_t capacity_per_layer = 1u << 20);
		/// Removes the hooks of the GpuVoxels.
		~ViewerScene();
		ViewerScene(const ViewerScene&) = delete;
		ViewerScene& operator=(const ViewerScene&) = delete;

		/// Show the named map (replaces its layer) and extract it now. Displayable: bit and probabilistic voxel maps, bit voxel lists, both octrees.
		/// \return false (and an error log from the GpuVoxels) for an unknown name; false for a type that cannot be drawn (distance map, counting list)
		bool show_map(const std::string& map_name, const MapStyle& style = {});
		/// Show the named primitive array in one colour (pack_colour): spheres as spheres, cuboids as cubes. \return false for an unknown name.
		bool show_primitives(const std::string& array_name, uint32_t colour = pack_colour(90, 150, 255));
		/// Remove the layer of a map / primitive array. \return false if there was none.
		bool remove_map(const std::string& map_name);
		bool remove_primitives(const std::string& array_name);

		/// Extract every layer again (the maps changed). Hidden layers are skipped (they are extracted when they are shown again).
		void refresh();

		struct LayerInfo
		{
			std::string name;
			bool primitives = false;   ///< a primitive array (names of maps and arrays are separate)
			bool visible = true;
			uint32_t cubes = 0;        ///< cubes stored at the last extraction
			uint32_t dropped = 0;      ///< cubes that did not fit into the layer
		};
		[[nodiscard]] std::vector<LayerInfo> layers() const;
		[[nodiscard]] std::size_t layer_count() const { return layers_.size(); }
		/// \return false for an index that does not exist. Showing a layer again extracts it.
		bool set_visible(std::size_t index, bool visible);
		bool toggle_visible(std::size_t index);

		/// The buffers of the visible layers in drawing order (valid until a layer is added or removed).
		[[nodiscard]] std::vector<const CubeBuffer*> buffers() const;
		/// The volume of the GpuVoxels in metres: from the origin to dimensions * voxel side length (for the first camera position).
		[[nodiscard]] std::array<float, 3> world_size() const;

	private:
		struct Layer;
		void extract(Layer& layer);
		Layer* find(const std::string& name, bool primitives);
		bool hook_map(const std::string& name, bool force_repaint);
		bool hook_primitives(const std::string& name, bool force_repaint);

		GpuVoxels& voxels_;
		uint32_t capacity_;
		CubeExtractor extractor_;
		std::vector<std::unique_ptr<Layer>> layers_;
	};
}

#endif
