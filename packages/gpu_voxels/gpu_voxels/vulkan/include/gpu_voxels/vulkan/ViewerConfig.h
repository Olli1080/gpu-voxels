// -- BEGIN LICENSE BLOCK ----------------------------------------------
// This file is part of the GPU Voxels Software Library.
// Licensed under the CDDL (see LICENSE.txt in the top directory).
// -- END LICENSE BLOCK ------------------------------------------------

/*!\file
 * \brief  Style file of the viewer (XML): which layers a ViewerScene shows and in which colours. Parsing is strict: an unknown element or attribute, a malformed
 *         value or a duplicate layer is an error with the line number, so typos do not silently show nothing.
 *
 * \code
 *   <viewer>
 *     <window background="#14171c" shaded="true"/>
 *     <layer name="environment" type="map">                       <!-- a bit map or bit list of the GpuVoxels -->
 *       <palette default="hidden">                                <!-- default: "hues" (every meaning but 0) or "hidden" (only the listed ones) -->
 *         <meaning id="1" colour="#2aa84a"/>                      <!-- #rrggbb or #rrggbbaa; "none" hides -->
 *         <meaning id="5" hidden="true"/>
 *       </palette>
 *     </layer>
 *     <layer name="occupancy" type="map"><prob min_value="10"/></layer>
 *     <layer name="octree" type="map">
 *       <octree min_level="1" side_length="0.05" occupied="#eb8c28" free="#2040c880" collision="#ff0000" min_value="1"/>
 *     </layer>
 *     <layer name="spheres" type="primitives" colour="#5a96ff"/>   <!-- a primitive array -->
 *     <layer name="arm" type="robot" snap_to_voxels="false" point_size="0.07">
 *       <colour>#f0c828</colour><colour>#e63cc8</colour>          <!-- one per link with a cloud, used round robin -->
 *     </layer>
 *   </viewer>
 * \endcode
 * Layers are shown in the order of the file. Which elements a layer takes: map: palette, prob, octree (each at most once, the one that fits the type of the map
 * is used); primitives: none; robot: colour.
 */
#ifndef GPU_VOXELS_VULKAN_VIEWER_CONFIG_H_INCLUDED
#define GPU_VOXELS_VULKAN_VIEWER_CONFIG_H_INCLUDED

#include <array>
#include <cstdint>
#include <filesystem>
#include <optional>
#include <string>
#include <vector>

#include <gpu_voxels/vulkan/ViewerScene.h>

namespace gpu_voxels::vulkan
{
	struct ViewerLayerConfig
	{
		std::string name;
		ViewerScene::LayerKind kind = ViewerScene::LayerKind::Map;
		MapStyle map;                                  ///< kind Map
		uint32_t colour = pack_colour(90, 150, 255);   ///< kind Primitives
		RobotDisplay robot;                            ///< kind Robot
	};

	struct ViewerConfig
	{
		std::vector<ViewerLayerConfig> layers;
		std::optional<std::array<float, 4>> background; ///< RGBA, 0 .. 1
		std::optional<bool> shaded;

		/// Applies the window settings of the file (background colour, shading) to the options a window renders with.
		void apply(RenderOptions& options) const;
	};

	/// \throws VulkanError for malformed XML, an unknown element or attribute, a malformed value or a duplicate layer (the message names the line)
	[[nodiscard]] ViewerConfig parse_viewer_config(const std::string& xml);
	/// \throws VulkanError if the file cannot be read, and as parse_viewer_config
	[[nodiscard]] ViewerConfig load_viewer_config(const std::filesystem::path& file);

	/// Shows the layers of the file in the scene (ViewerScene::show_map / show_primitives / show_robot with the styles of the file). \return the layers that
	/// could not be shown, as "map:<name>", "primitives:<name>" or "robot:<name>" (an unknown name, or a map type that cannot be drawn); the others are shown.
	std::vector<std::string> show_layers(ViewerScene& scene, const ViewerConfig& config);
}

#endif
