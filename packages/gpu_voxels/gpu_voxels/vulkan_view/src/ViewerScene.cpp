// -- BEGIN LICENSE BLOCK ----------------------------------------------
// This file is part of the GPU Voxels Software Library.
// Licensed under the CDDL (see LICENSE.txt in the top directory).
// -- END LICENSE BLOCK ------------------------------------------------

#include <gpu_voxels/vulkan_view/ViewerScene.h>

#include <algorithm>

namespace gpu_voxels::vulkan
{
	struct ViewerScene::Layer
	{
		std::string name;
		LayerKind kind = LayerKind::Map;
		bool visible = true;
		GpuVoxelsMapSharedPtr map;
		PrimitiveArraySharedPtr array;
		KinematicChainSharedPtr robot;
		RobotDisplay robot_style;
		MapStyle style;
		uint32_t colour = 0;
		std::unique_ptr<CubeBuffer> cubes;
	};

	namespace
	{
		bool displayable(const GpuVoxelsMap& map)
		{
			return map.is<BitVoxelMap>() || map.is<ProbVoxelMap>() || map.is<BitVoxelList>() || map.is<Octree>();
		}
	}

	ViewerScene::ViewerScene(GpuVoxels& voxels, uint32_t capacity_per_layer)
		: voxels_(voxels), capacity_(capacity_per_layer), extractor_(voxels.context())
	{
		if (capacity_per_layer == 0)
			throw VulkanError("ViewerScene: the capacity of a layer must be positive");
		voxels_.set_visualization_hooks([this](const std::string& name, bool force) { return hook_map(name, force); },
			[this](const std::string& name, bool force) { return hook_primitives(name, force); });
	}

	ViewerScene::~ViewerScene()
	{
		voxels_.set_visualization_hooks({}, {});
	}

	ViewerScene::Layer* ViewerScene::find(const std::string& name, LayerKind kind)
	{
		for (auto& layer : layers_)
			if (layer->kind == kind && layer->name == name)
				return layer.get();
		return nullptr;
	}

	void ViewerScene::extract(Layer& layer)
	{
		if (layer.kind == LayerKind::Primitives)
		{
			extractor_.extract(*layer.array, layer.colour, *layer.cubes);
			return;
		}
		if (layer.kind == LayerKind::Robot)
		{
			extractor_.extract(layer.robot->transformed_clouds(), voxels_.voxel_side_length(), layer.robot_style, *layer.cubes);
			return;
		}
		GpuVoxelsMap& map = *layer.map;
		if (auto* bit = map.as<BitVoxelMap>())
			extractor_.extract(*bit, layer.style.palette, *layer.cubes);
		else if (auto* prob = map.as<ProbVoxelMap>())
			extractor_.extract(*prob, layer.style.prob, *layer.cubes);
		else if (auto* list = map.as<BitVoxelList>())
			extractor_.extract(*list, layer.style.palette, *layer.cubes);
		else if (auto* tree = map.as<Octree>())
		{
			OctreeDisplay display = layer.style.octree;
			if (display.side_length == 0.0f && tree->voxel_side_length() == 0.0f)
				display.side_length = map.voxel_side_length();
			extractor_.extract(*tree, display, *layer.cubes);
		}
	}

	bool ViewerScene::show_map(const std::string& map_name, const MapStyle& style)
	{
		const GpuVoxelsMapSharedPtr map = voxels_.get_map(map_name);
		if (!map || !displayable(*map))
			return false;
		auto layer = std::make_unique<Layer>();
		layer->name = map_name;
		layer->map = map;
		layer->style = style;
		layer->cubes = std::make_unique<CubeBuffer>(voxels_.context(), capacity_);
		extract(*layer);
		if (Layer* existing = find(map_name, LayerKind::Map))
			*existing = std::move(*layer);
		else
			layers_.push_back(std::move(layer));
		return true;
	}

	bool ViewerScene::show_primitives(const std::string& array_name, uint32_t colour)
	{
		const PrimitiveArraySharedPtr array = voxels_.get_primitives(array_name);
		if (!array)
			return false;
		auto layer = std::make_unique<Layer>();
		layer->name = array_name;
		layer->kind = LayerKind::Primitives;
		layer->array = array;
		layer->colour = colour;
		layer->cubes = std::make_unique<CubeBuffer>(voxels_.context(), capacity_);
		extract(*layer);
		if (Layer* existing = find(array_name, LayerKind::Primitives))
			*existing = std::move(*layer);
		else
			layers_.push_back(std::move(layer));
		return true;
	}

	bool ViewerScene::show_robot(const std::string& robot_name, const RobotDisplay& style)
	{
		const KinematicChainSharedPtr robot = voxels_.get_robot(robot_name);
		if (!robot)
			return false;
		auto layer = std::make_unique<Layer>();
		layer->name = robot_name;
		layer->kind = LayerKind::Robot;
		layer->robot = robot;
		layer->robot_style = style;
		layer->cubes = std::make_unique<CubeBuffer>(voxels_.context(), capacity_);
		extract(*layer);
		if (Layer* existing = find(robot_name, LayerKind::Robot))
			*existing = std::move(*layer);
		else
			layers_.push_back(std::move(layer));
		return true;
	}

	bool ViewerScene::remove_map(const std::string& map_name)
	{
		const auto it = std::find_if(layers_.begin(), layers_.end(), [&](const auto& l) { return l->kind == LayerKind::Map && l->name == map_name; });
		if (it == layers_.end())
			return false;
		layers_.erase(it);
		return true;
	}

	bool ViewerScene::remove_primitives(const std::string& array_name)
	{
		const auto it = std::find_if(layers_.begin(), layers_.end(), [&](const auto& l) { return l->kind == LayerKind::Primitives && l->name == array_name; });
		if (it == layers_.end())
			return false;
		layers_.erase(it);
		return true;
	}

	bool ViewerScene::remove_robot(const std::string& robot_name)
	{
		const auto it = std::find_if(layers_.begin(), layers_.end(), [&](const auto& l) { return l->kind == LayerKind::Robot && l->name == robot_name; });
		if (it == layers_.end())
			return false;
		layers_.erase(it);
		return true;
	}

	void ViewerScene::refresh()
	{
		for (auto& layer : layers_)
			if (layer->visible)
				extract(*layer);
	}

	const CubeBuffer* ViewerScene::layer_buffer(std::size_t index) const
	{
		return index < layers_.size() ? layers_[index]->cubes.get() : nullptr;
	}

	bool ViewerScene::refresh_layer(std::size_t index)
	{
		if (index >= layers_.size())
			return false;
		extract(*layers_[index]);
		return true;
	}

	std::vector<ViewerScene::LayerInfo> ViewerScene::layers() const
	{
		std::vector<LayerInfo> result;
		for (const auto& layer : layers_)
			result.push_back({ layer->name, layer->kind, layer->visible, layer->cubes->count(), layer->cubes->dropped() });
		return result;
	}

	bool ViewerScene::set_visible(std::size_t index, bool visible)
	{
		if (index >= layers_.size())
			return false;
		Layer& layer = *layers_[index];
		if (visible && !layer.visible)
			extract(layer);   // it was not refreshed while hidden
		layer.visible = visible;
		return true;
	}

	bool ViewerScene::toggle_visible(std::size_t index)
	{
		return index < layers_.size() && set_visible(index, !layers_[index]->visible);
	}

	std::vector<const CubeBuffer*> ViewerScene::buffers() const
	{
		std::vector<const CubeBuffer*> result;
		for (const auto& layer : layers_)
			if (layer->visible)
				result.push_back(layer->cubes.get());
		return result;
	}

	std::array<float, 3> ViewerScene::world_size() const
	{
		const Dim3ui d = voxels_.dimensions();
		const float side = voxels_.voxel_side_length();
		return { static_cast<float>(d.x) * side, static_cast<float>(d.y) * side, static_cast<float>(d.z) * side };
	}

	bool ViewerScene::hook_map(const std::string& name, bool force_repaint)
	{
		if (Layer* layer = find(name, LayerKind::Map))
		{
			if (force_repaint && layer->visible)
				extract(*layer);
			return true;
		}
		return show_map(name);
	}

	bool ViewerScene::hook_primitives(const std::string& name, bool force_repaint)
	{
		if (Layer* layer = find(name, LayerKind::Primitives))
		{
			if (force_repaint && layer->visible)
				extract(*layer);
			return true;
		}
		return show_primitives(name);
	}
}
