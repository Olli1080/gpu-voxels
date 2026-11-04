// -- BEGIN LICENSE BLOCK ----------------------------------------------
// This file is part of the GPU Voxels Software Library.
// Licensed under the CDDL (see LICENSE.txt in the top directory).
// -- END LICENSE BLOCK ------------------------------------------------

#include <gpu_voxels/vulkan/ViewerScene.h>

#include <algorithm>

namespace gpu_voxels::vulkan
{
	struct ViewerScene::Layer
	{
		std::string name;
		bool primitives = false;
		bool visible = true;
		GpuVoxelsMapSharedPtr map;
		PrimitiveArraySharedPtr array;
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

	ViewerScene::Layer* ViewerScene::find(const std::string& name, bool primitives)
	{
		for (auto& layer : layers_)
			if (layer->primitives == primitives && layer->name == name)
				return layer.get();
		return nullptr;
	}

	void ViewerScene::extract(Layer& layer)
	{
		if (layer.primitives)
		{
			extractor_.extract(*layer.array, layer.colour, *layer.cubes);
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
		if (Layer* existing = find(map_name, false))
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
		layer->primitives = true;
		layer->array = array;
		layer->colour = colour;
		layer->cubes = std::make_unique<CubeBuffer>(voxels_.context(), capacity_);
		extract(*layer);
		if (Layer* existing = find(array_name, true))
			*existing = std::move(*layer);
		else
			layers_.push_back(std::move(layer));
		return true;
	}

	bool ViewerScene::remove_map(const std::string& map_name)
	{
		const auto it = std::find_if(layers_.begin(), layers_.end(), [&](const auto& l) { return !l->primitives && l->name == map_name; });
		if (it == layers_.end())
			return false;
		layers_.erase(it);
		return true;
	}

	bool ViewerScene::remove_primitives(const std::string& array_name)
	{
		const auto it = std::find_if(layers_.begin(), layers_.end(), [&](const auto& l) { return l->primitives && l->name == array_name; });
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

	std::vector<ViewerScene::LayerInfo> ViewerScene::layers() const
	{
		std::vector<LayerInfo> result;
		for (const auto& layer : layers_)
			result.push_back({ layer->name, layer->primitives, layer->visible, layer->cubes->count(), layer->cubes->dropped() });
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
		if (Layer* layer = find(name, false))
		{
			if (force_repaint && layer->visible)
				extract(*layer);
			return true;
		}
		return show_map(name);
	}

	bool ViewerScene::hook_primitives(const std::string& name, bool force_repaint)
	{
		if (Layer* layer = find(name, true))
		{
			if (force_repaint && layer->visible)
				extract(*layer);
			return true;
		}
		return show_primitives(name);
	}
}
