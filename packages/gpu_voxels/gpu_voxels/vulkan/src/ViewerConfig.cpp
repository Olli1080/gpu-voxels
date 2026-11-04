// -- BEGIN LICENSE BLOCK ----------------------------------------------
// This file is part of the GPU Voxels Software Library.
// Licensed under the CDDL (see LICENSE.txt in the top directory).
// -- END LICENSE BLOCK ------------------------------------------------

#include <gpu_voxels/vulkan/ViewerConfig.h>

#include <charconv>
#include <cmath>
#include <fstream>
#include <initializer_list>
#include <set>
#include <sstream>
#include <string_view>

#include <tinyxml2.h>

namespace gpu_voxels::vulkan
{
	namespace
	{
		using tinyxml2::XMLElement;

		[[noreturn]] void fail(const XMLElement* element, const std::string& what)
		{
			throw VulkanError("ViewerConfig: line " + std::to_string(element ? element->GetLineNum() : 0) + ": " + what);
		}

		/// Only these attributes are allowed.
		void check_attributes(const XMLElement& element, std::initializer_list<const char*> allowed)
		{
			for (const tinyxml2::XMLAttribute* a = element.FirstAttribute(); a; a = a->Next())
			{
				bool known = false;
				for (const char* name : allowed)
					known = known || std::string_view(a->Name()) == name;
				if (!known)
					fail(&element, std::string("unknown attribute '") + a->Name() + "' of <" + element.Name() + ">");
			}
		}

		/// Only these child elements are allowed.
		void check_children(const XMLElement& element, std::initializer_list<const char*> allowed)
		{
			for (const XMLElement* c = element.FirstChildElement(); c; c = c->NextSiblingElement())
			{
				bool known = false;
				for (const char* name : allowed)
					known = known || std::string_view(c->Name()) == name;
				if (!known)
					fail(c, std::string("unknown element <") + c->Name() + "> inside <" + element.Name() + ">");
			}
		}

		uint32_t parse_colour_text(const XMLElement& where, std::string_view text)
		{
			if (text == "none")
				return 0;
			const auto bad = [&] { fail(&where, "'" + std::string(text) + "' is not a colour (#rrggbb, #rrggbbaa or none)"); };
			if ((text.size() != 7 && text.size() != 9) || text[0] != '#')
				bad();
			uint32_t channels[4] = { 0, 0, 0, 255 };
			for (std::size_t i = 0; i * 2 + 1 < text.size(); ++i)
			{
				uint32_t value = 0;
				const auto r = std::from_chars(text.data() + 1 + i * 2, text.data() + 3 + i * 2, value, 16);
				if (r.ec != std::errc() || r.ptr != text.data() + 3 + i * 2)
					bad();
				channels[i] = value;
			}
			return pack_colour(static_cast<uint8_t>(channels[0]), static_cast<uint8_t>(channels[1]), static_cast<uint8_t>(channels[2]), static_cast<uint8_t>(channels[3]));
		}

		std::optional<std::string> attribute(const XMLElement& element, const char* name)
		{
			const char* value = element.Attribute(name);
			return value ? std::optional<std::string>(value) : std::nullopt;
		}

		std::string required(const XMLElement& element, const char* name)
		{
			const auto value = attribute(element, name);
			if (!value || value->empty())
				fail(&element, std::string("<") + element.Name() + "> needs the attribute '" + name + "'");
			return *value;
		}

		bool parse_bool(const XMLElement& element, const char* name, const std::string& text)
		{
			if (text == "true")
				return true;
			if (text == "false")
				return false;
			fail(&element, std::string("'") + name + "' must be true or false, not '" + text + "'");
		}

		float parse_float(const XMLElement& element, const char* name, const std::string& text)
		{
			float value = 0.0f;
			const auto r = std::from_chars(text.data(), text.data() + text.size(), value);
			if (r.ec != std::errc() || r.ptr != text.data() + text.size() || !std::isfinite(value))
				fail(&element, std::string("'") + name + "' must be a number, not '" + text + "'");
			return value;
		}

		int parse_int(const XMLElement& element, const char* name, const std::string& text, int low, int high)
		{
			int value = 0;
			const auto r = std::from_chars(text.data(), text.data() + text.size(), value);
			if (r.ec != std::errc() || r.ptr != text.data() + text.size() || value < low || value > high)
				fail(&element, std::string("'") + name + "' must be an integer from " + std::to_string(low) + " to " + std::to_string(high) + ", not '" + text + "'");
			return value;
		}

		void parse_palette(const XMLElement& element, BitPalette& palette)
		{
			check_attributes(element, { "default" });
			check_children(element, { "meaning" });
			if (const auto d = attribute(element, "default"))
			{
				if (*d == "hidden")
					palette = BitPalette{};
				else if (*d != "hues")
					fail(&element, "'default' must be hues or hidden, not '" + *d + "'");
			}
			std::set<int> seen;
			for (const XMLElement* m = element.FirstChildElement("meaning"); m; m = m->NextSiblingElement("meaning"))
			{
				check_attributes(*m, { "id", "colour", "hidden" });
				const int id = parse_int(*m, "id", required(*m, "id"), 0, 255);
				if (!seen.insert(id).second)
					fail(m, "meaning " + std::to_string(id) + " is listed twice");
				const auto colour = attribute(*m, "colour");
				const auto hidden = attribute(*m, "hidden");
				if (colour.has_value() == hidden.has_value())
					fail(m, "<meaning> needs exactly one of 'colour' and 'hidden'");
				if (colour)
					palette.set(static_cast<uint32_t>(id), parse_colour_text(*m, *colour));
				else if (parse_bool(*m, "hidden", *hidden))
					palette.hide(static_cast<uint32_t>(id));
				else
					fail(m, "hidden=\"false\" says nothing: give a colour instead");
			}
		}

		void parse_octree(const XMLElement& element, OctreeDisplay& display)
		{
			check_attributes(element, { "min_level", "side_length", "occupied", "free", "collision", "min_value" });
			if (const auto v = attribute(element, "min_level"))
				display.min_level = static_cast<uint32_t>(parse_int(element, "min_level", *v, 0, 13));
			if (const auto v = attribute(element, "side_length"))
			{
				display.side_length = parse_float(element, "side_length", *v);
				if (display.side_length < 0.0f)
					fail(&element, "'side_length' must not be negative");
			}
			if (const auto v = attribute(element, "occupied"))
				display.occupied_colour = parse_colour_text(element, *v);
			if (const auto v = attribute(element, "free"))
				display.free_colour = parse_colour_text(element, *v);
			if (const auto v = attribute(element, "collision"))
				display.collision_colour = parse_colour_text(element, *v);
			if (const auto v = attribute(element, "min_value"))
				display.min_value = parse_int(element, "min_value", *v, -1000, 1000);
		}

		ViewerLayerConfig parse_layer(const XMLElement& element)
		{
			check_attributes(element, { "name", "type", "colour", "snap_to_voxels", "point_size" });
			ViewerLayerConfig layer;
			layer.name = required(element, "name");
			const std::string type = required(element, "type");
			if (type == "map")
				layer.kind = ViewerScene::LayerKind::Map;
			else if (type == "primitives")
				layer.kind = ViewerScene::LayerKind::Primitives;
			else if (type == "robot")
				layer.kind = ViewerScene::LayerKind::Robot;
			else
				fail(&element, "'type' must be map, primitives or robot, not '" + type + "'");

			const auto only_for = [&](const char* attribute_name, ViewerScene::LayerKind kind, const char* kind_name)
			{
				if (element.Attribute(attribute_name) && layer.kind != kind)
					fail(&element, std::string("'") + attribute_name + "' only belongs to a layer of type " + kind_name);
			};
			only_for("colour", ViewerScene::LayerKind::Primitives, "primitives");
			only_for("snap_to_voxels", ViewerScene::LayerKind::Robot, "robot");
			only_for("point_size", ViewerScene::LayerKind::Robot, "robot");

			switch (layer.kind)
			{
			case ViewerScene::LayerKind::Map:
			{
				check_children(element, { "palette", "prob", "octree" });
				for (const char* name : { "palette", "prob", "octree" })
					if (element.FirstChildElement(name) && element.FirstChildElement(name)->NextSiblingElement(name))
						fail(element.FirstChildElement(name)->NextSiblingElement(name), std::string("<") + name + "> appears twice");
				if (const XMLElement* p = element.FirstChildElement("palette"))
					parse_palette(*p, layer.map.palette);
				if (const XMLElement* p = element.FirstChildElement("prob"))
				{
					check_attributes(*p, { "min_value" });
					if (const auto v = attribute(*p, "min_value"))
						layer.map.prob.min_value = parse_int(*p, "min_value", *v, -1000, 1000);
				}
				if (const XMLElement* o = element.FirstChildElement("octree"))
					parse_octree(*o, layer.map.octree);
				break;
			}
			case ViewerScene::LayerKind::Primitives:
				check_children(element, {});
				if (const auto v = attribute(element, "colour"))
					layer.colour = parse_colour_text(element, *v);
				break;
			case ViewerScene::LayerKind::Robot:
				check_children(element, { "colour" });
				if (const auto v = attribute(element, "snap_to_voxels"))
					layer.robot.snap_to_voxels = parse_bool(element, "snap_to_voxels", *v);
				if (const auto v = attribute(element, "point_size"))
				{
					layer.robot.point_size = parse_float(element, "point_size", *v);
					if (!(layer.robot.point_size > 0.0f))
						fail(&element, "'point_size' must be positive");
				}
				for (const XMLElement* c = element.FirstChildElement("colour"); c; c = c->NextSiblingElement("colour"))
				{
					check_attributes(*c, {});
					check_children(*c, {});
					layer.robot.colours.push_back(parse_colour_text(*c, c->GetText() ? c->GetText() : ""));
				}
				break;
			}
			return layer;
		}
	}

	void ViewerConfig::apply(RenderOptions& options) const
	{
		if (background)
			options.clear_colour = *background;
		if (shaded)
			options.shaded = *shaded;
	}

	ViewerConfig parse_viewer_config(const std::string& xml)
	{
		tinyxml2::XMLDocument document;
		if (document.Parse(xml.c_str(), xml.size()) != tinyxml2::XML_SUCCESS)
			throw VulkanError("ViewerConfig: line " + std::to_string(document.ErrorLineNum()) + ": " + document.ErrorStr());
		const XMLElement* root = document.RootElement();
		if (!root || std::string_view(root->Name()) != "viewer")
			fail(root, "the root element must be <viewer>");
		check_attributes(*root, {});
		check_children(*root, { "window", "layer" });

		ViewerConfig config;
		if (const XMLElement* window = root->FirstChildElement("window"))
		{
			if (window->NextSiblingElement("window"))
				fail(window->NextSiblingElement("window"), "<window> appears twice");
			check_attributes(*window, { "background", "shaded" });
			check_children(*window, {});
			if (const auto v = attribute(*window, "background"))
			{
				const uint32_t c = parse_colour_text(*window, *v);
				config.background = std::array<float, 4>{ static_cast<float>(c & 255u) / 255.0f, static_cast<float>((c >> 8) & 255u) / 255.0f,
					static_cast<float>((c >> 16) & 255u) / 255.0f, static_cast<float>(c >> 24) / 255.0f };
			}
			if (const auto v = attribute(*window, "shaded"))
				config.shaded = parse_bool(*window, "shaded", *v);
		}
		std::set<std::pair<int, std::string>> seen;
		for (const XMLElement* e = root->FirstChildElement("layer"); e; e = e->NextSiblingElement("layer"))
		{
			ViewerLayerConfig layer = parse_layer(*e);
			if (!seen.insert({ static_cast<int>(layer.kind), layer.name }).second)
				fail(e, "layer '" + layer.name + "' of this type is listed twice");
			config.layers.push_back(std::move(layer));
		}
		return config;
	}

	ViewerConfig load_viewer_config(const std::filesystem::path& file)
	{
		std::ifstream in(file, std::ios::binary);
		if (!in)
			throw VulkanError("ViewerConfig: cannot read " + file.string());
		std::ostringstream text;
		text << in.rdbuf();
		return parse_viewer_config(text.str());
	}

	std::vector<std::string> show_layers(ViewerScene& scene, const ViewerConfig& config)
	{
		std::vector<std::string> failed;
		for (const ViewerLayerConfig& layer : config.layers)
		{
			bool shown = false;
			const char* prefix = "map:";
			switch (layer.kind)
			{
			case ViewerScene::LayerKind::Map: shown = scene.show_map(layer.name, layer.map); break;
			case ViewerScene::LayerKind::Primitives: shown = scene.show_primitives(layer.name, layer.colour); prefix = "primitives:"; break;
			case ViewerScene::LayerKind::Robot: shown = scene.show_robot(layer.name, layer.robot); prefix = "robot:"; break;
			}
			if (!shown)
				failed.push_back(prefix + layer.name);
		}
		return failed;
	}
}
