// -- BEGIN LICENSE BLOCK ----------------------------------------------
// This file is part of the GPU Voxels Software Library.
// Licensed under the CDDL (see LICENSE.txt in the top directory).
// -- END LICENSE BLOCK ------------------------------------------------

#include <gpu_voxels/vulkan/PointCloudFile.h>

#include <algorithm>
#include <charconv>
#include <cmath>
#include <cstring>
#include <fstream>
#include <limits>
#include <string>
#include <string_view>

#include <gpu_voxels/vulkan/Log.h>

#ifdef _BUILD_GVL_WITH_PCL_SUPPORT_
#include <pcl/io/pcd_io.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#endif

namespace gpu_voxels::vulkan
{
	namespace
	{
		std::string read_all(const std::filesystem::path& file)
		{
			std::ifstream in(file, std::ios::in | std::ios::binary);
			if (!in)
				throw PointCloudFileError("could not open the file " + file.string());
			return std::string(std::istreambuf_iterator<char>(in), std::istreambuf_iterator<char>());
		}

		[[noreturn]] void fail(const std::filesystem::path& file, const std::string& what)
		{
			throw PointCloudFileError(file.filename().string() + ": " + what);
		}

		std::vector<std::string_view> split(std::string_view line)
		{
			std::vector<std::string_view> tokens;
			size_t i = 0;
			while (i < line.size())
			{
				while (i < line.size() && (line[i] == ' ' || line[i] == '\t' || line[i] == '\r'))
					++i;
				const size_t start = i;
				while (i < line.size() && !(line[i] == ' ' || line[i] == '\t' || line[i] == '\r'))
					++i;
				if (i > start)
					tokens.push_back(line.substr(start, i - start));
			}
			return tokens;
		}

		template<class T>
		bool parse_number(std::string_view token, T& value)
		{
			if (!token.empty() && token.front() == '+')
				token.remove_prefix(1); // from_chars does not take a plus sign
			const auto r = std::from_chars(token.data(), token.data() + token.size(), value);
			return r.ec == std::errc() && r.ptr == token.data() + token.size();
		}

		/// Calls f(line, line number) for every line (without the end of line characters).
		template<class F>
		void for_each_line(std::string_view text, F&& f)
		{
			size_t pos = 0, number = 0;
			while (pos < text.size())
			{
				size_t end = text.find('\n', pos);
				if (end == std::string_view::npos)
					end = text.size();
				std::string_view line = text.substr(pos, end - pos);
				if (!line.empty() && line.back() == '\r')
					line.remove_suffix(1);
				f(line, ++number);
				pos = end + 1;
			}
		}

		std::string lower(std::string s)
		{
			std::transform(s.begin(), s.end(), s.begin(), [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
			return s;
		}
	}

	// ---------------------------------------------------------------- xyz

	std::vector<Point3f> read_xyz_file(const std::filesystem::path& file)
	{
		const std::string text = read_all(file);
		std::vector<Point3f> points;
		size_t skipped = 0;
		for_each_line(text, [&](std::string_view line, size_t)
		{
			const auto tokens = split(line);
			if (tokens.empty() || tokens[0].front() == '#')
				return;
			Point3f p{};
			if (tokens.size() >= 3 && parse_number(tokens[0], p.x) && parse_number(tokens[1], p.y) && parse_number(tokens[2], p.z))
				points.push_back(p);
			else
				++skipped;
		});
		if (skipped > 0)
			GVL_LOG_WARNING(PointCloudFile, file.filename().string() << ": " << skipped << " line(s) do not start with three numbers and were skipped");
		return points;
	}

	// ---------------------------------------------------------------- pcd

	std::vector<Point3f> read_pcd_file(const std::filesystem::path& file)
	{
#ifdef _BUILD_GVL_WITH_PCL_SUPPORT_
		if (!std::filesystem::exists(file))
			throw PointCloudFileError("could not open the file " + file.string());
		pcl::PointCloud<pcl::PointXYZ> cloud;
		try
		{
			if (pcl::io::loadPCDFile<pcl::PointXYZ>(file.string(), cloud) == -1)
				fail(file, "PCL could not read the PCD file");
		}
		catch (const PointCloudFileError&)
		{
			throw;
		}
		catch (const std::exception& e)
		{
			fail(file, std::string("PCL could not read the PCD file: ") + e.what());
		}
		std::vector<Point3f> points;
		points.reserve(cloud.size());
		for (const auto& p : cloud.points)
			points.push_back({ p.x, p.y, p.z });
		return points;
#else
		fail(file, ".pcd files need a build with GVL_WITH_PCL");
#endif
	}

	// ---------------------------------------------------------------- binvox

	std::vector<Point3f> read_binvox_file(const std::filesystem::path& file)
	{
		const std::string text = read_all(file);
		size_t pos = 0;
		auto token = [&]() -> std::string_view {
			while (pos < text.size() && std::isspace(static_cast<unsigned char>(text[pos])))
				++pos;
			const size_t start = pos;
			while (pos < text.size() && !std::isspace(static_cast<unsigned char>(text[pos])))
				++pos;
			return std::string_view(text).substr(start, pos - start);
		};
		auto number = [&](auto& value) {
			const std::string_view t = token();
			if (!parse_number(t, value))
				fail(file, "bad number '" + std::string(t) + "' in the header");
		};

		if (token() != "#binvox")
			fail(file, "the first word is not #binvox");
		int version = 0;
		number(version);
		int depth = -1, height = 0, width = 0;
		float tx = 0, ty = 0, tz = 0, scale = 1;
		bool done = false;
		while (!done)
		{
			const std::string_view key = token();
			if (key.empty())
				fail(file, "the header ends without a data line");
			if (key == "data")
				done = true;
			else if (key == "dim")
			{
				number(depth);
				number(height);
				number(width);
			}
			else if (key == "translate")
			{
				number(tx);
				number(ty);
				number(tz);
			}
			else if (key == "scale")
				number(scale);
			else
			{
				while (pos < text.size() && text[pos] != '\n') // unknown keyword: skip the line
					++pos;
			}
		}
		if (depth < 0)
			fail(file, "the header has no dim line");
		if (depth <= 0 || height <= 0 || width <= 0)
			fail(file, "the dimensions must be positive");
		const uint64_t size = uint64_t(depth) * height * width;
		const float step = scale / static_cast<float>(width);

		if (pos < text.size())
			++pos; // the line feed after "data"
		std::vector<Point3f> points;
		uint64_t index = 0;
		const auto* bytes = reinterpret_cast<const unsigned char*>(text.data());
		while (index < size)
		{
			if (pos + 2 > text.size())
				fail(file, "the data ends at voxel " + std::to_string(index) + " of " + std::to_string(size));
			const unsigned char value = bytes[pos], count = bytes[pos + 1];
			pos += 2;
			if (index + count > size)
				fail(file, "a run reaches beyond the volume");
			if (value == 1)
				for (uint64_t i = index; i < index + count; ++i)
				{
					// x is the most significant axis, then z, then y
					const uint64_t y = i % static_cast<uint64_t>(width);
					const uint64_t z = (i / static_cast<uint64_t>(width)) % static_cast<uint64_t>(height);
					const uint64_t x = i / (static_cast<uint64_t>(width) * static_cast<uint64_t>(height));
					points.push_back({ step * static_cast<float>(x) + tx, step * static_cast<float>(y) + ty, step * static_cast<float>(z) + tz });
				}
			index += count;
		}
		return points;
	}

	std::vector<Point3f> read_point_cloud_file(const std::filesystem::path& file)
	{
		const std::string extension = lower(file.extension().string());
		if (extension == ".xyz")
			return read_xyz_file(file);
		if (extension == ".pcd")
			return read_pcd_file(file);
		if (extension == ".binvox")
			return read_binvox_file(file);
		throw PointCloudFileError(file.string() + " has no known file format (.xyz, .pcd, .binvox)");
	}

	void shift_point_cloud_to_zero(std::vector<Point3f>& points)
	{
		float lo[3] = { std::numeric_limits<float>::infinity(), std::numeric_limits<float>::infinity(), std::numeric_limits<float>::infinity() };
		bool any = false;
		for (const Point3f& p : points)
			if (std::isfinite(p.x) && std::isfinite(p.y) && std::isfinite(p.z))
			{
				lo[0] = std::min(lo[0], p.x);
				lo[1] = std::min(lo[1], p.y);
				lo[2] = std::min(lo[2], p.z);
				any = true;
			}
		if (!any)
			return;
		for (Point3f& p : points)
		{
			p.x -= lo[0];
			p.y -= lo[1];
			p.z -= lo[2];
		}
	}

	std::vector<Point3f> load_point_cloud(const std::filesystem::path& file, const std::filesystem::path& model_path,
		const PointCloudLoadOptions& options)
	{
		std::vector<Point3f> points = read_point_cloud_file(model_path / file);
		if (options.shift_to_zero)
			shift_point_cloud_to_zero(points);
		for (Point3f& p : points)
			p = { options.scaling * p.x + options.offset.x, options.scaling * p.y + options.offset.y, options.scaling * p.z + options.offset.z };
		return points;
	}

	// ---------------------------------------------------------------- box

	namespace
	{
		uint64_t saturating_mul(uint64_t a, uint64_t b)
		{
			return (b != 0 && a > std::numeric_limits<uint64_t>::max() / b) ? std::numeric_limits<uint64_t>::max() : a * b;
		}

		int64_t voxel_bound(float value, const char* what)
		{
			if (!std::isfinite(value) || std::fabs(value) > 2147483648.0f)
				throw VulkanError(std::string("the box ") + what + " is not a voxel coordinate within +-2^31");
			return static_cast<int64_t>(value);
		}
	}

	BoxVoxels box_voxel_coordinates(Point3f min, Point3f max, float spacing, float voxel_side_length, Dim3ui limit)
	{
		if (!(voxel_side_length > 0.0f) || !std::isfinite(voxel_side_length))
			throw VulkanError("the voxel side length must be positive and finite");
		if (!(spacing > 0.0f) || !std::isfinite(spacing))
			throw VulkanError("the spacing of the box must be positive and finite");
		const float lows[3] = { min.x, min.y, min.z }, highs[3] = { max.x, max.y, max.z };
		const uint32_t limits[3] = { limit.x, limit.y, limit.z };
		const float step_f = std::round(spacing / voxel_side_length);
		const int64_t step = step_f >= 1.0f ? (step_f > 2147483648.0f ? int64_t(1) << 31 : static_cast<int64_t>(step_f)) : 1;

		int64_t first[3];
		uint64_t count[3], inside[3];
		BoxVoxels box;
		box.total = 1;
		uint64_t generated = 1;
		for (int a = 0; a < 3; ++a)
		{
			if (!std::isfinite(lows[a]) || !std::isfinite(highs[a]))
				throw VulkanError("the corners of the box must be finite");
			const float cropped = lows[a] >= 0.0f ? lows[a] : 0.0f; // CUDA: the minimum is cut at 0
			const int64_t lo = voxel_bound(std::floor(cropped / voxel_side_length), "minimum");
			const int64_t hi = voxel_bound(std::ceil(highs[a] / voxel_side_length), "maximum");
			if (lo > hi)
				throw VulkanError("the box is empty: its voxel minimum lies above its maximum");
			first[a] = lo;
			count[a] = static_cast<uint64_t>((hi - lo) / step + 1);
			inside[a] = lo >= int64_t(limits[a]) ? 0 : std::min<uint64_t>(count[a], static_cast<uint64_t>((int64_t(limits[a]) - lo - 1) / step + 1));
			box.total = saturating_mul(box.total, count[a]);
			generated = saturating_mul(generated, inside[a]);
		}
		if (generated > kMaxBoxVoxels)
			throw VulkanError("the box has " + std::to_string(generated) + " voxels inside the map, at most " + std::to_string(kMaxBoxVoxels) + " are allowed per call");
		box.coordinates.reserve(static_cast<size_t>(generated));
		for (uint64_t x = 0; x < inside[0]; ++x)
			for (uint64_t y = 0; y < inside[1]; ++y)
				for (uint64_t z = 0; z < inside[2]; ++z)
					box.coordinates.push_back({ static_cast<uint32_t>(first[0] + int64_t(x) * step), static_cast<uint32_t>(first[1] + int64_t(y) * step),
						static_cast<uint32_t>(first[2] + int64_t(z) * step) });
		return box;
	}
}
