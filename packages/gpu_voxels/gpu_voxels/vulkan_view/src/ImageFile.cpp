// -- BEGIN LICENSE BLOCK ----------------------------------------------
// This file is part of the GPU Voxels Software Library.
// Licensed under the CDDL (see LICENSE.txt in the top directory).
// -- END LICENSE BLOCK ------------------------------------------------

#include <gpu_voxels/vulkan_view/ImageFile.h>

#include <algorithm>
#include <array>
#include <fstream>

namespace gpu_voxels::vulkan
{
	namespace
	{
		uint32_t crc32(std::span<const uint8_t> bytes, uint32_t crc = 0)
		{
			static const auto table = []
			{
				std::array<uint32_t, 256> t{};
				for (uint32_t n = 0; n < 256; ++n)
				{
					uint32_t c = n;
					for (int k = 0; k < 8; ++k)
						c = (c & 1u) ? 0xEDB88320u ^ (c >> 1) : c >> 1;
					t[n] = c;
				}
				return t;
			}();
			crc = ~crc;
			for (const uint8_t b : bytes)
				crc = table[(crc ^ b) & 0xFFu] ^ (crc >> 8);
			return ~crc;
		}

		void put32(std::vector<uint8_t>& out, uint32_t v)
		{
			for (int shift = 24; shift >= 0; shift -= 8)
				out.push_back(static_cast<uint8_t>(v >> shift));
		}

		void chunk(std::vector<uint8_t>& out, const char (&type)[5], std::span<const uint8_t> data)
		{
			put32(out, static_cast<uint32_t>(data.size()));
			const std::size_t start = out.size();
			out.insert(out.end(), type, type + 4);
			out.insert(out.end(), data.begin(), data.end());
			put32(out, crc32(std::span<const uint8_t>(out.data() + start, out.size() - start)));
		}
	}

	std::vector<uint8_t> encode_png(uint32_t width, uint32_t height, std::span<const uint8_t> rgba)
	{
		if (width == 0 || height == 0 || rgba.size() != static_cast<std::size_t>(width) * height * 4)
			throw VulkanError("encode_png: the pixel data does not match the image size");

		// scanlines: filter byte 0 + the row
		std::vector<uint8_t> raw;
		raw.reserve((static_cast<std::size_t>(width) * 4 + 1) * height);
		for (uint32_t y = 0; y < height; ++y)
		{
			raw.push_back(0);
			raw.insert(raw.end(), rgba.begin() + static_cast<std::ptrdiff_t>(y) * width * 4, rgba.begin() + static_cast<std::ptrdiff_t>(y + 1) * width * 4);
		}

		// zlib stream of stored deflate blocks
		std::vector<uint8_t> z{ 0x78, 0x01 };
		uint32_t a = 1, b = 0;
		for (const uint8_t byte : raw)
		{
			a = (a + byte) % 65521u;
			b = (b + a) % 65521u;
		}
		for (std::size_t pos = 0; pos < raw.size();)
		{
			const std::size_t n = std::min<std::size_t>(65535, raw.size() - pos);
			z.push_back(pos + n == raw.size() ? 1 : 0);
			z.push_back(static_cast<uint8_t>(n & 0xFF));
			z.push_back(static_cast<uint8_t>(n >> 8));
			z.push_back(static_cast<uint8_t>(~n & 0xFF));
			z.push_back(static_cast<uint8_t>((~n >> 8) & 0xFF));
			z.insert(z.end(), raw.begin() + static_cast<std::ptrdiff_t>(pos), raw.begin() + static_cast<std::ptrdiff_t>(pos + n));
			pos += n;
		}
		put32(z, (b << 16) | a);

		std::vector<uint8_t> out{ 0x89, 'P', 'N', 'G', 0x0D, 0x0A, 0x1A, 0x0A };
		std::vector<uint8_t> header;
		put32(header, width);
		put32(header, height);
		header.insert(header.end(), { 8, 6, 0, 0, 0 }); // 8 bit, RGBA, deflate, no filter method, no interlace
		chunk(out, "IHDR", header);
		chunk(out, "IDAT", z);
		chunk(out, "IEND", {});
		return out;
	}

	void write_png(const std::filesystem::path& file, uint32_t width, uint32_t height, std::span<const uint8_t> rgba)
	{
		const auto bytes = encode_png(width, height, rgba);
		std::ofstream out(file, std::ios::binary | std::ios::trunc);
		out.write(reinterpret_cast<const char*>(bytes.data()), static_cast<std::streamsize>(bytes.size()));
		if (!out)
			throw VulkanError("write_png: cannot write " + file.string());
	}
}
