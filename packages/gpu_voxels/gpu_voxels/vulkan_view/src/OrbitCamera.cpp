// -- BEGIN LICENSE BLOCK ----------------------------------------------
// This file is part of the GPU Voxels Software Library.
// Licensed under the CDDL (see LICENSE.txt in the top directory).
// -- END LICENSE BLOCK ------------------------------------------------

#include <gpu_voxels/vulkan_view/OrbitCamera.h>

#include <algorithm>
#include <cmath>
#include <numbers>

namespace gpu_voxels::vulkan
{
	namespace
	{
		constexpr float kDegrees = std::numbers::pi_v<float> / 180.0f;
		constexpr float kMaxPitch = 89.0f;
		constexpr float kMinDistance = 1e-3f;
	}

	void OrbitCamera::rotate(float dx_pixels, float dy_pixels, float degrees_per_pixel)
	{
		yaw_degrees = std::fmod(yaw_degrees - dx_pixels * degrees_per_pixel, 360.0f);
		pitch_degrees = std::clamp(pitch_degrees + dy_pixels * degrees_per_pixel, -kMaxPitch, kMaxPitch);
	}

	std::array<float, 3> OrbitCamera::eye() const
	{
		const float yaw = yaw_degrees * kDegrees, pitch = pitch_degrees * kDegrees;
		return { target[0] + distance * std::cos(pitch) * std::cos(yaw), target[1] + distance * std::cos(pitch) * std::sin(yaw), target[2] + distance * std::sin(pitch) };
	}

	void OrbitCamera::pan(float dx_pixels, float dy_pixels, float viewport_height)
	{
		if (!(viewport_height > 0.0f))
			return;
		// size of one pixel in world units at the distance of the target
		const float per_pixel = 2.0f * distance * std::tan(fov_y_degrees * kDegrees * 0.5f) / viewport_height;
		const float yaw = yaw_degrees * kDegrees, pitch = pitch_degrees * kDegrees;
		// camera axes: right = (-sin yaw, cos yaw, 0), up = (-sin pitch cos yaw, -sin pitch sin yaw, cos pitch)
		const std::array<float, 3> right{ -std::sin(yaw), std::cos(yaw), 0.0f };
		const std::array<float, 3> up{ -std::sin(pitch) * std::cos(yaw), -std::sin(pitch) * std::sin(yaw), std::cos(pitch) };
		for (int i = 0; i < 3; ++i)
			target[i] += (-dx_pixels * right[i] + dy_pixels * up[i]) * per_pixel;
	}

	void OrbitCamera::zoom(float steps, float factor)
	{
		distance = std::max(kMinDistance, distance * std::pow(factor, -steps));
	}

	void OrbitCamera::frame(const std::array<float, 3>& box_min, const std::array<float, 3>& box_max)
	{
		float radius = 0.0f;
		for (int i = 0; i < 3; ++i)
		{
			target[i] = 0.5f * (box_min[i] + box_max[i]);
			radius += 0.25f * (box_max[i] - box_min[i]) * (box_max[i] - box_min[i]);
		}
		radius = std::max(std::sqrt(radius), kMinDistance); // radius of the bounding sphere
		distance = radius / std::sin(fov_y_degrees * kDegrees * 0.5f) * 1.05f;
	}

	void OrbitController::button(MouseButton which, bool pressed)
	{
		switch (which)
		{
		case MouseButton::Left: left_ = pressed; break;
		case MouseButton::Right: right_ = pressed; break;
		case MouseButton::Middle: middle_ = pressed; break;
		}
	}

	void OrbitController::cursor(float x, float y, float viewport_height)
	{
		if (have_last_)
		{
			const float dx = x - last_x_, dy = y - last_y_;
			if (left_)
				camera_.rotate(dx, dy);
			else if (right_ || middle_)
				camera_.pan(dx, dy, viewport_height);
		}
		have_last_ = true;
		last_x_ = x;
		last_y_ = y;
	}

	ViewerCamera OrbitCamera::camera() const
	{
		ViewerCamera result;
		result.eye = eye();
		result.target = target;
		result.up = { 0.0f, 0.0f, 1.0f };
		result.fov_y_degrees = fov_y_degrees;
		result.z_near = std::max(distance * 0.001f, 1e-4f);
		result.z_far = distance * 1000.0f;
		return result;
	}
}
