// -- BEGIN LICENSE BLOCK ----------------------------------------------
// This file is part of the GPU Voxels Software Library.
// Licensed under the CDDL (see LICENSE.txt in the top directory).
// -- END LICENSE BLOCK ------------------------------------------------

/*!\file
 * \brief  Camera that orbits a target point (mouse drag rotates, wheel zooms, second button pans). Pure maths, no window system: the viewer window
 *         feeds it mouse movements; tests feed it numbers.
 */
#ifndef GPU_VOXELS_VULKAN_ORBIT_CAMERA_H_INCLUDED
#define GPU_VOXELS_VULKAN_ORBIT_CAMERA_H_INCLUDED

#include <array>

#include <gpu_voxels/vulkan_view/Viewer.h>

namespace gpu_voxels::vulkan
{
	class OrbitCamera
	{
	public:
		/// z is up (the voxel maps of GPU-Voxels are z up): yaw turns around z, pitch is the elevation above the xy plane.
		std::array<float, 3> target{ 0.0f, 0.0f, 0.0f };
		float distance = 10.0f;
		float yaw_degrees = -45.0f;
		float pitch_degrees = 30.0f;
		float fov_y_degrees = 60.0f;

		/// Turn around the target by a mouse movement in pixels (dx to the right turns the scene to the right, dy down looks from higher up).
		void rotate(float dx_pixels, float dy_pixels, float degrees_per_pixel = 0.3f);
		/// Move the target in the image plane by a mouse movement in pixels; `viewport_height` scales the movement so that the scene follows the cursor.
		void pan(float dx_pixels, float dy_pixels, float viewport_height);
		/// Mouse wheel: each step multiplies the distance by factor^-steps (positive steps move closer).
		void zoom(float steps, float factor = 1.15f);
		/// Look at a box from a distance that fits it into the view; keeps yaw and pitch.
		void frame(const std::array<float, 3>& box_min, const std::array<float, 3>& box_max);

		[[nodiscard]] std::array<float, 3> eye() const;
		/// The camera for the renderer; the clipping planes follow the distance.
		[[nodiscard]] ViewerCamera camera() const;
	};

	enum class MouseButton { Left, Right, Middle };

	/// Turns mouse events into camera movements: dragging with the left button rotates, with the right or middle button pans, the wheel zooms.
	/// A cursor movement without a pressed button only remembers the position, so that the next drag does not jump.
	class OrbitController
	{
	public:
		explicit OrbitController(OrbitCamera& camera) : camera_(camera) {}
		void button(MouseButton which, bool pressed);
		/// \param viewport_height height of the window in pixels (the scale of the panning)
		void cursor(float x, float y, float viewport_height);
		void scroll(float steps) { camera_.zoom(steps); }

	private:
		OrbitCamera& camera_;
		bool left_ = false, right_ = false, middle_ = false;
		bool have_last_ = false;
		float last_x_ = 0.0f, last_y_ = 0.0f;
	};
}

#endif
