// this is for emacs file handling -*- mode: c++; indent-tabs-mode: nil -*-

// -- BEGIN LICENSE BLOCK ----------------------------------------------
// This file is part of the GPU Voxels Software Library.
//
// This program is free software licensed under the CDDL
// (COMMON DEVELOPMENT AND DISTRIBUTION LICENSE Version 1.0).
// You can find a copy of this license in LICENSE.txt in the top
// directory of the source code.
//
// © Copyright 2014 FZI Forschungszentrum Informatik, Karlsruhe, Germany
//
// -- END LICENSE BLOCK ------------------------------------------------

//----------------------------------------------------------------------
/*!\file
 *
 * \author  Matthias Wagner
 * \date    2013-12-2
 *
 *  \brief Camera class for the voxel map visualizer on GPU
 * This class uses a right hand coordinate system.
 * In the world the Z axis points upwards.
 * The camera looks into the direction of positive X. Z points upwards.
 *
 */
 //----------------------------------------------------------------------
#ifndef GPU_VOXELS_VISUALIZATION_CAMERA_GPU_H_INCLUDED
#define GPU_VOXELS_VISUALIZATION_CAMERA_GPU_H_INCLUDED

#include <string>

#include <GL/glew.h>
#if defined (__APPLE__) || defined(MACOSX)
#include <GLUT/glut.h>
#else
#include <GL/freeglut.h>
#endif

#define GLM_FORCE_RADIANS
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>

namespace gpu_voxels {
	namespace visualization {

		class Camera_gpu
		{
		public:

			struct CameraContext
			{
				CameraContext();

				CameraContext(glm::vec3 cam_pos, glm::vec3 cam_focus,
					float horizontal_angle, float vertical_angle, float field_of_view);

				// the position of the camera
				glm::vec3 camera_position;
				// the point where the camera is looking in orbit mode
				glm::vec3 camera_target;

				float h_angle; // the horizontal angle (panning)
				float v_angle; // vertical angle (tilting)
				float foV;     // field of view (RAD)
			};

			Camera_gpu(float window_width, float window_height, CameraContext context);

			~Camera_gpu();

			void resetToInitialValues();
			void resizeWindow(float width, float height);
			///////////////////////////////////// functions for camera movement /////////////////////////////////

			/*!
			 * \brief moveAlongDirection
			 * Moves the camera along the view vector
			 * \param factor Use negative factor to move in negative direction.
			 */
			void moveAlongDirection(float factor);

			/*!
			 * \brief moveAlongRight
			 * Moves the camera along the camera's right vector.
			 * Is disabled in orbit mode.
			 * \param factor Use negative factor to move in negative direction.
			 */
			void moveAlongRight(float factor);

			/*!
			 * \brief moveAlongUp
			 * Move the camera along the camera's up vector.
			 * Is disabled in orbit mode.
			 * \param factor Use negative factor to move in negative direction.
			 */
			void moveAlongUp(float factor);

			/*!
			 * \brief moveFocusPointFromMouseInput
			 * Moves the center point around which the Oribt mode rotates
			 * \param xpos Mouse X Position
			 * \param ypos Mouse Y Position
			 */
			void moveFocusPointFromMouseInput(int32_t xpos, int32_t ypos);

			/**
			 * @brief moveFocusPointVerticalFromMouseInput
			 * Moves the focus point in z direction
			 * \param xpos Mouse X Position
			 * \param ypos Mouse Y Position
			 */
			void moveFocusPointVerticalFromMouseInput(int32_t xpos, int32_t ypos);

			/*!
			 * \brief updateViewMatrixFromMouseInput
			 * Update the view matrix.
			 * Call this function if the camera's right or direction vector
			 * or the camera position have changed.
			 * \param xpos Mouse X Position
			 * \param ypos Mouse Y Position
			 */
			void updateViewMatrixFromMouseInput(int32_t xpos, int32_t ypos);

			/*!
			 * \brief toggleCameraMode
			 * Switches between Orbit and Free flight mode
			 */
			void toggleCameraMode();

			[[nodiscard]] bool hasViewChanged() const;
			void setViewChanged(bool view_changed);

			//////////////////////////////////////// getter / setter /////////////////////////////////////////////
			[[nodiscard]] glm::vec3 getCameraDirection() const;

			[[nodiscard]] glm::vec3 getCameraPosition() const;

			[[nodiscard]] glm::vec3 getCameraTarget() const;

			void setCameraTarget(glm::vec3 camera_target);

			void setCameraTargetOfInitContext(glm::vec3 camera_target);

			[[nodiscard]] glm::mat4 getProjectionMatrix() const;

			[[nodiscard]] glm::mat4 getViewMatrix() const;

			[[nodiscard]] float getWindowHeight() const;

			void setWindowHeight(float windowHeight);

			[[nodiscard]] float getWindowWidth() const;

			void setWindowWidth(float windowWidth);

			void setMousePosition(int32_t x, int32_t y);

			[[nodiscard]] std::string getCameraInfo() const;
			// --- debug functions ---
			void printCameraPosDirR() const;

			[[nodiscard]] glm::vec3 getCameraRight() const;
			// --- end debug functions ---
			bool m_camera_orbit;

		private:

			void updateViewMatrix();

			void updateProjectionMatrix();

			void updateRotationMatrix();

			void updateCameraDirection();

			void updateCameraRight();

			// the initial context of the camera
			CameraContext m_init_context;
			// the current context of the camera
			CameraContext m_cur_context;

			glm::mat4 m_view_matrix;
			glm::mat4 m_projection_matrix;

			glm::vec3 m_camera_direction;
			glm::vec3 m_camera_right;

			float m_speed;
			float m_mouse_speed;

			float m_window_width;
			float m_window_height;


			int32_t m_mouse_old_x;
			int32_t m_mouse_old_y;

			bool m_has_view_changed;

		};
	} // end of namespace visualization
} // end of namespace gpu_voxels
#endif