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
 * \date    2014-02-24
 *
 * \brief this primitive represents a sphere.
 */
 //----------------------------------------------------------------------
#ifndef GPU_VOXELS_VISUALIZATION_SPHERE_H_INCLUDED
#define GPU_VOXELS_VISUALIZATION_SPHERE_H_INCLUDED

#include <gpu_visualization/Primitive.h>


namespace gpu_voxels {
	namespace visualization {

		class Sphere : public Primitive
		{
		public:

			class Elementbuffer
			{
			public:

				~Elementbuffer();

				void gl_generate(const std::vector<uint32_t>& indices);

				void gl_draw(GLenum mode, uint32_t number_of_draws);

			private:

				GLuint instance = 0;
				uint32_t size = 0;
			};

			Sphere();

			Sphere(glm::vec4 color, glm::vec3 position, float radius, uint32_t resolution);

			~Sphere() override;

			/**
			 * Creates all necessary VBOs for this sphere.
			 */
			void create(bool with_lighting) override;

			/**
			 * draw the sphere multiple times.
			 * All uniform variables of the shaders must be set before call.
			 *
			 * @param number_of_draws: the number of draw calls for this sphere.
			 */
			void draw(uint32_t number_of_draws, bool with_lighting) override;

		private:

			glm::vec3 m_position;
			float m_radius;

			uint32_t longitudeEntries;
			uint32_t latitudeEntries;

			Elementbuffer north_pole;
			Elementbuffer south_pole;
			Elementbuffer sphere_body;
		};

	} // end of namespace visualization
} // end of namespace gpu_voxels

#endif
