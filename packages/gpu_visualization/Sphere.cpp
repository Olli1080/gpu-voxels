#include "Sphere.h"

#include "constants.hpp"

namespace gpu_voxels
{
	namespace visualization
	{
		Sphere::Elementbuffer::~Elementbuffer()
		{
			glDeleteBuffers(1, &instance);
		}

		void Sphere::Elementbuffer::gl_generate(const std::vector<uint32_t>& indices)
		{
			size = static_cast<uint32_t>(indices.size());

			glDeleteBuffers(1, &instance);

			glGenBuffers(1, &instance);
			glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, instance);
			glBufferData(GL_ELEMENT_ARRAY_BUFFER, static_cast<GLsizeiptr>(size) * sizeof(uint32_t),
				indices.data(), GL_STATIC_DRAW);
		}

		void Sphere::Elementbuffer::gl_draw(GLenum mode, uint32_t number_of_draws)
		{
			glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, instance);

			// Draw the triangles !
			glDrawElementsInstanced(mode, // mode
				size, // count
				GL_UNSIGNED_INT, // type
				nullptr // element array buffer offset
				, number_of_draws);

			//TODO:: maybe unbind
		}


		Sphere::Sphere()
			: m_position(0),
			m_radius(10),
			longitudeEntries(16),
			latitudeEntries(16)
		{
			m_color = glm::vec4(1.f);
			m_is_created = false;
		}

		Sphere::Sphere(glm::vec4 color, glm::vec3 position, float radius, uint32_t resolution)
			: m_position(position),
			m_radius(radius),
			longitudeEntries(resolution),
			latitudeEntries(resolution)
		{
			m_color = color;
			m_is_created = false;
		}

		Sphere::~Sphere()
		{
			glDeleteBuffers(1, &m_vbo);
		}

		void Sphere::create(bool with_lighting)
		{
			std::vector<uint32_t> indices_north_pole;
			std::vector<uint32_t> indices_south_pole;
			std::vector<uint32_t> indices_body;

			std::vector<glm::vec3> vertices;
			std::vector<uint32_t> indices;

			for (float j = 1; j < latitudeEntries + 1; j++)
			{
				for (float i = 0; i < longitudeEntries; i++)
				{
					float latitude = j / (latitudeEntries + 1) * M_PI<float> -M_PI_2<float>; // ]-PI/2, PI/2[ without the edges, so that the poles get excluded
					float longitude = i / (longitudeEntries) * 2 * M_PI<float>; // [0, 2*PI]
					float x = m_radius * cosf(latitude) * cosf(longitude) + m_position.x;
					float z = m_radius * cosf(latitude) * sinf(longitude) + m_position.z;
					float y = m_radius * sinf(latitude) + m_position.y;
					vertices.emplace_back(x, y, z);
				}
			}
			// assert(vertices.size() == size_t(longitudeEntries * latitudeEntries));
			//north pole of the sphere
			vertices.push_back(m_position + glm::vec3(0, m_radius, 0));

			//south pole of the sphere
			vertices.push_back(m_position - glm::vec3(0, m_radius, 0));

			///////////////////////////////insert indices of north pole
			indices_north_pole.push_back(vertices.size() - 2); // north pole is at index (size - 2)
			for (int32_t i = longitudeEntries - 1; i >= 0; i--)
			{
				uint32_t index = (latitudeEntries - 1) * longitudeEntries + i;
				indices_north_pole.push_back(index);
			}
			//to close the fan add the last point from points again
			indices_north_pole.push_back((latitudeEntries - 1) * longitudeEntries + longitudeEntries - 1);

			////////////////////////////////////insert indices of south pole
			indices_south_pole.push_back(static_cast<uint32_t>(vertices.size() - 1)); // south pole is at index (size - 1)
			for (uint32_t i = 0; i < longitudeEntries; i++)
			{
				indices_south_pole.push_back(i);
			}
			// to close the fan add the first point from points again
			indices_south_pole.push_back(0);

			//////////////////////////////// insert indices of sphere body
			uint32_t index = 0;
			for (uint32_t latitude = 1; latitude < latitudeEntries; latitude++)
			{
				const uint32_t next_index = longitudeEntries * latitude;
				for (uint32_t longitude = 0; longitude < longitudeEntries; longitude++)
				{
					indices_body.push_back(index + longitude);
					indices_body.push_back(next_index + longitude);
				}
				// repeat the first one to close the strip
				indices_body.push_back(index);
				indices_body.push_back(next_index);
				indices_body.push_back(0xffff); //insert restart index
				index = next_index;
			}
			glDeleteBuffers(1, &m_vbo);

			glGenBuffers(1, &m_vbo);
			glBindBuffer(GL_ARRAY_BUFFER, m_vbo);
			glBufferData(GL_ARRAY_BUFFER, static_cast<GLsizeiptr>(vertices.size()) * sizeof(glm::vec3), vertices.data(), GL_STATIC_DRAW);
			glBindBuffer(GL_ARRAY_BUFFER, 0);

			north_pole.gl_generate(indices_north_pole);
			south_pole.gl_generate(indices_south_pole);
			sphere_body.gl_generate(indices_body);

			m_is_created = true;
			m_lighting_mode = with_lighting;
		}

		void Sphere::draw(uint32_t number_of_draws, bool with_lighting)
		{
			if (!m_is_created)
			{ /*create a new cuboid if it hasn't been created jet
			 (no check for lighting necessary because the vbo is for both identically )*/
				create(with_lighting);
			}
			glPrimitiveRestartIndex(0xffff);
			glBindBuffer(GL_ARRAY_BUFFER, m_vbo);

			glEnableVertexAttribArray(0);
			glVertexAttribPointer(0, // attribute 0 (must match the layout in the shader).
				3, // size
				GL_FLOAT, // type
				GL_FALSE, // normalized?
				0, // stride
				nullptr // array buffer offset
			);

			if (with_lighting)
			{
				glEnableVertexAttribArray(1);
				// the Normals of a sphere are the normalized positions so simple use the positions again
				glVertexAttribPointer(1, // attribute 0 (must match the layout in the shader).
					3, // size
					GL_FLOAT, // type
					GL_FALSE, // normalized?
					0, // stride
					nullptr // array buffer offset
				);
			}
			else
			{
				glDisableVertexAttribArray(1);
			}

			ExitOnGLError("ERROR: Couldn't set the vertex attribute pointer.");
			///////////////////////////draw north pole////////////////////////////
			north_pole.gl_draw(GL_TRIANGLE_FAN, number_of_draws);

			///////////////////////////draw south pole////////////////////////////
			south_pole.gl_draw(GL_TRIANGLE_FAN, number_of_draws);

			///////////////////////////draw sphere body////////////////////////////
			glEnable(GL_PRIMITIVE_RESTART);

			sphere_body.gl_draw(GL_TRIANGLE_STRIP, number_of_draws);

			glDisable(GL_PRIMITIVE_RESTART);
		}
	}
}