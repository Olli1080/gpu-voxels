// this is for emacs file handling -*- mode: c++; indent-tabs-mode: nil -*-

// -- BEGIN LICENSE BLOCK ----------------------------------------------
// This file is from: http://www.opengl-tutorial.org/
// -- END LICENSE BLOCK ------------------------------------------------

//----------------------------------------------------------------------
/*!\file
 *
 * \author  Matthias Wagner
 * \date    2013-12-17
 *
 *  \brief Loads and compiles the fragment and vertex shader.
 *              from: http://www.opengl-tutorial.org/
 */
 //----------------------------------------------------------------------
#include <GL/glew.h>

#include <gpu_visualization/logging/logging_visualization.h>
#include "shader.h"

using namespace std;


namespace gpu_voxels {
	namespace visualization {
		GLuint loadShaders(const char* vertex_shader, const char* fragment_shader)
		{

			// Create the shaders
			const GLuint VertexShaderID = glCreateShader(GL_VERTEX_SHADER);
			const GLuint FragmentShaderID = glCreateShader(GL_FRAGMENT_SHADER);

			GLint Result = GL_FALSE;
			int32_t InfoLogLength;

			// Compile Vertex Shader
			LOGGING_DEBUG(Shader, "Compiling Vertex shader" << endl);
			glShaderSource(VertexShaderID, 1, &vertex_shader, nullptr);
			glCompileShader(VertexShaderID);

			// Check Vertex Shader
			glGetShaderiv(VertexShaderID, GL_COMPILE_STATUS, &Result);
			if (Result != 1) LOGGING_ERROR(Shader, "Vertex Shader Compile Result is: " << Result << endl);
			glGetShaderiv(VertexShaderID, GL_INFO_LOG_LENGTH, &InfoLogLength);
			if (InfoLogLength > 1)
			{
				std::vector<char> VertexShaderErrorMessage(InfoLogLength + 1);
				glGetShaderInfoLog(VertexShaderID, InfoLogLength, nullptr, VertexShaderErrorMessage.data());
				LOGGING_ERROR(Shader, VertexShaderErrorMessage.data() << endl);
			}

			// Compile Fragment Shader
			LOGGING_DEBUG(Shader, "Compiling Fragment shader" << endl);
			glShaderSource(FragmentShaderID, 1, &fragment_shader, nullptr);
			glCompileShader(FragmentShaderID);

			// Check Fragment Shader
			glGetShaderiv(FragmentShaderID, GL_COMPILE_STATUS, &Result);
			if (Result != 1) LOGGING_ERROR(Shader, "Frgament Shader Compile Result is: " << Result << endl);
			glGetShaderiv(FragmentShaderID, GL_INFO_LOG_LENGTH, &InfoLogLength);
			if (InfoLogLength > 1)
			{
				std::vector<char> FragmentShaderErrorMessage(InfoLogLength + 1);
				glGetShaderInfoLog(FragmentShaderID, InfoLogLength, nullptr, FragmentShaderErrorMessage.data());
				LOGGING_ERROR(Shader, FragmentShaderErrorMessage.data() << endl);
			}

			// Link the program
			LOGGING_DEBUG(Shader, "Linking program" << endl);
			const GLuint ProgramID = glCreateProgram();
			glAttachShader(ProgramID, VertexShaderID);
			glAttachShader(ProgramID, FragmentShaderID);
			glLinkProgram(ProgramID);

			// Check the program
			glGetProgramiv(ProgramID, GL_LINK_STATUS, &Result);
			glGetProgramiv(ProgramID, GL_INFO_LOG_LENGTH, &InfoLogLength);
			if (InfoLogLength > 1)
			{
				std::vector<char> ProgramErrorMessage(InfoLogLength + 1);
				glGetProgramInfoLog(ProgramID, InfoLogLength, nullptr, ProgramErrorMessage.data());
				LOGGING_ERROR(Shader, ProgramErrorMessage.data() << endl);
			}

			glDeleteShader(VertexShaderID);
			glDeleteShader(FragmentShaderID);

			return ProgramID;
		}

	} // end of namespace visualization
} // end of namespace gpu_voxels