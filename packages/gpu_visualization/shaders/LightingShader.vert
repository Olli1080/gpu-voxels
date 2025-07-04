// !!! DO NOT EDIT THIS FILE - IT'S ONLY A CONVENCIENCE COPY OF THE CORRESPONDING *Shader.h FILE !!!

#version 410 core

// Input vertex data, different for all executions of this shader.
layout(location = 0) in vec3 vertexPosition_modelspace;
layout(location = 1) in vec3 vertexNormal_modelspace;
layout(location = 2) in vec4 vtranslation_and_scale;//(x,y,z) contain the translation and (w) the scal factor

// Ouput data
out vec4 fragmentColor;
out vec3 normal_cameraspace;
out vec3 position_cameraspace;


// Values that stay constant for the whole mesh.
uniform mat4 VP; // View-Projektion-Matrix
uniform mat4 V;  // View-Matrix
uniform mat4 V_inverse_transpose; // inverse-transpose View-Matrix


uniform vec4 startColor;
uniform vec4 endColor;
// if interpolation is false, no color interpolation takes place. The startColor will be used.
uniform bool interpolation;
uniform vec3 interpolationLength;
uniform vec3 translationOffset;

void main(){
  gl_PointSize = 6.f;
  vec3 v_translation = vtranslation_and_scale.xyz + translationOffset;
  float scale = vtranslation_and_scale.w;

  mat4 M = mat4(vec4(scale,0,0,0),
                vec4(0,scale,0,0),
                vec4(0,0,scale,0),
                vec4(v_translation,1));

  mat4 M_inverse_transpose = mat4(1/scale,0,0,-v_translation.x/scale,
                                  0,1/scale,0,-v_translation.y/scale,
                                  0,0,1/scale,-v_translation.z/scale,
                                  0,0,0,                1);

  ////////////// apply vertex and normal transformations /////////////////////////////////
  gl_Position = VP * M * vec4(vertexPosition_modelspace, 1.0f);

  position_cameraspace = vec3(V * M * vec4(vertexPosition_modelspace,1));
  normal_cameraspace = vec3(V_inverse_transpose * M_inverse_transpose * vec4(vertexNormal_modelspace,0));

  ///////caluclate the color of the vertex /////////////////////////////////////
  if (interpolation) {
    float vertexPos_world_z = v_translation.z + vertexPosition_modelspace.z;
    float a = (mod(vertexPos_world_z, interpolationLength.z)) / interpolationLength.z;
    a = a > 0.5f ? -2*a+2 : 2*a;
    fragmentColor = (a * startColor + (1 - a) * endColor);
  }
  else {
    fragmentColor = startColor;
  }
}
