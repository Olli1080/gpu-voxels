#version 450
// Draws the cubes of a CubeBuffer: 36 vertices per instance, the instance record is read through its buffer device address (push constant), so that
// no vertex buffer and no descriptor set is needed. Faces are drawn without culling (the depth test hides the back faces).
#extension GL_EXT_buffer_reference : require
#extension GL_EXT_shader_explicit_arithmetic_types_int64 : require
#extension GL_GOOGLE_include_directive : enable
#include "viewer_cube_common.glsl"

layout(buffer_reference, std430, buffer_reference_align = 16) readonly buffer Instances { CubeInstance items[]; };
layout(push_constant) uniform Push
{
    mat4 view_proj;
    uint64_t instances;
    uint shaded;
    uint pad;
} pc;

layout(location = 0) out vec4 v_colour;
layout(location = 1) out vec3 v_normal;

// two triangles of a face: the corners 0, 1, 2 and 2, 1, 3 of a quad whose corner q has the coordinates (q & 1, q >> 1)
const uint QUAD[6] = uint[6](0u, 1u, 2u, 2u, 1u, 3u);

void main()
{
    const CubeInstance inst = Instances(pc.instances).items[gl_InstanceIndex];
    const uint face = uint(gl_VertexIndex) / 6u;          // 0 .. 5
    const uint corner = QUAD[uint(gl_VertexIndex) % 6u];
    const uint axis = face >> 1u;                         // the axis the face is perpendicular to
    const float sign_ = (face & 1u) != 0u ? 1.0 : -1.0;
    vec3 local;
    local[axis] = sign_ > 0.0 ? 1.0 : 0.0;
    local[(axis + 1u) % 3u] = float(corner & 1u);
    local[(axis + 2u) % 3u] = float(corner >> 1u);
    vec3 normal = vec3(0.0);
    normal[axis] = sign_;

    gl_Position = pc.view_proj * vec4(inst.pos_size.xyz + local * inst.pos_size.w, 1.0);
    v_colour = unpackUnorm4x8(inst.rgba);
    v_normal = normal;
}
