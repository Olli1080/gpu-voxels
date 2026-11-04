#version 450
// Spheres as impostors: the cube around the sphere is rasterised (viewer_cube.vert), every fragment casts the ray from the eye through its position on
// the cube and intersects it with the inscribed sphere; no hit = discard. The depth of the hit point is written, so spheres and cubes hide each other correctly.
#extension GL_EXT_shader_explicit_arithmetic_types_int64 : require
layout(push_constant) uniform Push
{
    mat4 view_proj;
    uint64_t instances;
    uint shaded;
    uint pad;
    vec4 eye;
} pc;

layout(location = 0) in vec4 v_colour;
layout(location = 1) in vec3 v_normal;
layout(location = 2) in vec3 v_world;
layout(location = 3) flat in vec4 v_sphere;
layout(location = 0) out vec4 out_colour;

void main()
{
    const vec3 direction = normalize(v_world - pc.eye.xyz);
    const vec3 to_eye = pc.eye.xyz - v_sphere.xyz;
    const float b = dot(to_eye, direction);
    const float discriminant = b * b - (dot(to_eye, to_eye) - v_sphere.w * v_sphere.w);
    if (discriminant < 0.0)
        discard;
    const float t = -b - sqrt(discriminant);
    if (t < 0.0)
        discard;     // the eye is inside the sphere
    const vec3 hit = pc.eye.xyz + direction * t;
    const vec3 normal = normalize(hit - v_sphere.xyz);
    const vec4 clip = pc.view_proj * vec4(hit, 1.0);
    gl_FragDepth = clip.z / clip.w;
    if (pc.shaded == 0u)
    {
        out_colour = vec4(v_colour.rgb, 1.0);
        return;
    }
    const vec3 light = normalize(vec3(-0.4, 0.8, 0.45));
    out_colour = vec4(v_colour.rgb * (0.35 + 0.65 * max(dot(normal, light), 0.0)), 1.0);
}
