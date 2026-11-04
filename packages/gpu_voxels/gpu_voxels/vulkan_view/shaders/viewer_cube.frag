#version 450
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
layout(location = 0) out vec4 out_colour;

void main()
{
    if (pc.shaded == 0u)
    {
        out_colour = vec4(v_colour.rgb, 1.0); // flat colour: the exact palette colour (tests)
        return;
    }
    // fixed light from above-left-front in world coordinates: ambient 0.35 + diffuse 0.65 (the three visible faces of a cube get different shades)
    const vec3 light = normalize(vec3(-0.4, 0.8, 0.45));
    const float diffuse = max(dot(normalize(v_normal), light), 0.0);
    out_colour = vec4(v_colour.rgb * (0.35 + 0.65 * diffuse), 1.0);
}
