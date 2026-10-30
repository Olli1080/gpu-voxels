// Event generation of the probabilistic map: for one measurement the voxel updates in CUDA order
// (kernelInsertSensorData): free-space updates along the ray from the sensor voxel to the measured voxel, then the
// measured voxel itself. The including shader (after prob_push.glsl) defines
//   void prob_emit(uint voxel_id, int increment);  // one update
//   void prob_skip_outside();                      // measurement dropped: outside the map / NaN
//   void prob_skip_cut();                          // measurement dropped: robot voxel
// and the three kernels share this code, so counting, writing and applying see the same events in the same order.
layout(std430, set = 0, binding = 0) readonly buffer Raw { uint raw[]; };
layout(std430, set = 0, binding = 1) readonly buffer Robot { uint robot_vox[]; };

bool prob_target(uint i, uvec3 dim, out uvec3 hit)
{
    if (pc.input_kind == 0u)
    {
        const vec3 p = vec3(uintBitsToFloat(raw[3u * i]), uintBitsToFloat(raw[3u * i + 1u]), uintBitsToFloat(raw[3u * i + 2u]));
        return voxel_of_point(p, pc.side, dim, hit);
    }
    hit = uvec3(raw[3u * i], raw[3u * i + 1u], raw[3u * i + 2u]);
    return hit.x < dim.x && hit.y < dim.y && hit.z < dim.z;
}

void prob_process_point(uint i)
{
    const uvec3 dim = uvec3(pc.dim_x, pc.dim_y, pc.dim_z);
    uvec3 hit;
    if (!prob_target(i, dim, hit))
    {
        prob_skip_outside();
        return;
    }
    uvec3 origin = uvec3(0u);
    if (pc.require_origin != 0u && !voxel_of_point(vec3(pc.origin_x, pc.origin_y, pc.origin_z), pc.side, dim, origin))
    {
        prob_skip_outside();
        return;
    }
    if (pc.cut_robot != 0u)
    {
        const uint word = voxel_id(hit, dim) * WORDS_PER_VOXEL + (pc.robot_bit >> 5u);
        if (((robot_vox[word] >> (pc.robot_bit & 31u)) & 1u) != 0u)
        {
            prob_skip_cut();
            return;
        }
    }

    if (pc.raycast != 0u)
    {
        // CUDA RayCaster::rayCast: 6-connected walk, visits 1 + dx + dy + dz voxels including both end points
        const ivec3 target = ivec3(hit);
        ivec3 p = ivec3(origin);
        const ivec3 d = abs(target - p);
        const ivec3 step = ivec3(target.x > p.x ? 1 : -1, target.y > p.y ? 1 : -1, target.z > p.z ? 1 : -1);
        int n = 1 + d.x + d.y + d.z;
        int error_xy = d.x - d.y;
        int error_xz = d.x - d.z;
        int error_yz = d.y - d.z;
        const ivec3 dd = 2 * d;
        for (; n > 0; --n)
        {
            // the walk never leaves the box spanned by its end points (checked exhaustively in the tests); the guard
            // only protects the buffer against a violation of that property
            if (all(greaterThanEqual(p, ivec3(0))) && all(lessThan(p, ivec3(dim))))
                prob_emit(voxel_id(uvec3(p), dim), pc.free_inc);
            if (error_xy > 0 && error_xz > 0)
            {
                p.x += step.x;
                error_xy -= dd.y;
                error_xz -= dd.z;
            }
            else if (error_yz > 0)
            {
                p.y += step.y;
                error_xy += dd.x;
                error_yz -= dd.z;
            }
            else
            {
                p.z += step.z;
                error_xz += dd.x;
                error_yz += dd.y;
            }
        }
    }
    prob_emit(voxel_id(hit, dim), pc.hit_inc);
}
