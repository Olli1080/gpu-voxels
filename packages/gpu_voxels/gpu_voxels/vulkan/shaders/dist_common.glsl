// Shared by the kernels of the distance voxel map (DistanceVoxelMap).
// A voxel is one 32-bit word with the position of its nearest obstacle packed as x | y << 10 | z << 20 (CUDA DistanceVoxel);
// all three coordinates 1023 mean "uninitialised" (no obstacle known).
#include "list_entry.glsl" // voxel_of_point(), voxel_id(), coords_of_id()

const uint DIST_UNINIT = 0x3FFFFFFFu;
const int DIST_INF = 1 << 26;            // cost of "no obstacle"; far above any real squared distance (< 3 * 1023^2)
const int DIST_MAX_OBSTACLE_DISTANCE = 2147483647;

uint dist_pack(uvec3 c) { return c.x | (c.y << 10u) | (c.z << 20u); }
uvec3 dist_unpack(uint w) { return uvec3(w & 1023u, (w >> 10u) & 1023u, w >> 20u); }

// DistanceVoxel::squaredObstacleDistance for the voxel at `position`
int dist_squared_distance(uint word, uvec3 position)
{
    const uvec3 o = dist_unpack(word);
    if (o.x == 1023u || o.y == 1023u || o.z == 1023u)
        return DIST_MAX_OBSTACLE_DISTANCE;
    const ivec3 d = ivec3(position) - ivec3(o);
    return d.x * d.x + d.y * d.y + d.z * d.z;
}
