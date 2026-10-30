// Push constants shared by the event kernels (prob_points_apply, prob_events_count, prob_events_write).
// Must match struct EventsPush in ProbVoxelMap.cpp.
#include "prob_common.glsl"

layout(push_constant) uniform Push
{
    uint dim_x; uint dim_y; uint dim_z;
    uint first;          // index of the first point (or coordinate triple) handled by this dispatch
    uint count;          // number of points handled by this dispatch
    uint input_kind;     // 0: float xyz triples, 1: uint voxel coordinate triples
    float side;
    float origin_x; float origin_y; float origin_z; // sensor origin
    uint raycast;        // 1: decrease the occupancy along the ray from the origin to the point
    uint require_origin; // 1: measurements are dropped when the origin lies outside the map (sensor data)
    uint cut_robot;      // 1: drop measurements whose voxel has bit `robot_bit` in the robot map
    uint robot_bit;
    int free_inc;        // increment of the voxels along the ray
    int hit_inc;         // increment of the measured voxel
    uint mode;           // prob_points_apply: 0 = atomic update, 1 = accumulate
    uint assign_unknown; // prob_points_apply: set the voxel to unknown instead of adding
    uint base;           // prob_events_write: event index of the first event of this chunk
    uint batch_first;    // prob_events_write: `first` of the whole batch (offsets are relative to it)
} pc;
