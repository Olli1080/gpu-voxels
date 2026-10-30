// Shared by the kernels of the probabilistic voxel map (ProbVoxelMap).
// A voxel is one 32-bit word holding the sign extended int8 occupancy (log-odds); -128 means "unknown".
#include "list_entry.glsl" // voxel_of_point(), voxel_id()

const uint PROB_UNKNOWN_WORD = 0xFFFFFF80u;
const int PROB_UNKNOWN = -128;
const int PROB_MIN = -127;
const int PROB_MAX = 127;

// CUDA ProbabilisticVoxel::updateOccupancy: the first update of an unknown voxel assigns the increment, later updates
// add with saturation.
int prob_apply(int old_value, int increment)
{
    return old_value == PROB_UNKNOWN ? increment : clamp(old_value + increment, PROB_MIN, PROB_MAX);
}
