// Linear invocation index of a 1D problem that may be dispatched as a 2D grid (the group count of one dimension is limited
// to 65535 on some devices): the host uses gx = min(groups, 65535) and gy = ceil(groups / gx).
uint oct_gid() { return gl_GlobalInvocationID.y * gl_NumWorkGroups.x * gl_WorkGroupSize.x + gl_GlobalInvocationID.x; }
