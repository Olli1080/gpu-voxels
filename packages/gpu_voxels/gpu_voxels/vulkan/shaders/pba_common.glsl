// Parallel Banding Algorithm (PBA) kernels of the DistanceVoxelMap: GLSL port of the CUDA kernels in
// packages/gpu_voxels/gpu_voxels/voxelmap/include/gpu_voxels/voxelmap/kernels/VoxelMapOperationsPBA.hpp.
//
// The kernels are derived from the original PBA implementation, whose licence requires this notice to be retained:
/*
===============================================================================

Copyright (c) 2010, School of Computing, National University of Singapore.
All rights reserved.

Project homepage: http://www.comp.nus.edu.sg/~tants/pba.html

If you use PBA and you like it or have comments on its usefulness etc., we
would love to hear from you at <tants@comp.nus.edu.sg>. You may share with us
your experience and any possibilities that we may improve the work/code.

===============================================================================

Redistribution and use in source and binary forms, with or without modification,
are permitted provided that the following conditions are met:

Redistributions of source code must retain the above copyright notice, this list of
conditions and the following disclaimer. Redistributions in binary form must reproduce
the above copyright notice, this list of conditions and the following disclaimer
in the documentation and/or other materials provided with the distribution.

Neither the name of the National University of University nor the names of its contributors
may be used to endorse or promote products derived from this software without specific
prior written permission from the National University of Singapore.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY
EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO THE IMPLIED WARRANTIES
OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT
SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT,
INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED
TO, PROCUREMENT OF SUBSTITUTE  GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR
BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN
ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH
DAMAGE.

*/
// Layout: a voxel (x, y, z) of an nx * ny * nz volume has the linear index (z * ny + y) * nx + x and holds the packed
// position of an obstacle (see dist_common.glsl). Phase 1 floods obstacles along z, phases 2 and 3 compute the Voronoi
// diagram along y; the volume is transposed between the two passes, so "y" is the axis currently processed.
#include "dist_common.glsl"

const int PBA_U = 1023; // PBA_UNINITIALISED_COORD

ivec3 pba_load(uint word) { return ivec3(dist_unpack(word)); }
uint pba_store(ivec3 c) { return dist_pack(uvec3(c)); }

// y coordinate where the perpendicular bisector plane of the obstacles a = (x1, y1, z1) and b = (x2, y2, z2), y1 < y2,
// intersects the line (x0, *, z0) (CUDA devicePBAintersection3DY)
float pba_intersection(int x1, int y1, int z1, int x2, int y2, int z2, int x0, int z0)
{
    const float xm = float(x1 + x2) / 2.0;
    const float ym = float(y1 + y2) / 2.0;
    const float zm = float(z1 + z2) / 2.0;
    const float nx = float(x2 - x1);
    const float ny = float(y2 - y1);
    const float nz = float(z2 - z1);
    return ym + (nx * (xm - float(x0)) + nz * (zm - float(z0))) / ny;
}
