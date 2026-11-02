# architecture — Index

Description of the **current** (CUDA, `master`) code base. Nothing here is about Vulkan; for that see [../vulkan_migration/](../vulkan_migration/summary.md).

| File | Content |
|---|---|
| [library_modules.md](library_modules.md) | Module/target map, who links whom, what is actually built on `master` |
| [data_model.md](data_model.md) | Map/list/octree types, voxel types, BitVector, enums, `GpuVoxels` façade API |
| [cuda_coupling_inventory.md](cuda_coupling_inventory.md) | Measured counts of CUDA/Thrust usage per area, kernel inventory, hard-to-port constructs |
| [build_system.md](build_system.md) | CMake layout, presets, vcpkg manifest and overlay ports, legacy ICMAKER leftovers |
| [visualization_and_ipc.md](visualization_and_ipc.md) | Visualizer process, shared-memory + CUDA-IPC protocol, GL interop |
| [vulkan_bit_voxel_map.md](vulkan_bit_voxel_map.md) | Slice S1: Vulkan `BitVoxelMap` (insert, clear, collide): scope, deviations from CUDA, verification incl. mutation testing |
| [vulkan_meta_cloud_and_robot.md](vulkan_meta_cloud_and_robot.md) | Slice S2: `MetaPointCloud` (GPU transform), meta cloud insertion, self collision, DH `KinematicChain`; CUDA quirks found; verification |
| [vulkan_octree_sensor.md](vulkan_octree_sensor.md) | Slice S6f (sensor pipeline, point cloud insert with free space): S6f-1 `Octree::voxelize_points` (pose, voxel grid, distinct voxels with counts and sensor-model occupancy), CUDA differences, verification |
| [vulkan_octree_facade.md](vulkan_octree_facade.md) | Slice S6e: `Octree` voxel side length, `insert_point_cloud` / `insert_coordinate_list` / `insert_meta_point_cloud` (meanings, GPU conversion), `OctreeStorage` interface and the native file format; CUDA quirks and stubs; verification |
| [vulkan_octree_dynamic.md](vulkan_octree_dynamic.md) | Slices S6d-2 .. S6d-3b: `Octree::insert` / `insert_nodes` / `propagate` (both kinds, all levels, hard and soft mode), chunk lists, garbage, algorithm, verification against the dynamic reference |
| [vulkan_octree_collision.md](vulkan_octree_collision.md) | Slice S6c: `Octree` collisions with bit maps, probabilistic maps, bit lists, Morton ids and other trees, collision flags; CUDA quirks found; wavefront instead of the load balancer; verification |
| [vulkan_octree_build.md](vulkan_octree_build.md) | Slice S6b: `Octree` (deterministic and probabilistic): native node formats with 48-bit device-address child pointers, bottom-up GPU build, `find`, wavefront `extract`, host `check_tree` / `dump`; deviations from CUDA; verification |
| [vulkan_octree_foundation.md](vulkan_octree_foundation.md) | Slice S6a: buffer device addresses in the runtime, Morton codes (host + GLSL), the independent CPU octree reference, measured behaviour of bad device addresses; verification |
| [octree_cuda_design.md](octree_cuda_design.md) | The CUDA octree (`NTree`/`GvlNTree`): size, node layout and status rules, memory management, operations, load balancer, constants (read, not run) |
| [vulkan_distance_map.md](vulkan_distance_map.md) | Slice S5: `DistanceVoxelMap` (nearest obstacle per voxel), exact separable and Parallel Banding algorithms, queries, `extract_distances`; CUDA PBA bugs found; verification |
| [vulkan_prob_voxel_map.md](vulkan_prob_voxel_map.md) | Slice S3: `ProbVoxelMap` (int8 log-odds), update modes (Atomic / Accumulate / Sequential), sensor data with ray casting, robot cut, collisions with bit/prob maps; CUDA quirks found; verification |
| [vulkan_voxel_lists.md](vulkan_voxel_lists.md) | Slice S4: `BitVoxelList` / `CountingVoxelList` (insert, make_unique, remove_underpopulated, collide, subtract), list->map merge, map->list extraction; CUDA quirks found; verification |
| [vulkan_primitives.md](vulkan_primitives.md) | Slice P2: exclusive scan, stable radix sort of pairs, iota (Thrust replacement); limits; verification |
| [vulkan_runtime.md](vulkan_runtime.md) | The new Vulkan backend runtime (`gpu_voxels_vulkan`): API, design choices, tests, verified results |
| [testing.md](testing.md) | Test infrastructure: `gvl_add_test`, categories, how to run, verified results |
