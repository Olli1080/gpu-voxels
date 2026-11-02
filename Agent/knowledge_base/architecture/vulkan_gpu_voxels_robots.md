# DH robots in the façade (slice S7-3, 2026-11-02)

Status: **Tested, not Parity** (the CUDA behaviour is a reading of `GpuVoxels.h`, `GpuVoxels.cu`, `KinematicChain.cu`, `MetaPointCloud.cu`, `robot_interface.h` and the map classes; nothing ran under CUDA). Builds on [vulkan_gpu_voxels_facade.md](vulkan_gpu_voxels_facade.md) and [vulkan_gpu_voxels_files.md](vulkan_gpu_voxels_files.md); the chain itself is the S2 `KinematicChain` ([vulkan_meta_cloud_and_robot.md](vulkan_meta_cloud_and_robot.md)). Header `vulkan/include/gpu_voxels/vulkan/GpuVoxels.h`, code `vulkan/src/GpuVoxels.cpp`. Plan: [S7 checklist](../../Analysis/2026-11-02_s7_facade_checklist.md), rows 10, 11, 13, 14, 18, 19.

Not part of this slice: URDF robots (no parser, row 12), joint limits (the CUDA DH robot has none either: `getLowerJointLimits` logs "not implemented"), a robot interface class (needed only when the URDF robot arrives), deleting a robot (CUDA has no `delRobot`).

## Types
| Type | Contract |
|---|---|
| `KinematicChainSharedPtr` | what `get_robot` hands out: the S2 `KinematicChain` itself, shared. A handle must not outlive the `GpuVoxels` (the `Context`) that made it, as for maps |
| robots of an instance | `robots_` is declared after the owned context, so an owning instance destroys the robots (device buffers) before the context |

## Adding a robot
| Call | Contract |
|---|---|
| `add_robot(name, convention, link_names, dh_parameters, files, model_path)` | link `i` is `link_names[i]` with `dh_parameters[i]`. **As in CUDA a file belongs to the link whose name is the file string**: `files` are the model files of the links that have geometry and every entry must be a link name; links without a file only add their DH transformation. Files are read with `load_point_cloud(file, model_path, {})` (no shift, scaling 1) |
| `add_robot(name, convention, link_names, dh_parameters, clouds, cloud_names)` | the sub clouds of a `MetaPointCloud` are copied (downloaded once). `cloud_names[k]` is the link that sub cloud `k` belongs to (CUDA keeps these names inside its `MetaPointCloud`, the Vulkan one has none) |
| `get_robot(name)` | the robot, empty (and an error log) for an unknown name |

Order of the checks: a used name returns `false` first (error log, the first robot stays, as CUDA); then the context must exist (an owning instance before `initialize()`: `VulkanError` "before initialize()"); then the arguments (`VulkanError`): number of names = number of DH sets, no duplicate link name (`add_robot: duplicate link name 'x'`), `cloud_names.size() == clouds.num_clouds()`, every cloud name is a link name, clouds in link order; **then** the files are read. A file that cannot be read (missing, unknown extension, malformed) returns `false` with an error log and registers nothing; the name stays free.

The robot's sub clouds are numbered over the links that have a cloud, in link order (`KinematicChain::cloud_index`). That is what meanings and masks of the self collision insert refer to.

## Using a robot
| Call | Contract |
|---|---|
| `set_robot_configuration(name, joint_values)` | `KinematicChain::set_configuration`; names that are no joint are ignored (as CUDA) |
| `get_robot_configuration(name, joint_values)` | **extends** the map with every joint of the robot and overwrites existing keys (CUDA `getConfiguration`); other keys stay |
| `set_robot_base_transformation(name, matrix)` | the base (transform 0); every cloud moves with it |
| `get_robot_transformation(name, idx, matrix)` | 0 = base, `num_links()` = end effector. A larger index **throws** (CUDA: logs and returns the identity) |
| `update_robot_part(name, link, points)` | `update_link_cloud`: the link must have a cloud and the number of points must not change (`VulkanError`; CUDA logs inside `MetaPointCloud`); the next evaluation uses the new cloud |
| `insert_robot_into_map(robot, map, meaning, outside*)` | `map->insert_meta_point_cloud(transformed_clouds(), meaning)`, for every map type and with the meaning rules of the type (distance map and octrees: occupied only, ...). `false` for an unknown robot or map, `outside` untouched then |
| `insert_robot_into_map_self_coll_aware(robot, map, meanings = {}, masks = {})` | **bit map only**: `insert_meta_point_cloud_with_self_collision_check` with CUDA's defaults for empty arguments (meaning of sub cloud `k` = `eBVM_SWEPT_VOLUME_START + k`, every cloud collides with all but itself and `eBVM_COLLISION`). Returns the whole `SelfCollisionResult` (colliding, colliding meanings, points outside); `nullopt` and an error log for an unknown robot or map. Every other map type **throws** "the operation 'insert_robot_into_map_self_coll_aware' is not supported by the map type ..." and the map stays as it was |

All wrappers return `false` (and log) for an unknown robot, as CUDA.

## Differences from CUDA
| CUDA | Here |
|---|---|
| a file that is not a link name stays a static cloud that no joint moves (and is inserted untransformed) | throws; static geometry would need a link with an identity transformation |
| the cloud numbering follows the order of the files | follows the order of the links; files / cloud names must be in link order (a different order throws) |
| duplicate link names: the `m_links` map keeps one, the chain uses it for both | throws |
| number of names and DH sets differ: `exit(-1)` of the process | `VulkanError` |
| unreadable file: error log, a `MetaPointCloud` without clouds, `addRobot` returns **true** | `false`, nothing registered |
| `insertRobotIntoMapSelfCollAware` returns only "collided", so an unknown name and "no collision" are both `false`; other map types log "not supported" and return `true` | `optional<SelfCollisionResult>`; other types throw |
| `getRobotTransformation` with a bad index: identity | throws |
| `addRobot` has no base transformation argument | same (set it with `set_robot_base_transformation`) |

## Verification
- `tests/vulkan/test_gpu_voxels_robots.cpp`, 9 cases (`RobotFacadeTest`): both conventions and both ways of adding a robot against the **independent double precision forward kinematics** of `tests/reference` (transforms of every index, the world clouds, random DH parameters, a prismatic joint, a link without cloud, partial configurations, a base); the model files `hollie/arm_{0,1,2}_link.xyz` as link geometry (clouds equal to `read_xyz_file`); every rule of adding (used name, unreadable files, every refused argument with its message, checks before reading, robots without links or clouds); the owning instance destroying its robots before its context; `update_robot_part` (new cloud, after a move, refused updates change nothing); `insert_robot_into_map` for **all seven map types against the class used directly with a twin chain** (contents and outside counts, three bases including poses partly outside) and for the bit map against the host reference; the self collision aware insertion against the host reference (default and custom meanings / masks, 10 poses with and without collision, outside count) and against the class; the refusal for the six other types.
- Mutation runner `tools/mutation/mutate_s7c_robots.py` (51 mutations): **all 51 killed**, none equivalent (log `tools/mutation/logs/s7c_robots.log`). One gap was closed while writing it: the masks of the self collision insert were not passed on without notice because the neighbour-only masks gave the same result as the defaults on the sampled poses; a third run with masks that make nobody a partner (nothing may collide) now catches it. Two further gaps were closed before the run: the message of duplicate link names (the chain throws one too, with another text) and "more names than sub clouds", plus a key that already exists in the map given to `get_robot_configuration`
- **Not verified:** CUDA parity; URDF; a handle that outlives its `GpuVoxels` is forbidden, not detected; robots were only run with the default `ContextCreateInfo` of the owning instance.
