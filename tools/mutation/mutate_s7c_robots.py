# Hand-written mutation runner (slice S7-3, robots in the façade: GpuVoxels::add_robot, get_robot, the wrappers, insert_robot_into_map, insert_robot_into_map_self_coll_aware): one mutation at a time, build, run the tests, restore.
# Usage: python mutate_s7c_robots.py [name prefixes, comma separated]
# Method: Agent/knowledge_base/mutation_testing.md. Run one runner at a time on build/verify-vk.
import subprocess, os, sys
root = os.environ.get('GVL_ROOT', os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))).replace(chr(92), '/') + '/'
GV = root + 'packages/gpu_voxels/gpu_voxels/vulkan/src/GpuVoxels.cpp'
env = dict(os.environ, VULKAN_SDK='C:\\VulkanSDK\\1.4.363.0')

FILES_HEAD = "std::span<const std::string> files, const std::filesystem::path& model_path)\n\t{\n\t\tif (robots_.contains(robot_name))"
META_HEAD = "const MetaPointCloud& clouds, std::span<const std::string> cloud_names)\n\t{\n\t\tif (robots_.contains(robot_name))"
LOAD = "clouds.push_back(load_point_cloud(file, model_path, PointCloudLoadOptions{}));"
EMPLACE = "\n\t\trobots_.emplace(robot_name, std::make_shared<KinematicChain>(ctx, convention, links));"
UNKNOWN = "if (!robot)\n\t\t\treturn false;\n\t\t"
SELF = "bit_map->insert_meta_point_cloud_with_self_collision_check(robot->transformed_clouds(), voxel_meanings, collision_masks);"

MUT = [
    ('H1 sizes of names and parameters not compared', GV, "if (link_names.size() != num_dh_parameters)", "if (false)"),
    ('H2 more parameters than names accepted', GV, "if (link_names.size() != num_dh_parameters)", "if (link_names.size() > num_dh_parameters)"),
    ('H3 duplicate link names not refused here (the chain refuses them with another message)', GV, "if (link_names[i] == link_names[j])", "if (false)"),
    ('H4 every link is its own duplicate', GV, "std::size_t j = i + 1; j < link_names.size()", "std::size_t j = i; j < link_names.size()"),
    ('H5 cloud of an unknown link accepted', GV, "if (it == link_names.end())", "if (false)"),
    ('H6 two clouds for one link accepted', GV, "link <= links.back()", "link < links.back()"),
    ('H7 clouds in any order', GV, "if (!links.empty() && link <= links.back())", "if (false)"),
    ('H8 cloud k goes to link k', GV, "links[link_of_cloud[k]].cloud = std::move(clouds[k]);", "links[k].cloud = std::move(clouds[k]);"),
    ('H9 DH parameters of link 0 for every link', GV, "links.push_back({ link_names[i], dh_parameters[i], std::nullopt });", "links.push_back({ link_names[i], dh_parameters[0], std::nullopt });"),
    ('F1 files: used robot name accepted', GV, FILES_HEAD, FILES_HEAD.replace("if (robots_.contains(robot_name))", "if (false)")),
    ('F2 meta cloud: used robot name accepted', GV, META_HEAD, META_HEAD.replace("if (robots_.contains(robot_name))", "if (false)")),
    ('F3 files: the arguments are not checked', GV, "links_of_clouds(link_names, dh_parameters.size(), files)", "links_of_clouds(link_names, dh_parameters.size(), std::span<const std::string>{})"),
    ('F4 files: shifted to zero', GV, LOAD, "clouds.push_back(load_point_cloud(file, model_path, PointCloudLoadOptions{ true, Point3f{ 0.0f, 0.0f, 0.0f }, 1.0f }));"),
    ('F5 files: scaled', GV, LOAD, "clouds.push_back(load_point_cloud(file, model_path, PointCloudLoadOptions{ false, Point3f{ 0.0f, 0.0f, 0.0f }, 2.0f }));"),
    ('F6 files: model path ignored', GV, "clouds.push_back(load_point_cloud(file, model_path, ", "clouds.push_back(load_point_cloud(file, std::filesystem::path{}, "),
    ('F7 files: file errors not caught', GV, LOAD + "\n\t\t}\n\t\tcatch (const PointCloudFileError& e)", LOAD + "\n\t\t}\n\t\tcatch (const std::bad_alloc& e)"),
    ('F8 files: file error reported as success', GV, "e.what() << endl);\n\t\t\treturn false;\n\t\t}\n\t\tconst std::vector<LinkSpec> links", "e.what() << endl);\n\t\t\treturn true;\n\t\t}\n\t\tconst std::vector<LinkSpec> links"),
    ('F9 files: convention ignored', GV, "link_of_cloud, clouds);" + EMPLACE, "link_of_cloud, clouds);" + EMPLACE.replace("convention", "DHConvention::Classic")),
    ('F10 meta cloud: convention ignored', GV, "link_of_cloud, points);" + EMPLACE, "link_of_cloud, points);" + EMPLACE.replace("convention", "DHConvention::Classic")),
    ('F11 meta cloud: names not counted', GV, "if (cloud_names.size() != clouds.num_clouds())", "if (false)"),
    ('F12 meta cloud: fewer names than clouds accepted', GV, "if (cloud_names.size() != clouds.num_clouds())", "if (cloud_names.size() > clouds.num_clouds())"),
    ('F13 meta cloud: more names than clouds accepted', GV, "if (cloud_names.size() != clouds.num_clouds())", "if (cloud_names.size() < clouds.num_clouds())"),
    ('F14 meta cloud: the first sub cloud for every link', GV, "points.push_back(clouds.download_cloud(k));", "points.push_back(clouds.download_cloud(0));"),
    ('F15 meta cloud: the last sub cloud is lost', GV, "for (uint32_t k = 0; k < clouds.num_clouds(); ++k)", "for (uint32_t k = 0; k + 1 < clouds.num_clouds(); ++k)"),
    ('W1 configuration not passed on', GV, "robot->set_configuration(joint_values);", "(void)joint_values;"),
    ('W2 get configuration keeps existing keys', GV, "joint_values[name] = value;", "joint_values.emplace(name, value);"),
    ('W3 get configuration clears the map first', GV, "if (!robot)\n\t\t\treturn false;\n\t\tfor (const auto& [name, value] : robot->configuration())", "if (!robot)\n\t\t\treturn false;\n\t\tjoint_values.clear();\n\t\tfor (const auto& [name, value] : robot->configuration())"),
    ('W4 base transformation not passed on', GV, "robot->set_base_transformation(transformation);", "(void)transformation;"),
    ('W5 transformation 0 for every index', GV, "transformation = robot->transform(idx);", "transformation = robot->transform(0);"),
    ('W6 transformation not stored', GV, "transformation = robot->transform(idx);", "(void)robot->transform(idx);"),
    ('W7 robot part not passed on', GV, "robot->update_link_cloud(link_name, points);", "(void)points;"),
    ('W8 set configuration: unknown robot is fine', GV, UNKNOWN + "robot->set_configuration", UNKNOWN.replace("false", "true") + "robot->set_configuration"),
    ('W9 get configuration: unknown robot is fine', GV, UNKNOWN + "for (const auto& [name, value]", UNKNOWN.replace("false", "true") + "for (const auto& [name, value]"),
    ('W10 set base: unknown robot is fine', GV, UNKNOWN + "robot->set_base_transformation", UNKNOWN.replace("false", "true") + "robot->set_base_transformation"),
    ('W11 get transformation: unknown robot is fine', GV, UNKNOWN + "transformation = robot->transform", UNKNOWN.replace("false", "true") + "transformation = robot->transform"),
    ('W12 update part: unknown robot is fine', GV, UNKNOWN + "robot->update_link_cloud", UNKNOWN.replace("false", "true") + "robot->update_link_cloud"),
    ('W13 insert robot: unknown robot is fine', GV, UNKNOWN + "const GpuVoxelsMapSharedPtr map = get_map(map_name);", UNKNOWN.replace("false", "true") + "const GpuVoxelsMapSharedPtr map = get_map(map_name);"),
    ('I1 insert robot: meaning changed', GV, "map->insert_meta_point_cloud(robot->transformed_clouds(), voxel_meaning);", "map->insert_meta_point_cloud(robot->transformed_clouds(), uint8_t(voxel_meaning + 1));"),
    ('I2 insert robot: the clouds before the transformation', GV, "map->insert_meta_point_cloud(robot->transformed_clouds(), voxel_meaning);", "map->insert_meta_point_cloud(robot->link_clouds(), voxel_meaning);"),
    ('I3 insert robot: counter not stored', GV, "robot->transformed_clouds(), voxel_meaning);\n\t\tif (outside != nullptr)\n\t\t\t*outside = dropped;", "robot->transformed_clouds(), voxel_meaning);\n\t\tif (outside != nullptr)\n\t\t\t*outside = 0;"),
    ('I4 insert robot: unknown map is fine', GV, "if (!map)\n\t\t\treturn false;\n\t\tconst uint32_t dropped = map->insert_meta_point_cloud(robot->", "if (!map)\n\t\t\treturn true;\n\t\tconst uint32_t dropped = map->insert_meta_point_cloud(robot->"),
    ('S1 self collision: meanings and masks dropped', GV, SELF, "bit_map->insert_meta_point_cloud_with_self_collision_check(robot->transformed_clouds(), {}, {});"),
    ('S2 self collision: masks dropped', GV, SELF, "bit_map->insert_meta_point_cloud_with_self_collision_check(robot->transformed_clouds(), voxel_meanings, {});"),
    ('S3 self collision: meanings dropped', GV, SELF, "bit_map->insert_meta_point_cloud_with_self_collision_check(robot->transformed_clouds(), {}, collision_masks);"),
    ('S4 self collision: other maps are not refused', GV, "if (bit_map == nullptr)\n\t\t\tunsupported(", "if (false)\n\t\t\tunsupported("),
    ('S5 self collision: the refusal names another operation', GV, "unsupported(map->type(), \"insert_robot_into_map_self_coll_aware\")", "unsupported(map->type(), \"insert_robot_into_map\")"),
    ('S6 self collision: unknown robot gives a result', GV, "if (!robot)\n\t\t\treturn std::nullopt;", "if (!robot)\n\t\t\treturn BitVoxelMap::SelfCollisionResult{};"),
    ('S7 self collision: unknown map gives a result', GV, "if (!map)\n\t\t\treturn std::nullopt;", "if (!map)\n\t\t\treturn BitVoxelMap::SelfCollisionResult{};"),
    ('D1 the context is destroyed before the robots', GV, "GpuVoxels::~GpuVoxels() = default;", "GpuVoxels::~GpuVoxels() { owned_context_.reset(); }"),
]

only = tuple(sys.argv[1].split(',')) if len(sys.argv) > 1 and sys.argv[1] else ()
originals = {}


def read(path):
    if path not in originals:
        originals[path] = open(path, encoding='utf-8', newline='').read()
    return originals[path]


try:
    for name, path, old, new in MUT:
        if only and not name.startswith(only):
            continue
        orig = read(path)
        nl = '\r\n' if '\r\n' in orig else '\n'
        text = orig.replace('\r\n', '\n')
        if text.count(old) != 1:
            print(name, ': PATTERN COUNT', text.count(old), flush=True)
            continue
        open(path, 'w', encoding='utf-8', newline='').write(text.replace(old, new).replace('\n', nl))
        b = subprocess.run(['cmake', '--build', root + 'build/verify-vk', '--config', 'Debug', '--target', 'test_vulkan', '--parallel'],
                           env=env, capture_output=True, text=True)
        if b.returncode != 0:
            print(name, ': BUILD FAILED', flush=True)
            open(path, 'w', encoding='utf-8', newline='').write(orig)
            continue
        try:
            r = subprocess.run([root + 'build/verify-vk/tests/vulkan/Debug/test_vulkan.exe', '--gtest_fail_fast', '--gtest_filter=RobotFacadeTest.*'],
                               env=env, capture_output=True, text=True, timeout=300)
            verdict = 'SURVIVED' if r.returncode == 0 else 'KILLED'
        except subprocess.TimeoutExpired:
            verdict = 'KILLED (timeout)'
        print(name, ':', verdict, flush=True)
        open(path, 'w', encoding='utf-8', newline='').write(orig)
finally:
    for path, orig in originals.items():
        open(path, 'w', encoding='utf-8', newline='').write(orig)
    print('restored', flush=True)
