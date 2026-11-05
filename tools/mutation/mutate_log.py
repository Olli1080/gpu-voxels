# Hand-written mutation runner (log messages of the Vulkan backend, gpu_voxels/vulkan/Log.h and src/Log.cpp): one mutation at a time, build, run the tests, restore.
# Usage: python mutate_log.py [name prefixes, comma separated]
# Method: Agent/knowledge_base/mutation_testing.md. Run one runner at a time on build/verify-vk.
import subprocess, os, sys
root = os.environ.get('GVL_ROOT', os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))).replace(chr(92), '/') + '/'
CPP = root + 'packages/gpu_voxels/gpu_voxels/vulkan/src/Log.cpp'
HDR = root + 'packages/gpu_voxels/gpu_voxels/vulkan/include/gpu_voxels/vulkan/Log.h'
CTX = root + 'packages/gpu_voxels/gpu_voxels/vulkan/src/Context.cpp'
env = dict(os.environ, VULKAN_SDK='C:\\VulkanSDK\\1.4.363.0')
FILTER = 'Log.*:VulkanTest.BitMapReports*:VulkanTest.FacadeReports*:VulkanTest.ContextReports*'

MUT = [
    ('L1 level compared with < instead of <=', CPP, "return static_cast<int>(level) <= g_level.load();", "return static_cast<int>(level) < g_level.load();"),
    ('L2 default level Warning', CPP, "std::atomic<int> g_level{static_cast<int>(LogLevel::Info)};", "std::atomic<int> g_level{static_cast<int>(LogLevel::Warning)};"),
    ('L3 log_message does not filter', CPP, "if (!log_enabled(level))\n\t\t\treturn;\n\n\t\tstd::shared_ptr<const LogHandler> handler;", "std::shared_ptr<const LogHandler> handler;"),
    ('L4 handler never used', CPP, "if (handler)\n\t\t{\n\t\t\t(*handler)(level, source, message);", "if (false)\n\t\t{\n\t\t\t(*handler)(level, source, message);"),
    ('L5 empty handler stored', CPP, "if (handler)\n\t\t\tnext = std::make_shared<const LogHandler>(std::move(handler));", "next = std::make_shared<const LogHandler>(std::move(handler));"),
    ('L6 source and text swapped', CPP, "(*handler)(level, source, message);", "(*handler)(level, message, source);"),
    ('L7 default output format', CPP, "\"[gvl] %s %.*s: %.*s\\n\"", "\"[gvl] %s %.*s - %.*s\\n\""),
    ('L8 level name', CPP, "case LogLevel::Warning: return \"Warning\";", "case LogLevel::Warning: return \"Warn\";"),
    ('L9 handler set without lock', CPP, "std::lock_guard lock(g_handler_mutex);\n\t\tg_handler = std::move(next);", "g_handler = std::move(next);"),
    ('L10 level never stored', CPP, "g_level.store(static_cast<int>(level));", ""),
    ('L11 level order changed', HDR, "Warning = 1,\n\t\tInfo = 2", "Warning = 2,\n\t\tInfo = 1"),
    ('L12 text always built', HDR, "if (::gpu_voxels::vulkan::log_enabled(level)) \\\n", "if (true) \\\n"),
    ('L13 source without the function', HDR, "std::string(#class_name \"::\") + __func__", "std::string(#class_name \"::\")"),
    ('L14 error macro logs a warning', HDR, "GVL_LOG(::gpu_voxels::vulkan::LogLevel::Error, class_name, stream_expression)", "GVL_LOG(::gpu_voxels::vulkan::LogLevel::Warning, class_name, stream_expression)"),
    ('L15 device line text', CTX, "GVL_LOG_INFO(Context, \"Vulkan device '\"", "GVL_LOG_INFO(Context, \"Device '\""),
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
            r = subprocess.run([root + 'build/verify-vk/tests/vulkan/Debug/test_vulkan.exe', '--gtest_fail_fast', '--gtest_filter=' + FILTER],
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
