# Offline shader compilation (GLSL -> SPIR-V). Included from the root CMakeLists.txt when GVL_BACKEND_VULKAN is ON.
# glslangValidator comes from the Vulkan SDK (found in FindDependencies.cmake).
#
#   gvl_add_shaders(<target> OUTPUT_DIR <dir> SOURCES <shader.comp>... [INCLUDE_DIRS <dir>...])
#       Compiles every source into <dir>/<file name>.spv and makes <target> depend on the result.
#       Used by tests that load the .spv files at run time.
#
#   gvl_embed_shaders(<target> SOURCES <shader.comp>... [INCLUDE_DIRS <dir>...])
#       Compiles the sources and embeds the SPIR-V into <target> as generated C++ arrays:
#         namespace gpu_voxels::vulkan::shaders { extern const uint32_t <name>_comp[]; extern const std::size_t <name>_comp_words; }
#       with <name>_comp derived from the file name ("bit_voxel_collide.comp" -> bit_voxel_collide_comp).
#       The library therefore needs no shader files at run time.

set(GVL_EMBED_SPIRV_SCRIPT "${CMAKE_CURRENT_LIST_DIR}/EmbedSpirv.cmake")

function(gvl_add_shaders target)
  cmake_parse_arguments(ARG "" "OUTPUT_DIR" "SOURCES;INCLUDE_DIRS" ${ARGN})
  if(NOT GVL_GLSLANG_VALIDATOR)
    message(FATAL_ERROR "gvl_add_shaders: glslangValidator not found (install the Vulkan SDK and set VULKAN_SDK)")
  endif()
  if(NOT ARG_OUTPUT_DIR OR NOT ARG_SOURCES)
    message(FATAL_ERROR "gvl_add_shaders(${target}): OUTPUT_DIR and SOURCES are required")
  endif()

  set(include_flags "")
  foreach(dir IN LISTS ARG_INCLUDE_DIRS)
    list(APPEND include_flags "-I${dir}")
  endforeach()

  set(spv_files "")
  foreach(source IN LISTS ARG_SOURCES)
    get_filename_component(source_abs "${source}" ABSOLUTE)
    get_filename_component(source_name "${source}" NAME)
    set(spv "${ARG_OUTPUT_DIR}/${source_name}.spv")
    # Shader headers (*.glsl) in the include directories also trigger a rebuild.
    set(header_deps "")
    foreach(dir IN LISTS ARG_INCLUDE_DIRS)
      file(GLOB dir_headers CONFIGURE_DEPENDS "${dir}/*.glsl")
      list(APPEND header_deps ${dir_headers})
    endforeach()
    add_custom_command(
      OUTPUT "${spv}"
      COMMAND ${CMAKE_COMMAND} -E make_directory "${ARG_OUTPUT_DIR}"
      COMMAND "${GVL_GLSLANG_VALIDATOR}" -V --target-env vulkan1.3 ${include_flags} -o "${spv}" "${source_abs}"
      DEPENDS "${source_abs}" ${header_deps}
      COMMENT "SPIR-V: ${source_name}"
      VERBATIM)
    list(APPEND spv_files "${spv}")
  endforeach()

  add_custom_target(${target}_shaders DEPENDS ${spv_files})
  add_dependencies(${target} ${target}_shaders)
endfunction()

function(gvl_embed_shaders target)
  cmake_parse_arguments(ARG "" "" "SOURCES;INCLUDE_DIRS" ${ARGN})
  if(NOT ARG_SOURCES)
    message(FATAL_ERROR "gvl_embed_shaders(${target}): SOURCES is required")
  endif()
  set(out_dir "${CMAKE_CURRENT_BINARY_DIR}/embedded_shaders")
  gvl_add_shaders(${target} OUTPUT_DIR "${out_dir}" SOURCES ${ARG_SOURCES} INCLUDE_DIRS ${ARG_INCLUDE_DIRS})

  foreach(source IN LISTS ARG_SOURCES)
    get_filename_component(source_name "${source}" NAME)
    string(MAKE_C_IDENTIFIER "${source_name}" symbol)
    set(spv "${out_dir}/${source_name}.spv")
    set(cpp "${out_dir}/${source_name}.spv.cpp")
    add_custom_command(
      OUTPUT "${cpp}"
      COMMAND ${CMAKE_COMMAND} -DINPUT=${spv} -DOUTPUT=${cpp} -DSYMBOL=${symbol} -P "${GVL_EMBED_SPIRV_SCRIPT}"
      DEPENDS "${spv}" "${GVL_EMBED_SPIRV_SCRIPT}"
      COMMENT "Embedding ${source_name}.spv"
      VERBATIM)
    target_sources(${target} PRIVATE "${cpp}")
  endforeach()
endfunction()
