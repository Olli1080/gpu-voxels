// Shared definitions of the viewer kernels. A cube instance is 32 bytes: the minimum corner and the edge length in world units, a packed colour
// (R in the low byte, A in the high byte: unpackUnorm4x8) and 12 bytes of padding (std430: the struct is 16-byte aligned).
struct CubeInstance
{
    vec4 pos_size;
    uint rgba;
    uint pad0;
    uint pad1;
    uint pad2;
};

// Words of the draw arguments buffer: the four words of VkDrawIndirectCommand (vertexCount = 36, instanceCount, firstVertex, firstInstance) and the
// number of cubes that were found (also those that did not fit into the instance buffer).
const uint ARG_VERTEX_COUNT = 0u;
const uint ARG_INSTANCE_COUNT = 1u;
const uint ARG_FIRST_VERTEX = 2u;
const uint ARG_FIRST_INSTANCE = 3u;
const uint ARG_FOUND = 4u;
const uint CUBE_VERTEX_COUNT = 36u;
