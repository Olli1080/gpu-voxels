// Shared definitions of the BitVoxelMap kernels.
// A BitVoxel<256> is 256 bits = 8 words; bit i of the voxel (word i/32, bit i%32) is set when the voxel has the
// BitVoxelMeaning i. This is byte-for-byte the memory layout of the CUDA BitVector<256> on little-endian hosts.
const uint WORDS_PER_VOXEL = 8u;
const uint BVM_COLLISION = 2u;

uint meaning_word(uint meaning) { return meaning >> 5u; }
uint meaning_mask(uint meaning) { return 1u << (meaning & 31u); }
