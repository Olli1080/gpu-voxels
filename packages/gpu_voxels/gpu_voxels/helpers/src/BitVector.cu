// this is for emacs file handling -*- mode: c++; indent-tabs-mode: nil -*-

// -- BEGIN LICENSE BLOCK ----------------------------------------------
// This file is part of the GPU Voxels Software Library.
//
// This program is free software licensed under the CDDL
// (COMMON DEVELOPMENT AND DISTRIBUTION LICENSE Version 1.0).
// You can find a copy of this license in LICENSE.txt in the top
// directory of the source code.
//
// © Copyright 2014 FZI Forschungszentrum Informatik, Karlsruhe, Germany
//
// -- END LICENSE BLOCK ------------------------------------------------

//----------------------------------------------------------------------
/*!\file
 *
 * \author  Florian Drews
 * \date    2014-07-08
 *
 */
//----------------------------------------------------------------------/*
#include <gpu_voxels/helpers/BitVector.cuhpp>

namespace gpu_voxels
{
    template class BitVector<BIT_VECTOR_LENGTH>;

    template __host__ __device__ void performLeftShift<BIT_VECTOR_LENGTH>(BitVector<BIT_VECTOR_LENGTH>& bit_vector, const uint8_t shift_size);
    template __host__ __device__ bool bitMarginCollisionCheck<BIT_VECTOR_LENGTH>(const BitVector<BIT_VECTOR_LENGTH>& v1, const BitVector<BIT_VECTOR_LENGTH>& v2,
            BitVector<BIT_VECTOR_LENGTH>* collisions, const uint8_t margin, const uint32_t sv_offset);

    template struct BitvectorOr<BIT_VECTOR_LENGTH>;
} // end of ns