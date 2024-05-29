#ifndef GPU_VOXELS_BIT_VECTOR_H_INCLUDED
#define GPU_VOXELS_BIT_VECTOR_H_INCLUDED

#include <cstdint>
#include <iostream>
#include <bitset>

#include <cuda_runtime_api.h>

namespace gpu_voxels
{
    /**
     * @brief This template class represents a vector of bits with a given number of bits.
     */
    template<size_t num_bits>
    class BitVector
    {

    public:
        typedef uint8_t item_type;

    protected:

        /**
         * @brief getByte Gets the reference to the byte that contains the bit at the given index position (given in bits).
         * @return Reference to the byte that contains the bit at the given bit index position
         */
        __host__ __device__
        item_type* getByte(uint32_t index);

    public:

        __host__ __device__
        BitVector();

        /**
         * @brief clear Sets all bits to zero.
         */
        __host__ __device__
        void clear();

        /**
         * @brief operator | Bitwise or-operator
         * @param o Other operand

         */
        __host__ __device__
        BitVector operator|(const BitVector& o) const;

        /**
         * @brief operator == Bitwise equal comparison
         * @param o Other operand
         */
        __host__ __device__
        bool operator==(const BitVector& o) const;

        /**
         * @brief operator |= Bitwise or-operator
         * @param o Other operand
         */
        __host__ __device__
        void operator|=(const BitVector& o);

        /**
         * @brief operator ~ Bitwise not-operator
         * @return Returns the bitwise not of 'this'
         */
        __host__ __device__
        BitVector operator~() const;

        /**
         * @brief operator ~ Bitwise and-operator
         * @return Returns the bitwise and of 'this'
         */
        __host__ __device__
        BitVector operator&(const BitVector& o) const;

        /**
         * @brief isZero Checks the bit vector for zero
         * @return True if all bits are zero, false otherwise
         */
        __host__ __device__
        [[nodiscard]] bool isZero() const;

        /**
         * @brief noneButEmpty Checks for semantic emptiness
         * @return True, if none or only the eBVM_FREE bit is set
         */
        __host__ __device__
        [[nodiscard]] bool noneButEmpty() const;

        /**
         * @brief anyNotEmpty Checks for semantic occupation
         * @return True, if any bit is set ignoring eBVM_FREE
         */
        __host__ __device__
        [[nodiscard]] bool anyNotEmpty() const;

        /**
         * @brief getBit Gets the bit at the given bit index.
         * @return Value of the selected bit.
         */
        __host__ __device__
        [[nodiscard]] bool getBit(const uint32_t index) const;

        /**
         * @brief clearBit Clears the bit at the given bit index
         */
        __host__ __device__
        void clearBit(const uint32_t index);

        /**
         * @brief setBit Sets the bit at the given bit index.
         */
        __host__ __device__
        void setBit(const uint32_t index);

        /**
         * @brief getByte Gets the byte that contains the bit at the given index position (given in Bits).
         * Note: The dummy argument helps nvcc to distinguish this function from the protected pointer version
         *
         * @return Byte that contains the bit at the given bit index position
         */
        __host__ __device__
        [[nodiscard]] item_type getByte(const uint32_t index, const uint8_t dummy = 0) const;

        /**
         * @brief setByte Sets the byte at the given bit index position.
         * @param index Which byte to set (given in bits)
         * @param data Data to write into byte
         */
        __host__ __device__
        void setByte(const uint32_t index, const item_type data);

        __host__ __device__
        void dump();

        /**
         * @brief operator << Overloaded ostream operator. Please note that the output bit string is starting from
         * Type 0.
         */
        template<typename T>
        __host__
        friend T& operator<<(T& os, const BitVector& dt)
        {
            constexpr size_t byte_size = sizeof(item_type);
            for (uint32_t i = 0; i < num_bits; i += byte_size * 8)
            {
                item_type byte = dt.getByte(i);
                // reverse bit order
                byte = (byte & 0xF0) >> 4 | (byte & 0x0F) << 4;
                byte = (byte & 0xCC) >> 2 | (byte & 0x33) << 2;
                byte = (byte & 0xAA) >> 1 | (byte & 0x55) << 1;
                std::bitset<byte_size * 8> bs(byte);
                os << bs.to_string();
            }
            return os;
        }

        /**
         * @brief operator >> Overloaded istream operator. Please note that the input bit string should
         * be starting from Type 0 and it should be complete, meaning it should have all Bits defined.
         */
        __host__
        friend std::istream& operator>>(std::istream& in, BitVector& dt)
        {
            //TODO: Check the lengths of the input stream!
            item_type byte;
            std::bitset<num_bits> bs;
            in >> bs;
            const size_t byte_size = sizeof(item_type);
            for (uint32_t i = 0; i < num_bits; i += byte_size * 8)
            {
                // The Bit reverse is in here
                byte = bs[i + 7] + 2 * bs[i + 6] + 4 * bs[i + 5] + 8 * bs[i + 4] + 16 * bs[i + 3] + 32 * bs[i + 2] + 64 * bs[i + 1] + 128 * bs[i + 0];

                // Fill last bit first
                dt.setByte(static_cast<uint32_t>(num_bits - i - 1), byte);
            }
            return in;
        }

        // This CUDA Code was taken from Florians BitVoxelFlags that got replaced by BitVectors
        __device__
        static void reduce(BitVector<num_bits>& flags, const int thread_id, const int num_threads,
            BitVector<num_bits>* shared_mem);

        __device__
        static void reduceAtomic(BitVector<num_bits>& flags, BitVector<num_bits>& global_flags);

    protected:
        static constexpr uint32_t m_size = (num_bits + 7) / 8; // the size in Byte; + 7 ensures that no bits get dropped
        item_type m_bytes[m_size];

    }; // END OF CLASS BitVector


    /**
     * @brief performLeftShift Shifts the bits of a bitvector to the left
     * (decrease the SV Meaning and therefore shift the bits to the right)
     * This function sets the non Swept-Volume Meanings to 0!
     * @param shift_size How many bits to shift. Must be smaller than 56 bits due to buffer size.
     */
    template<std::size_t num_bits>
    __host__ __device__
    void performLeftShift(BitVector<num_bits>& bit_vector, const uint8_t shift_size);



    /**
     * @brief bitMarginCollisionCheck
     * @param v1 Bitvector 1
     * @param v2 Bitvector 2
     * @param collisions Aggregates the colliding bits. This will NOT get reset!
     * @param margin Fuzzyness of the check. How many bits get checked aside the actual colliding bit.
     * @param sv_offset Bit-Offset added to v1 before colliding
     * @return
     */
    template<std::size_t num_bits>
    __host__ __device__
    bool bitMarginCollisionCheck(const BitVector<num_bits>& v1, const BitVector<num_bits>& v2,
        BitVector<num_bits>* collisions, const uint8_t margin, const uint32_t sv_offset);



    /*!
     * \brief The BitvectorOr struct
     * Thrust operator that calculates the OR operation on two BitVectors
     */
    template<std::size_t num_bits>
    struct BitvectorOr
    {
        __host__ __device__
        BitVector<num_bits> operator()(const BitVector<num_bits>& lhs, const BitVector<num_bits>& rhs) const;
    };
}

#endif