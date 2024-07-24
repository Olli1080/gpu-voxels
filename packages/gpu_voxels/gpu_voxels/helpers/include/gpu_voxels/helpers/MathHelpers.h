// this is for emacs file handling -&- mode: c++; indent-tabs-mode: nil -&-

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
 * \author  Andreas Hermann
 * \date    2015-10-02
 *
 */
//----------------------------------------------------------------------
#ifndef GPU_VOXELS_HELPERS_MATH_HELPERS_H_INCLUDED
#define GPU_VOXELS_HELPERS_MATH_HELPERS_H_INCLUDED

#include <cstdint>

#include <vector>
#include <string>
#include <map>

namespace gpu_voxels
{

    /*!
     * \brief computeLinearLoad
     * \param nr_of_items
     * \param blocks
     * \param threads_per_block
     */
    void computeLinearLoad(uint32_t nr_of_items, uint32_t& blocks, uint32_t& threads_per_block);

    /*! Interpolate linear between the values \a value1 and \a value2 using the given \a ratio.
     *  Using values out of [0.0, 1.0] will extrapolate, a value of 0.5 will interpolate in the
     *  middle.
     */
    template<std::floating_point T>
    float interpolateLinear(T value1, T value2, T ratio)
    {
        return (value1 * (static_cast<T>(1.) - ratio) + value2 * ratio);
    }

    /*! Interpolate linear between the robot joint vectors \a joint_state1 and \a joint_state2
     *  using the given \a ratio.
     *  Using values out of [0.0, 1.0] will extrapolate, a value of 0.5 will interpolate in the
     *  middle.
     */
    template<std::floating_point T>
    std::vector<T> interpolateLinear(const std::vector<T>& joint_state1,
        const std::vector<T>& joint_state2, T ratio)
    {
        assert(joint_state1.size() == joint_state2.size());

        std::vector<T> result(joint_state1.size());
        for (std::size_t i = 0; i < joint_state1.size(); ++i)
        {
            result[i] = interpolateLinear<T>(joint_state1[i], joint_state2[i], ratio);
        }
        return result;
    }

    /*! Interpolate linear between the robot JointValueMaps \a joint_state1 and \a joint_state2
     *  using the given \a ratio.
     *  Using values out of [0.0, 1.0] will extrapolate, a value of 0.5 will interpolate in the
     *  middle.
     */
    template<std::floating_point T>
    std::map<std::string, T> interpolateLinear(const std::map<std::string, T>& joint_state1,
        const std::map<std::string, T>& joint_state2, T ratio)
    {
        assert(joint_state1.size() == joint_state2.size());

        std::map result(joint_state1);
        for (auto it = joint_state1.begin(); it != joint_state1.end(); ++it)
        {
            result[it->first] = interpolateLinear<T>(joint_state1.at(it->first), joint_state2.at(it->first), ratio);
        }
        return result;
    }


    template<std::floating_point T>
    std::map<std::string, T> interpolateLinear(const std::map<std::string, T>& joint_state1,
        const std::map<std::string, T>& joint_state2, const std::map<std::string, T>& ratio)
    {
        assert(joint_state1.size() == joint_state2.size() && joint_state1.size() == ratio.size());

        std::map result(joint_state1);
        for (auto it = joint_state1.begin(); it != joint_state1.end(); ++it)
        {
            result[it->first] = interpolateLinear<T>(joint_state1.at(it->first), joint_state2.at(it->first), ratio.at(it->first));
        }
        return result;
    }
} // end of namespace

#endif