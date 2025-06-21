// this is for emacs file handling -*- mode: c++; indent-tabs-mode: nil -*-

// -- BEGIN LICENSE BLOCK ----------------------------------------------
// This program is free software licensed under the CDDL
// (COMMON DEVELOPMENT AND DISTRIBUTION LICENSE Version 1.0).
// You can find a copy of this license in LICENSE in the top
// directory of the source code.
//
// © Copyright 2018 FZI Forschungszentrum Informatik, Karlsruhe, Germany
// -- END LICENSE BLOCK ------------------------------------------------

//----------------------------------------------------------------------
/*!\file
 *
 * \author  Klaus Uhl <uhl@fzi.de>
 * \date    2010-06-30
 *
 * \brief   Contains macros to deprecate classes, types, functions and variables.
 *
 * Deprecation warnings can be disabled by compiling with the
 * ICL_CORE_NO_DEPRECATION macro defined.
 */
//----------------------------------------------------------------------
#ifndef ICL_CORE_DEPRECATE_H_INCLUDED
#define ICL_CORE_DEPRECATE_H_INCLUDED

#define ICL_CORE_DEPRECATE [[deprecated]]
#define ICL_CORE_DEPRECATE_COMMENT(arg) [[deprecated(#arg)]]

#define ICL_CORE_VC_DEPRECATE ICL_CORE_DEPRECATE
#define ICL_CORE_VC_DEPRECATE_COMMENT(arg) ICL_CORE_DEPRECATE_COMMENT(arg)

# define ICL_CORE_GCC_DEPRECATE ICL_CORE_DEPRECATE
# define ICL_CORE_GCC_DEPRECATE_COMMENT(arg) ICL_CORE_DEPRECATE_COMMENT(arg)


// Special comment for deprecation due to obsolete style.
#define ICL_CORE_DEPRECATE_STYLE ICL_CORE_DEPRECATE_COMMENT("Please follow the new Coding Style Guidelines.")

#define ICL_CORE_VC_DEPRECATE_STYLE ICL_CORE_DEPRECATE_STYLE
#define ICL_CORE_GCC_DEPRECATE_STYLE ICL_CORE_DEPRECATE_STYLE

// Special comment for changing to new source sink pattern.
#define ICL_CORE_DEPRECATE_SOURCESINK ICL_CORE_DEPRECATE_COMMENT("Please follow the new Source Sink Pattern.")

#define ICL_CORE_VC_DEPRECATE_SOURCESINK ICL_CORE_DEPRECATE_SOURCESINK
#define ICL_CORE_GCC_DEPRECATE_SOURCESINK ICL_CORE_DEPRECATE_SOURCESINK

// Special comment for moving to ROS workspace.
#define ICL_CORE_DEPRECATE_MOVE_ROS ICL_CORE_DEPRECATE_COMMENT("This was moved to a ROS package. Please use the implementation in ros_icl or ros_sourcesink instead.")

#define ICL_CORE_VC_DEPRECATE_MOVE_ROS ICL_CORE_DEPRECATE_MOVE_ROS
#define ICL_CORE_GCC_DEPRECATE_MOVE_ROS ICL_CORE_DEPRECATE_MOVE_ROS

// Special comment for deprecation due to obsolete style which
// provides the name of the function that superseded the obsolete one.
#define ICL_CORE_DEPRECATE_STYLE_USE(arg) ICL_CORE_DEPRECATE_COMMENT("Please follow the new Coding Style Guidelines and use " #arg " instead.")

#define ICL_CORE_VC_DEPRECATE_STYLE_USE(arg) ICL_CORE_DEPRECATE_STYLE_USE(arg)
#define ICL_CORE_GCC_DEPRECATE_STYLE_USE(arg) ICL_CORE_DEPRECATE_STYLE_USE(arg)

#endif
