// this is for emacs file handling -*- mode: c++; indent-tabs-mode: nil -*-

// -- BEGIN LICENSE BLOCK ----------------------------------------------
// This file is part of the GPU Voxels Software Library.
//
// This program is free software licensed under the CDDL
// (COMMON DEVELOPMENT AND DISTRIBUTION LICENSE Version 1.0).
// You can find a copy of this license in LICENSE.txt in the top
// directory of the source code.
// -- END LICENSE BLOCK ------------------------------------------------

//----------------------------------------------------------------------
/*!ile
 *
 * rief   Log stream of the Vulkan backend runtime.
 * \date    2026-10-22
 */
//----------------------------------------------------------------------
#ifndef LOGGING_VULKAN_H_
#define LOGGING_VULKAN_H_

#include <icl_core_logging/Logging.h>
namespace gpu_voxels
{
	DECLARE_LOG_STREAM(VulkanLog)
	using icl_core::logging::endl;
}
#endif /* LOGGING_VULKAN_H_ */
