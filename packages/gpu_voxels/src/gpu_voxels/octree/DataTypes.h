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
 * \date    2013-11-07
 *
 */
 //----------------------------------------------------------------------/*
#ifndef GPU_VOXELS_OCTREE_DATATYPES_H_INCLUDED
#define GPU_VOXELS_OCTREE_DATATYPES_H_INCLUDED

//#define DEBUG_MODE

//#include <cstdio>
//#include <iostream>
#include <stdint.h> // for fixed size datatypes
#include <assert.h>
#include <stdio.h>
#include <limits.h>
#include <math.h>
#include <time.h>
#include <chrono>
#include <algorithm>
//#include <sys/utsname.h>
#include <gpu_voxels/helpers/common_defines.h>
#include <gpu_voxels/helpers/cuda_datatypes.hpp>
#include <gpu_voxels/helpers/BitVector.h>

#include "CommonValues.h"

namespace gpu_voxels {
	namespace NTree {

		//-Wno-unknown-pragmas -Wno-unused-function

#define D_PTR(X) thrust::raw_pointer_cast((X).data())
#define MAX_VALUE(TYPE) ((TYPE)((1 << (sizeof(TYPE) * 8)) - 1))

#define INVALID_VOXEL ULONG_MAX
#define DISABLE_SEPARATE_COMPILTION


// NTree
#define FEW_MESSAGES
//#define PROPAGATE_MESSAGES
//#define INSERT_MESSAGES
//#define FREESPACE_MESSAGES
#define REBUILD_MESSAGES
//#define EXTRACTCUBE_MESSAGES
//#define TRAFO_MESSAGES
//#define NEWKINECTDATA_MESSAGES
#define INTERSECT_MESSAGES
//#define FREE_BOUNDING_BOX_MESSAGES
//#define SENSOR_DATA_PREPROCESSING_MESSAGES

#define KINECT_PREPROCESSS_ON_GPU
//#define CHECK_SORTING
#define LOAD_BALANCING_PROPAGATE
#define PROPAGATE_BOTTOM_UP
//#define COUNT_BEFORE_EXTRACT

// Kinect
//#define DEPTHCALLBACK_MESSAGES

#define MODE_KINECT
//#define MANUAL_MODE
//#define PAN_TILT
//#define IGNORE_FREE_SPACE

// ########## LoadBalancer ##########
#define DEFAULT_PROPAGATE_QUEUE_NTASKS 1024 // good choice for number of tasks/blocks due to experimental evaluation
#define PROPAGATE_TRAVERSAL_THREADS 128//128
#define PROPAGATE_IDLETASKS_FOR_ABORT_FACTOR 2/3

#define QUEUE_NTASKS 2688//2688//96//1024 //128
#define QUEUE_SIZE_PER_TASK 2600//2600
#define QUEUE_SIZE_PER_TASK_GLOBAL 700//50000
#define QUEUE_IDLETASKS_FOR_ABORT QUEUE_NTASKS*2/3//2/3
#define TRAVERSAL_THREADS 128 //32//64 //64
#define QUEUE_SIZE_PER_TASK_INIT TRAVERSAL_THREADS*3
#define WARP_SIZE 32
#define MAX_NUMBER_OF_THREADS 1024

// Define min/max functions to handle different namespaces of host and device code
#ifdef __CUDACC__
#define MIN(x,y) min(x,y)
#define MAX(x,y) max(x,y)
#else
#define MIN(x,y) std::min(x,y)
#define MAX(x,y) std::max(x,y)
#endif

// ######################################################################

		typedef uint32_t voxel_count;

		/*
		 * Returns the difference in milliseconds
		 */
		inline static double timeDiff(timespec start, timespec end)
		{
			double ms = 0.0;
			if ((end.tv_nsec - start.tv_nsec) < 0)
			{
				ms = static_cast<double>(1000000000 + end.tv_nsec - start.tv_nsec) / 1000000.0;
				ms += static_cast<double>(end.tv_sec - start.tv_sec - 1) * 1000.0;
			}
			else
			{
				ms = static_cast<double>(end.tv_nsec - start.tv_nsec) / 1000000.0;
				ms += static_cast<double>(end.tv_sec - start.tv_sec) * 1000.0;
			}
			return ms;
		}

		inline static timespec getCPUTime()
		{
			auto tp = std::chrono::system_clock::now();

			auto secs = std::chrono::time_point_cast<std::chrono::seconds>(tp);
			auto ns = std::chrono::time_point_cast<std::chrono::nanoseconds>(tp) -
				std::chrono::time_point_cast<std::chrono::nanoseconds>(secs);

			return timespec{ secs.time_since_epoch().count(), static_cast<long>(ns.count()) };
		}

		inline static std::string getTime_str()
		{
			time_t rawtime;
			struct tm* timeinfo;
			char buffer[80];

			time(&rawtime);
			timeinfo = localtime(&rawtime);

			strftime(buffer, 80, "%F_%H.%M.%S", timeinfo);
			return buffer;
		}

		inline std::string to_string(int _Val, const char format[] = "%d")
		{   // convert long long to string
			char _Buf[50];
			sprintf(_Buf, format, _Val);
			return { _Buf };
		}

		/*
	inline utsname getUname()
	{
	  utsname tmp;
	  uname(&tmp);
	  return tmp;
	}*/

	//#define lookup_type_8 uint8_t
	//#define lookup_type_64 uint8_t
	//#define lookup_type_512 uint16_t
	//#define lookup_type_4096 uint16_t
	//#define lookup_type(X) lookup_type_"X"

	//#define third_root(X) third_root_"X"
	//#define third_root_8 2
	//#define third_root_64 4
	//#define third_root_512 8

	//// ##### type comparison at compile time #####
	//template<typename T>
	//struct is_same<T, T>
	//{
	//    static const bool value = true;
	//};
	//
	//template<typename T, typename U>
	//struct is_same
	//{
	//    static const bool value = false;
	//};
	//
	//template<typename T, typename U>
	//bool eqlTypes() { return is_same<T, U>::value; }
	//// #################################################

		template<typename t0, typename t1, typename t2, typename t3, typename t4, typename t5 = uint32_t, typename t6 = uint32_t>
		__host__
		uint32_t linearApprox(t0 y1, t1 x1, t2 y2, t3 x2, t4 x,
			t5 alignment = 1, t6 max_val = UINT_MAX)
		{
			const float a = static_cast<float>(y1 - y2) / static_cast<float>(x1 - x2);
			const float b = y1 - a * x1;
			const float y = a * x + b;
			const float y_aligned = ceil(y / alignment) * alignment;
			const uint32_t y_min_max = (std::min)(static_cast<uint32_t>((std::max)(y_aligned, static_cast<float>(alignment))), static_cast<uint32_t>(max_val));

			return y_min_max;
		}

	} // end of ns
} // end of ns
#endif