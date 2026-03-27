// this is for emacs file handling -*- mode: c++; indent-tabs-mode: nil -*-

// -- BEGIN LICENSE BLOCK ----------------------------------------------
// -- END LICENSE BLOCK ------------------------------------------------

//----------------------------------------------------------------------
/*!\file
 *
 * \author  Florian Drews
 * \date    2013-11-18
 *
 */
 //----------------------------------------------------------------------/*

#include "Sensor.h"
#include <gpu_voxels/helpers/cuda_handling.h>
#include "kernels/kernel_PointCloud.h"

#include <icl_core_performance_monitor/PerformanceMonitor.h>

using namespace std;

namespace gpu_voxels {
	namespace NTree {

		GVL_HOST
			void Sensor::_processDepthImage(const DepthData* h_sensor_data,
				parallel::device_vector<Vector3f>& d_free_space_points,
				parallel::device_vector<Vector3f>& d_object_points)
		{
#ifdef IC_PERFORMANCE_MONITOR
			const string prefix = "processSensorData";
			const string temp_timer = prefix + "_temp";
			PERF_MON_START(prefix);
			PERF_MON_START(temp_timer);
#endif

			constexpr uint32_t num_threads = 128;
			const uint32_t num_blocks = data_width * data_height / num_threads / 4;

			const bool data_equals = object_data.equals(free_space_data) && object_data.m_process_data;
			const bool process_object_data = object_data.m_process_data;
			const bool process_free_space_data = !data_equals && free_space_data.m_process_data;
			LOGGING_INFO_C(OctreeLog, Sensor, "process_free_space_data " << process_free_space_data << endl);

			parallel::device_vector<DepthData> d_depth_image(data_width * data_height);
			parallel::device_vector<DepthData> d_depth_image_free_space;
			d_object_points.resize(data_width * data_height);

			if (process_free_space_data)
			{
				d_depth_image_free_space.resize(data_width * data_height);
				d_free_space_points.resize(data_width * data_height);
			}

			// copy to GPU
			GVL_HANDLE_ERROR(GVL_MEMCPY(D_PTR(d_depth_image), h_sensor_data, d_depth_image.size() * sizeof(DepthData), GVL_MEMCPY_HOST_TO_DEVICE));

			// wait for copy to complete
			//GVL_HANDLE_ERROR(GVL_SYNCHRONIZE());
#ifdef IC_PERFORMANCE_MONITOR
			PERF_MON_PRINT_AND_RESET_INFO_P(temp_timer, "Memcpy", prefix);
#endif
			//printf("Memcpy depth image(): %f ms\n", timeDiff(time, getCPUTime()));

			// TODO: (preprocess depth image on GPU) e.g. filter
			//preprocessDepthImage(D_PTR(d_depth_image), width, height, noSampleValue, shadowValue, DepthData(MAX_RANGE));

			// preprocessing of depth image to mark invalid data, remove out of range measures and so on
			if (process_free_space_data)
			{
				// copy depth image for separate processing
				d_depth_image_free_space = d_depth_image;
				GVL_HANDLE_ERROR(GVL_SYNCHRONIZE());
				kernel_preprocessDepthImage << <num_blocks, num_threads >> >
					(D_PTR(d_depth_image_free_space), data_width, data_height, free_space_data);
				GVL_HANDLE_ERROR(GVL_SYNCHRONIZE());
			}
			if (process_object_data)
			{
				kernel_preprocessDepthImage << <num_blocks, num_threads >> >
					(D_PTR(d_depth_image), data_width, data_height, object_data);
				GVL_HANDLE_ERROR(GVL_SYNCHRONIZE());
			}
#ifdef IC_PERFORMANCE_MONITOR
			PERF_MON_PRINT_AND_RESET_INFO_P(temp_timer, "Preprocessing", prefix);
#endif
			// transform depth image to point cloud in sensor coordinate system
			parallel::device_vector<Sensor> d_sensor(1, *this);
			if (process_free_space_data)
			{
				kernel_transformDepthImage << <num_blocks, num_threads >> >
					(D_PTR(d_depth_image_free_space),
						D_PTR(d_free_space_points),
						D_PTR(d_sensor),
						free_space_data.m_invalid_measure);
				GVL_HANDLE_ERROR(GVL_SYNCHRONIZE());
			}

			if (process_object_data)
			{
				kernel_transformDepthImage << <num_blocks, num_threads >> >
					(D_PTR(d_depth_image),
						D_PTR(d_object_points),
						D_PTR(d_sensor),
						object_data.m_invalid_measure);
				GVL_HANDLE_ERROR(GVL_SYNCHRONIZE());
			}
#ifdef IC_PERFORMANCE_MONITOR
			PERF_MON_PRINT_AND_RESET_INFO_P(temp_timer, "ToPointCloud", prefix);
#endif
			if (data_equals && free_space_data.m_process_data)
			{
				d_free_space_points = d_object_points;
			}
		}

		GVL_HOST
		void Sensor::processSensorData(const Vector3f* h_points,
			parallel::device_vector<Voxel>*& d_free_space_voxel,
			parallel::device_vector<Voxel>*& d_object_voxel)
		{

			if (!d_free_space_voxel) d_free_space_voxel = new parallel::device_vector<Voxel>;
			if (!d_object_voxel) d_object_voxel = new parallel::device_vector<Voxel>;

			parallel::device_vector<gpu_voxels::Vector3f> d_points_free(data_width * data_height);
			parallel::device_vector<gpu_voxels::Vector3f> d_points_object(data_width * data_height);

			// copy to GPU
			GVL_HANDLE_ERROR(GVL_MEMCPY(D_PTR(d_points_object), h_points, d_points_free.size() * sizeof(gpu_voxels::Vector3f), GVL_MEMCPY_HOST_TO_DEVICE));

			d_points_free = d_points_object;
			_processSensorData(d_points_free, d_points_object, *d_free_space_voxel, *d_object_voxel);
		}

		GVL_HOST
		void Sensor::_processSensorData(parallel::device_vector<Vector3f>& d_free_space_points,
			parallel::device_vector<Vector3f>& d_object_points,
			parallel::device_vector<Voxel>& d_free_space_voxel,
			parallel::device_vector<Voxel>& d_object_voxel)
		{
#ifdef IC_PERFORMANCE_MONITOR
			const string prefix = "processSensorData";
			const string temp_timer = prefix + "_temp";
			PERF_MON_START(prefix);
			PERF_MON_START(temp_timer);
#endif

			bool data_equals = object_data.equals(free_space_data) && object_data.m_process_data;
			bool process_object_data = object_data.m_process_data;
			bool process_free_space_data = !data_equals && free_space_data.m_process_data;

			// finally remove the invalid data from the point cloud
			if (process_free_space_data)
				removeInvalidPoints(d_free_space_points);

			if (process_object_data)
				removeInvalidPoints(d_object_points);

			GVL_HANDLE_ERROR(GVL_SYNCHRONIZE());
#ifdef IC_PERFORMANCE_MONITOR
			PERF_MON_PRINT_AND_RESET_INFO_P(temp_timer, "RemoveInvalidPoints", prefix);

			std::size_t n_free = 0, n_object = 0;
			if (process_free_space_data)
				n_free = data_equals ? d_object_points.size() : d_free_space_points.size();
			if (process_object_data)
				n_object = d_object_points.size();

			PERF_MON_ADD_DATA_NONTIME_P("FreeSpacePoints", n_free, prefix);
			PERF_MON_ADD_DATA_NONTIME_P("ObjectPoints", n_object, prefix);
#endif
			if (process_free_space_data)
			{
				Sensor free_space_sensor = *this;
				free_space_sensor.sensorModel.setInitialProbability(free_space_data.m_initial_probability);
				free_space_sensor.sensorModel.setUpdateProbability(free_space_data.m_update_probability);
				parallel::device_vector<Sensor> d_tmp_sensor(1);
				d_tmp_sensor[0] = free_space_sensor;
				GVL_HANDLE_ERROR(GVL_SYNCHRONIZE());

				transformKinectPointCloud_simple(D_PTR(d_free_space_points),
					static_cast<uint32_t>(d_free_space_points.size()),
					d_free_space_voxel,
					D_PTR(d_tmp_sensor),
					free_space_data.m_voxel_side_length);
			}
			if (process_object_data)
			{
				Sensor object_sensor = *this;
				object_sensor.sensorModel.setInitialProbability(object_data.m_initial_probability);
				object_sensor.sensorModel.setUpdateProbability(object_data.m_update_probability);
				parallel::device_vector<Sensor> d_tmp_sensor(1);
				d_tmp_sensor[0] = object_sensor;
				GVL_HANDLE_ERROR(GVL_SYNCHRONIZE());

				transformKinectPointCloud_simple(D_PTR(d_object_points),
					static_cast<uint32_t>(d_object_points.size()),
					d_object_voxel,
					D_PTR(d_tmp_sensor),
					object_data.m_voxel_side_length);
			}

			if (data_equals && free_space_data.m_process_data)
			{
				d_free_space_voxel = d_object_voxel;
			}
#ifdef IC_PERFORMANCE_MONITOR
			PERF_MON_PRINT_INFO_P(temp_timer, "transformKinectPointCloud_simple", prefix);

			n_free = n_object = 0;
			if (process_free_space_data)
				n_free = d_free_space_voxel.size();
			if (process_object_data)
				n_object = d_object_voxel.size();
			PERF_MON_ADD_DATA_NONTIME_P("FreeSpaceVoxel", n_free, prefix);
			PERF_MON_ADD_DATA_NONTIME_P("ObjectVoxel", n_object, prefix);
			PERF_MON_PRINT_INFO_P(prefix, "", prefix);
#endif
		}

		GVL_HOST
			void Sensor::processSensorData(const DepthData* h_sensor_data,
				parallel::device_vector<Voxel>*& d_free_space_voxel,
				parallel::device_vector<Voxel>*& d_object_voxel)
		{
			if (!d_free_space_voxel) d_free_space_voxel = new parallel::device_vector<Voxel>;
			if (!d_object_voxel) d_object_voxel = new parallel::device_vector<Voxel>;

			parallel::device_vector<Vector3f> d_free_space_points;
			parallel::device_vector<Vector3f> d_object_points;
			_processDepthImage(h_sensor_data, d_free_space_points, d_object_points);
			_processSensorData(d_free_space_points, d_object_points, *d_free_space_voxel, *d_object_voxel);

			//  const string prefix = __FUNCTION__;
			//  const string temp_timer = prefix + "_temp";
			//  PERF_MON_START(prefix);
			//  PERF_MON_START(temp_timer);
			//
			//  const uint32_t num_threads = 128;
			//  const uint32_t num_blocks = data_width*data_height / num_threads / 4;
			//
			//  bool data_equals = object_data.equals(free_space_data) && object_data.m_process_data;
			//  bool process_object_data = object_data.m_process_data;
			//  bool process_free_space_data = !data_equals && free_space_data.m_process_data;
			//  printf("process_free_space_data %i \n", process_free_space_data);
			//
			//  parallel::device_vector<DepthData> d_depth_image(data_width * data_height);
			//  parallel::device_vector<DepthData> d_depth_image_free_space;
			//  parallel::device_vector<gpu_voxels::Vector3f> free_space_points; // object and free space points
			//  parallel::device_vector<gpu_voxels::Vector3f> object_points(data_width * data_height); // only object points
			//
			//  if(process_free_space_data)
			//  {
			//    d_depth_image_free_space.resize(data_width * data_height);
			//    free_space_points.resize(data_width * data_height);
			//  }
			//
			//  // copy to GPU
			//  GVL_HANDLE_ERROR(GVL_MEMCPY(D_PTR(d_depth_image), h_sensor_data, d_depth_image.size() * sizeof(DepthData), GVL_MEMCPY_HOST_TO_DEVICE));
			//
			//  // wait for copy to complete
			//  //GVL_HANDLE_ERROR(GVL_SYNCHRONIZE());
			//
			//  PERF_MON_PRINT_AND_RESET_INFO_P(temp_timer, "Memcpy", prefix);
			//
			//  //printf("Memcpy depth image(): %f ms\n", timeDiff(time, getCPUTime()));
			//
			//  // TODO: (preprocess depth image on GPU) e.g. filter
			//  //preprocessDepthImage(D_PTR(d_depth_image), width, height, noSampleValue, shadowValue, DepthData(MAX_RANGE));
			//
			//  // preprocessing of depth image to mark invalid data, remove out of range measures and so on
			//  if (process_free_space_data)
			//  {
			//    // copy depth image for separate processing
			//    d_depth_image_free_space = d_depth_image;
			//    GVL_HANDLE_ERROR(GVL_SYNCHRONIZE());
			//    kernel_preprocessDepthImage<<<num_blocks, num_threads>>>
			//    (D_PTR(d_depth_image_free_space), data_width, data_height, free_space_data);
			//    GVL_CHECK_ERROR();
			//  }
			//  if (process_object_data)
			//  {
			//    kernel_preprocessDepthImage<<<num_blocks, num_threads>>>
			//    (D_PTR(d_depth_image), data_width, data_height, object_data);
			//    GVL_CHECK_ERROR();
			//  }
			//  GVL_HANDLE_ERROR(GVL_SYNCHRONIZE());
			//  PERF_MON_PRINT_AND_RESET_INFO_P(temp_timer, "Preprocessing", prefix);
			//
			//  // transform depth image to point cloud in sensor coordinate system
			//  parallel::device_vector<Sensor> d_sensor(1, *this);
			//  if (process_free_space_data)
			//  {
			//    kernel_transformDepthImage<<<num_blocks, num_threads>>>
			//    (D_PTR(d_depth_image_free_space),
			//        D_PTR(free_space_points),
			//        D_PTR(d_sensor),
			//        free_space_data.m_invalid_measure);
			//    GVL_CHECK_ERROR();
			//  }
			//
			//  if (process_object_data)
			//  {
			//    kernel_transformDepthImage<<<num_blocks, num_threads>>>
			//    (D_PTR(d_depth_image),
			//        D_PTR(object_points),
			//        D_PTR(d_sensor),
			//        object_data.m_invalid_measure);
			//    GVL_CHECK_ERROR();
			//  }
			//  GVL_HANDLE_ERROR(GVL_SYNCHRONIZE());
			//  PERF_MON_PRINT_AND_RESET_INFO_P(temp_timer, "ToPointCloud", prefix);
			//
			//
			//  // finally remove the invalid data from the point cloud
			//  if (process_free_space_data)
			//  {
			//    removeInvalidPoints(free_space_points);
			//  }
			//  if (process_object_data)
			//  {
			//    removeInvalidPoints(object_points);
			//  }
			//  GVL_HANDLE_ERROR(GVL_SYNCHRONIZE());
			//  PERF_MON_PRINT_AND_RESET_INFO_P(temp_timer, "RemoveInvalidPoints", prefix);
			//
			//  std::size_t n_free = 0, n_object = 0;
			//  if (process_free_space_data)
			//    n_free = data_equals ? object_points.size() : d_depth_image_free_space.size();
			//  if (process_object_data)
			//    n_object = object_points.size();
			//
			//  PERF_MON_ADD_DATA_NONTIME_P("FreeSpacePoints", n_free, prefix);
			//  PERF_MON_ADD_DATA_NONTIME_P("ObjectPoints", n_object, prefix);
			//
			//  if (process_free_space_data)
			//  {
			//    Sensor free_space_sensor = *this;
			//    free_space_sensor.sensorModel.setInitialProbability(free_space_data.m_initial_probability);
			//    free_space_sensor.sensorModel.setUpdateProbability(free_space_data.m_update_probability);
			//    parallel::device_vector<Sensor> d_tmp_sensor(1);
			//    d_tmp_sensor[0] = free_space_sensor;
			//    GVL_HANDLE_ERROR(GVL_SYNCHRONIZE());
			//
			//    transformKinectPointCloud_simple(D_PTR(free_space_points),
			//    free_space_points.size(),
			//    d_free_space_voxel,
			//    D_PTR(d_tmp_sensor),
			//    free_space_data.m_voxel_side_length);
			//  }
			//  if (process_object_data)
			//  {
			//    Sensor object_sensor = *this;
			//    object_sensor.sensorModel.setInitialProbability(object_data.m_initial_probability);
			//    object_sensor.sensorModel.setUpdateProbability(object_data.m_update_probability);
			//    parallel::device_vector<Sensor> d_tmp_sensor(1);
			//    d_tmp_sensor[0] = object_sensor;
			//    GVL_HANDLE_ERROR(GVL_SYNCHRONIZE());
			//
			//    transformKinectPointCloud_simple(D_PTR(object_points),
			//    object_points.size(),
			//    d_object_voxel,
			//    D_PTR(d_tmp_sensor),
			//    object_data.m_voxel_side_length);
			//  }
			//
			//  if (data_equals && free_space_data.m_process_data)
			//  {
			//    d_free_space_voxel = d_object_voxel;
			//  }
			//
			//  PERF_MON_PRINT_INFO_P(temp_timer, "transformKinectPointCloud_simple", prefix);
			//
			//  n_free = n_object = 0;
			//  if (process_free_space_data)
			//    n_free = d_free_space_voxel.size();
			//  if (process_object_data)
			//    n_object = d_object_voxel.size();
			//  PERF_MON_ADD_DATA_NONTIME_P("FreeSpaceVoxel", n_free, prefix);
			//  PERF_MON_ADD_DATA_NONTIME_P("ObjectVoxel", n_objec, prefixt);
			//  PERF_MON_PRINT_INFO_P(prefix, "", prefix);
		}

	}
}