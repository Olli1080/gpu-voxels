// this is for emacs file handling -*- mode: c++; indent-tabs-mode: nil -*-

// -- BEGIN LICENSE BLOCK ----------------------------------------------
// -- END LICENSE BLOCK ------------------------------------------------

//----------------------------------------------------------------------
/*!\file
 *
 * \author  Sebastian Klemm
 * \date    2012-09-13
 *
 */
 //----------------------------------------------------------------------
#include "KinematicLink.h"
#include "gpu_voxels/logging/logging_robot.h"

namespace gpu_voxels
{
	namespace robot
	{
		template<>
		void DHParameters<CLASSIC>::convertDHtoM(Matrix4f& m) const
		{
			//  printf("theta, d, a, alpha : \t%f, %f, %f, %f\n", theta, d, a, alpha);
			float d_current = d;
			float theta_current = theta;

			switch (joint_type)
			{
			case PRISMATIC:
			{
				// if prismatic joint, increment d with joint value
				d_current = d + value;
				break;
			}
			case REVOLUTE:
			{
				// if revolute joint, increment theta with joint value
				theta_current = theta + value;
				break;
			}
			default:
			{
				LOGGING_ERROR_C(RobotLog, DHParameters, "Illegal joint type" << endl);
			}
			}

			const float ca = cos(alpha);
			const float sa = sin(alpha);
			const float ct = cos(theta_current);
			const float st = sin(theta_current);

			//1st row
			m(0, 0) = ct;
			m(0, 1) = -st * ca;
			m(0, 2) = st * sa;
			m(0, 3) = a * ct;

			//2nd row
			m(1, 0) = st;
			m(1, 1) = ct * ca;
			m(1, 2) = -ct * sa;
			m(1, 3) = a * st;

			//3rd row
			m(2, 0) = 0.f;
			m(2, 1) = sa;
			m(2, 2) = ca;
			m(2, 3) = d_current;

			//4th row
			m(3, 0) = 0.f;
			m(3, 1) = 0.f;
			m(3, 2) = 0.f;
			m(3, 3) = 1.f;
		}

		template<>
		void DHParameters<CRAIGS>::convertDHtoM(Matrix4f& m) const
		{
			//  printf("theta, d, a, alpha : \t%f, %f, %f, %f\n", theta, d, a, alpha);
			float d_current = d;
			float theta_current = theta;

			switch (joint_type)
			{
			case PRISMATIC:
			{
				// if prismatic joint, increment d with joint value
				d_current = d + value;
				break;
			}
			case REVOLUTE:
			{
				// if revolute joint, increment theta with joint value
				theta_current = theta + value;
				break;
			}
			default:
			{
				LOGGING_ERROR_C(RobotLog, DHParameters, "Illegal joint type" << endl);
			}
			}

			const float ca = cos(alpha);
			const float sa = sin(alpha);
			const float ct = cos(theta_current);
			const float st = sin(theta_current);

			//1st row
			m(0, 0) = ct;
			m(0, 1) = -st;
			m(0, 2) = 0.f;
			m(0, 3) = a;

			//2nd row
			m(1, 0) = st * ca;
			m(1, 1) = ct * ca;
			m(1, 2) = -sa;
			m(1, 3) = -d_current * sa;

			//3rd row
			m(2, 0) = st * sa;
			m(2, 1) = ct * sa;
			m(2, 2) = ca;
			m(2, 3) = d_current * ca;

			//4th row
			m(3, 0) = 0.f;
			m(3, 1) = 0.f;
			m(3, 2) = 0.f;
			m(3, 3) = 1.f;
		}

		template<DHConvention convention>
		KinematicLink<convention>::KinematicLink(const DHParameters<convention>& dh_parameters)
			: m_dh_parameters(dh_parameters)
		{}

		template<DHConvention convention>
		KinematicLink<convention>::KinematicLink(float d, float theta, float a, float alpha, float joint_value, DHJointType joint_type)
			: m_dh_parameters(d, theta, a, alpha, joint_value, joint_type)
		{}

		template<DHConvention convention>
		void KinematicLink<convention>::setDHParam(float d, float theta, float a, float alpha, float joint_value, DHJointType joint_type)
		{
			m_dh_parameters.d = d;
			m_dh_parameters.theta = theta;
			m_dh_parameters.a = a;
			m_dh_parameters.alpha = alpha;
			m_dh_parameters.value = joint_value;
			m_dh_parameters.joint_type = joint_type;
		}

		template<DHConvention convention>
		void KinematicLink<convention>::setDHParam(const DHParameters<convention>& dh_parameters)
		{
			m_dh_parameters = dh_parameters;
		}

		template<DHConvention convention>
		void KinematicLink<convention>::getMatrixRepresentation(Matrix4f& m) const
		{
			m_dh_parameters.convertDHtoM(m);
		}

		template class KinematicLink<CLASSIC>;
		template class KinematicLink<CRAIGS>;

	} // end of namespace
} // end of namespace