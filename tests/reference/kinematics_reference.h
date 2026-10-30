// Independent CPU reference for rigid transformations and Denavit-Hartenberg forward kinematics (double precision).
//
// The DH matrices are NOT copied from the CUDA code: they are composed from elementary transformations,
//   classic:  Rz(theta) * Tz(d) * Tx(a) * Rx(alpha)
//   Craig:    Rx(alpha) * Tx(a) * Rz(theta) * Tz(d)
// so that the formulas written in robot/src/KinematicLink.cpp are verified, not repeated.
#ifndef GVL_TEST_KINEMATICS_REFERENCE_H_INCLUDED
#define GVL_TEST_KINEMATICS_REFERENCE_H_INCLUDED

#include <array>
#include <cmath>
#include <vector>

namespace gvl_reference
{
	struct Mat4
	{
		std::array<double, 16> m{ 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1 };

		double& at(int r, int c) { return m[r * 4 + c]; }
		[[nodiscard]] double at(int r, int c) const { return m[r * 4 + c]; }

		[[nodiscard]] Mat4 operator*(const Mat4& o) const
		{
			Mat4 r;
			for (int i = 0; i < 4; ++i)
				for (int j = 0; j < 4; ++j)
				{
					double sum = 0;
					for (int k = 0; k < 4; ++k)
						sum += at(i, k) * o.at(k, j);
					r.at(i, j) = sum;
				}
			return r;
		}

		/// R * p + t
		template<typename P>
		[[nodiscard]] std::array<double, 3> apply(const P& p) const
		{
			const double x = p.x, y = p.y, z = p.z;
			return { at(0, 0) * x + at(0, 1) * y + at(0, 2) * z + at(0, 3), at(1, 0) * x + at(1, 1) * y + at(1, 2) * z + at(1, 3),
				at(2, 0) * x + at(2, 1) * y + at(2, 2) * z + at(2, 3) };
		}
	};

	inline Mat4 translation(double x, double y, double z)
	{
		Mat4 t;
		t.at(0, 3) = x; t.at(1, 3) = y; t.at(2, 3) = z;
		return t;
	}

	inline Mat4 rotation_x(double a)
	{
		Mat4 r;
		r.at(1, 1) = std::cos(a); r.at(1, 2) = -std::sin(a);
		r.at(2, 1) = std::sin(a); r.at(2, 2) = std::cos(a);
		return r;
	}

	inline Mat4 rotation_y(double a)
	{
		Mat4 r;
		r.at(0, 0) = std::cos(a); r.at(0, 2) = std::sin(a);
		r.at(2, 0) = -std::sin(a); r.at(2, 2) = std::cos(a);
		return r;
	}

	inline Mat4 rotation_z(double a)
	{
		Mat4 r;
		r.at(0, 0) = std::cos(a); r.at(0, 1) = -std::sin(a);
		r.at(1, 0) = std::sin(a); r.at(1, 1) = std::cos(a);
		return r;
	}

	struct DH
	{
		double d = 0, theta = 0, a = 0, alpha = 0, value = 0;
		bool prismatic = false;
		bool craig = false;
	};

	inline Mat4 dh_matrix(const DH& p)
	{
		const double d = p.prismatic ? p.d + p.value : p.d;
		const double theta = p.prismatic ? p.theta : p.theta + p.value;
		if (!p.craig)
			return rotation_z(theta) * translation(0, 0, d) * translation(p.a, 0, 0) * rotation_x(p.alpha);
		return rotation_x(p.alpha) * translation(p.a, 0, 0) * rotation_z(theta) * translation(0, 0, d);
	}

	/// transforms[0] = base, transforms[i + 1] = transforms[i] * DH_i
	inline std::vector<Mat4> forward_kinematics(const Mat4& base, const std::vector<DH>& links)
	{
		std::vector<Mat4> transforms{ base };
		for (const auto& link : links)
			transforms.push_back(transforms.back() * dh_matrix(link));
		return transforms;
	}
}

#endif
