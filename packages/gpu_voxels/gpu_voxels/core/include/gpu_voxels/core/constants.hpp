#pragma once

#ifdef M_PI
#undef M_PI
#endif

#ifdef M_PI_2
#undef M_PI_2
#endif

#ifndef __CUDACC__
#include <concepts>
#include <numbers>

template <typename Floating>
inline constexpr Floating M_PI = std::numbers::pi_v<Floating>;

template <typename Floating>
inline constexpr Floating M_PI_2 = M_PI<Floating> / static_cast<Floating>(2.0);

#else

template <typename Floating>
constexpr Floating M_PI = static_cast<Floating>(3.14159265358979323846);

template <typename Floating>
constexpr Floating M_PI_2 = M_PI<Floating> / static_cast<Floating>(2.0);
#endif

//template <> inline constexpr float M_PI_2<