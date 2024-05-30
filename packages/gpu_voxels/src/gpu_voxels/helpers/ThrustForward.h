#pragma once

namespace thrust
{
    template<typename T>
    class device_allocator;

    template <class T, typename> class device_vector;
    template <class T> class device_ptr;
}

template <class T>
using ThrustDeviceVector = thrust::device_vector<T, thrust::device_allocator<T>>;