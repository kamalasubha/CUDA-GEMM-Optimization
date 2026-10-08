// Maps the CUDA WMMA API (nvcuda::wmma) onto rocWMMA, which provides the
// same fragment types and load/store/mma functions.
#ifndef HIP_COMPAT_MMA_H
#define HIP_COMPAT_MMA_H

#include <rocwmma/rocwmma.hpp>

namespace nvcuda
{
namespace wmma = rocwmma;
} // namespace nvcuda

#endif // HIP_COMPAT_MMA_H
