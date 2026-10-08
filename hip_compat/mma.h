// Maps the CUDA WMMA API (nvcuda::wmma) onto rocWMMA, which provides the
// same fragment types and load/store/mma functions.
#ifndef HIP_COMPAT_MMA_H
#define HIP_COMPAT_MMA_H

#include <hip/hip_fp16.h>
#include <rocwmma/rocwmma.hpp>

namespace nvcuda
{
namespace wmma
{
using namespace rocwmma;
using rocwmma::mma_sync;

// The matrix cores of gfx9 GPUs accumulate in FP32, and rocWMMA does not
// resolve an FP16 accumulator on them. Accumulate in FP32 there and round the
// result back to FP16 after each step, as rocWMMA does internally for FP16
// accumulators on these GPUs.
template <uint32_t BlockM, uint32_t BlockN, uint32_t BlockK, typename LayoutA,
          typename LayoutB>
__device__ void
mma_sync(fragment<accumulator, BlockM, BlockN, BlockK, __half>& d,
         fragment<matrix_a, BlockM, BlockN, BlockK, __half, LayoutA> const& a,
         fragment<matrix_b, BlockM, BlockN, BlockK, __half, LayoutB> const& b,
         fragment<accumulator, BlockM, BlockN, BlockK, __half>& c)
{
#if defined(__GFX9__)
    using AccFragment = fragment<accumulator, BlockM, BlockN, BlockK, float>;
    using HalfAccFragment =
        fragment<accumulator, BlockM, BlockN, BlockK, __half>;
    static_assert(AccFragment::num_elements == HalfAccFragment::num_elements);
    AccFragment acc;
    for (uint32_t i{0U}; i < acc.num_elements; ++i)
    {
        acc.x[i] = __half2float(c.x[i]);
    }
    rocwmma::mma_sync(acc, a, b, acc);
    for (uint32_t i{0U}; i < acc.num_elements; ++i)
    {
        d.x[i] = __float2half(acc.x[i]);
    }
#else
    rocwmma::mma_sync(d, a, b, c);
#endif
}
} // namespace wmma
} // namespace nvcuda

#endif // HIP_COMPAT_MMA_H
