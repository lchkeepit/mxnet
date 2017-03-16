/*!
 * created by chenhao li
*/

#include "./binary-scale.h"
#include <vector>
#include <cstdlib>
#include <cmath>
#include <curand.h>
#include <curand_kernel.h>

namespace mshadow {
namespace cuda {
template <typename DType>
__global__ void BinaryScaleForwardKernel(DType *data, DType *wValue, DType *out, int Y, int data_stride, int out_stride, int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) return;
    int idx1 = i / Y * data_stride + i % Y;
    int idx2 = i / Y * out_stride + i % Y;
    *(out + idx2)  = (*(data + idx1)) * (*wValue);
}
template<typename DType>
void BinaryScaleForward(Tensor<gpu, 2, DType> &data, Tensor<gpu, 2, DType> &wValue, Tensor<gpu, 2, DType> &out) {
    DType *data_ptr = data.dptr_;
    DType *wValue_ptr = wValue.dptr_;
    DType *out_ptr = out.dptr_;

    cudaStream_t stream = Stream<gpu>::GetStream(out.stream_);
    int size = data.shape_[0] * data.shape_[1];

    dim3 numBlocks((size + kMaxThreadsPerBlock - 1)/kMaxThreadsPerBlock);
    dim3 threadsPerBlock(kMaxThreadsPerBlock);

    BinaryScaleForwardKernel<DType><<<numBlocks, threadsPerBlock, 0, stream>>>(data_ptr, wValue_ptr, out_ptr,
                                                                                data.shape_[1], data.stride_, out.stride_, size);
}
template <typename DType>
__global__ void BinaryScaleBackwardKernel(DType *data, DType *grad, DType *wValue, DType *gwValue, DType *gdata,
                                          int Y, int data_stride, int grad_stride, int gdata_stride, int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) return;
    int idx1 = i / Y * data_stride + i % Y;
    int idx2 = i / Y * grad_stride + i % Y;
    int idx3 = i / Y * gdata_stride + i % Y;
    *gwValue += (*(data + idx1)) * (*(grad + idx2));
    *(gdata + idx3) = (*(grad + idx2)) * (*wValue);
}
template<typename DType>
void BinaryScaleBackward(Tensor<gpu, 2, DType> &data, Tensor<gpu, 2, DType> &grad, Tensor<gpu, 2, DType> &wValue,
                         Tensor<gpu, 2, DType> &gwValue, Tensor<gpu, 2, DType> &gdata) {
    DType *grad_ptr = grad.dptr_;
    DType *gwValue_ptr = gwValue.dptr_;
    DType *data_ptr = data.dptr_;
    DType *wValue_ptr = wValue.dptr_;
    DType *gdata_ptr = gdata.dptr_;

    cudaStream_t stream = Stream<gpu>::GetStream(gdata.stream_);
    int size = grad.shape_[0] * grad.shape_[1];

    dim3 numBlocks((size + kMaxThreadsPerBlock - 1)/kMaxThreadsPerBlock);
    dim3 threadsPerBlock(kMaxThreadsPerBlock);

    BinaryScaleBackwardKernel<DType><<<numBlocks, threadsPerBlock, 0, stream>>>(data_ptr, grad_ptr, wValue_ptr, gwValue_ptr, gdata_ptr,
                                                                                data.shape_[1], data.stride_, grad.stride_, gdata.stride_, size);
}
}//cuda
template<typename DType>
void BinaryScaleForward(Tensor<gpu, 2, DType> &data, Tensor<gpu, 2, DType> &wValue, Tensor<gpu, 2, DType> &out) {
    cuda::BinaryScaleForward(data, wValue, out);
}
template<typename DType>
void BinaryScaleBackward(Tensor<gpu, 2, DType> &data, Tensor<gpu, 2, DType> &grad, Tensor<gpu, 2, DType> &wValue,
                         Tensor<gpu, 2, DType> &gwValue, Tensor<gpu, 2, DType> &gdata) {
    cuda::BinaryScaleBackward(data, grad, wValue, gwValue, gdata);
}
}


namespace mxnet {
namespace op {
template<>
Operator* CreateOp<gpu>(BinaryScaleParam param, int dtype) {
    Operator *op = NULL;
    MSHADOW_REAL_TYPE_SWITCH(dtype, DType, {
            op = new BinaryScaleOp<gpu, DType>(param);
    })
    return op;
}

}  // namespace op
}  // namespace mxnet

