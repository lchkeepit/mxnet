/*!
 * created by chenhao li
*/

#include "./binary-activation.h"
#include <vector>
#include <cstdlib>
#include <cmath>
#include <curand.h>
#include <curand_kernel.h>

namespace mshadow {
namespace cuda {
template <typename DType>
__global__ void BinaryActivationForwardKernel(DType *data, DType *out, int Y, int data_stride, int out_stride, int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) return;
    //*(out + i) = *(data + i); return ;
    int idx1 = i / Y * data_stride + i % Y;
    int idx2 = i / Y * out_stride + i % Y;

    DType x = *(data + idx1);
    //x = (x + 1) / 2.;
    //curandState_t state;
    //x = x - curand_uniform (&state);
    if (x >= 0)
        *(out + idx2) = 1;
    else
        *(out + idx2) = -1;
}
template<typename DType>
void BinaryActivationForward(Tensor<gpu, 2, DType> &data, Tensor<gpu, 2, DType> &out) {
    DType *data_ptr = data.dptr_;
    DType *out_ptr = out.dptr_;
    int size = data.shape_[0] * data.shape_[1];
    cudaStream_t stream = Stream<gpu>::GetStream(out.stream_);

    dim3 numBlocks((size + kMaxThreadsPerBlock - 1)/kMaxThreadsPerBlock);
    dim3 threadsPerBlock(kMaxThreadsPerBlock);

    BinaryActivationForwardKernel<DType><<<numBlocks, threadsPerBlock, 0, stream>>>(data_ptr, out_ptr,
                                                                                    data.shape_[1], data.stride_, out.stride_, size);

}
template <typename DType>
__global__ void BinaryActivationBackwardKernel(DType *grad, DType *gdata, int Y, int grad_stride, int gdata_stride, int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N){
        int idx1 = i / Y * grad_stride + i % Y;
        int idx2 = i / Y * gdata_stride + i % Y;
        *(gdata + idx2) = *(grad + idx1);
    }

}
template<typename DType>
void BinaryActivationBackward(Tensor<gpu, 2, DType> &grad, Tensor<gpu, 2, DType> &gdata) {
    DType *grad_ptr = grad.dptr_;
    DType *gdata_ptr = gdata.dptr_;
    cudaStream_t stream = Stream<gpu>::GetStream(gdata.stream_);
    int size = grad.shape_[0] * grad.shape_[1];

    dim3 numBlocks((size + kMaxThreadsPerBlock - 1)/kMaxThreadsPerBlock);
    dim3 threadsPerBlock(kMaxThreadsPerBlock);

    BinaryActivationBackwardKernel<DType><<<numBlocks, threadsPerBlock, 0, stream>>>(grad_ptr, gdata_ptr,
                                                                                    grad.shape_[1], grad.stride_, gdata.stride_, size);
}
}//cuda
template<typename DType>
void BinaryActivationForward(Tensor<gpu, 2, DType> &data, Tensor<gpu, 2, DType> &out) {
    cuda::BinaryActivationForward(data, out);
}
template<typename DType>
void BinaryActivationBackward(Tensor<gpu, 2, DType> &grad, Tensor<gpu, 2, DType> &gdata) {
    cuda::BinaryActivationBackward(grad, gdata);
}
}


namespace mxnet {
namespace op {
template<>
Operator* CreateOp<gpu>(BinaryActivationParam param, int dtype) {
    Operator *op = NULL;
    MSHADOW_REAL_TYPE_SWITCH(dtype, DType, {
            op = new BinaryActivationOp<gpu, DType>(param);
    })
    return op;
}

}  // namespace op
}  // namespace mxnet

