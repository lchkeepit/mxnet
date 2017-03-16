/*!
 * created by chenhao li
*/

#include "./hardtanh.h"
#include <vector>
#include "./mshadow_op.h"
namespace mshadow {
namespace cuda {
template<typename DType>
__global__ void HardTanhForwardKernel(DType *data, DType *out, int Y, int data_stride, int out_stride, int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) {
        int idx1 = i / Y * data_stride + i % Y;
        int idx2 = i / Y * out_stride + i % Y;
        DType x = *(data + idx1);
        if (x > 1.)
            *(out + idx2) = 1;
        else if (x < -1.)
            *(out + idx2) = -1;
        else
            *(out + idx2) = x;
    }
}
template<typename DType>
void HardTanhForward(Tensor<gpu, 2, DType> &data, Tensor<gpu, 2, DType> &out) {
    DType *data_ptr = data.dptr_;
    DType *out_ptr = out.dptr_;
    int size = data.shape_[0] * data.shape_[1];
    cudaStream_t stream = Stream<gpu>::GetStream(out.stream_);

    dim3 numBlocks((size + kMaxThreadsPerBlock - 1)/kMaxThreadsPerBlock);
    dim3 threadsPerBlock(kMaxThreadsPerBlock);

    HardTanhForwardKernel<DType><<<numBlocks, threadsPerBlock, 0, stream>>>(data_ptr, out_ptr, data.shape_[1], data.stride_, out.stride_, size);
}
template <typename DType>
__global__ void HardTanhBackwardKernel(DType *data, DType *grad, DType *gdata, int Y, int data_stride, int grad_stride, int gdata_stride, int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    if (i < N) {
        int idx1 = i / Y * data_stride + i % Y;
        int idx2 = i / Y * grad_stride + i % Y;
        int idx3 = i / Y * gdata_stride + i % Y;
        if (*(data + idx1) > 1. || *(data + idx1) < -1.)
            *(gdata + idx3) = 0;
        else
            *(gdata + idx3) = *(grad + idx2);
    }
}
template<typename DType>
void HardTanhBackward(Tensor<gpu, 2, DType> &data, Tensor<gpu, 2, DType> &grad, Tensor<gpu, 2, DType> &gdata) {
    DType *grad_ptr = grad.dptr_;
    DType *gdata_ptr = gdata.dptr_;
    DType *data_ptr = data.dptr_;

    int size = grad.shape_[0] * grad.shape_[1];
    cudaStream_t stream = Stream<gpu>::GetStream(gdata.stream_);

    dim3 numBlocks((size + kMaxThreadsPerBlock - 1)/kMaxThreadsPerBlock);
    dim3 threadsPerBlock(kMaxThreadsPerBlock);

    HardTanhBackwardKernel<DType><<<numBlocks, threadsPerBlock, 0, stream>>>(data_ptr, grad_ptr, gdata_ptr,
                                                                            data.shape_[1], data.stride_, grad.stride_, gdata.stride_, size);
}
}
using namespace mshadow::expr;
template<typename DType>
void HardTanhForward(Tensor<gpu, 2, DType> &data, Tensor<gpu, 2, DType> &out) {
    cuda::HardTanhForward(data, out);
    //out = F<mxnet::op::mshadow_op::identity>(data);
}
template<typename DType>
void HardTanhBackward(Tensor<gpu, 2, DType> &data, Tensor<gpu, 2, DType> &grad, Tensor<gpu, 2, DType> &gdata) {
    cuda::HardTanhBackward(data, grad, gdata);
    //gdata = F<mxnet::op::mshadow_op::identity>(grad);
}

}

namespace mxnet {
namespace op {
template<>
Operator* CreateOp<gpu>(HardTanhParam param, int dtype) {
    Operator *op = NULL;
    MSHADOW_REAL_TYPE_SWITCH(dtype, DType, {
            op = new HardTanhOp<gpu, DType>(param);
    })
    return op;
}

}  // namespace op
}  // namespace mxnet

