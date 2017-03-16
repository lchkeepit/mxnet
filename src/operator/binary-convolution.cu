/*!
 * created by chenhao li
*/

#include "./binary-convolution.h"
#include <vector>

namespace mshadow {
namespace cuda {
template  <typename DType>
__global__ void BinaryConvolution(DType *wmat, DType *wbmat, int Y, int wmat_stride, int wbmat_stride, int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) {
        int idx1 = i / Y * wmat_stride + i % Y;
        int idx2 = i / Y * wbmat_stride + i % Y;
        if (*(wmat + idx1) >= 0)
            *(wbmat + idx2) = 1;
        else
            *(wbmat + idx2) = -1;
    }
}

template <typename DType>
void BinaryConvolutionForward(Tensor<gpu, 2, DType> wmat, Tensor<gpu, 2, DType> data, Tensor<gpu, 2, DType> out) {
    Tensor<gpu, 2, DType> wbmat(wmat.shape_);
    AllocSpace(&wbmat);

    int size = wmat.shape_[0] * wmat.shape_[1];
    cudaStream_t stream = Stream<gpu>::GetStream(out.stream_);
    dim3 numBlocks((size + kMaxThreadsPerBlock - 1)/kMaxThreadsPerBlock);
    dim3 threadsPerBlock(kMaxThreadsPerBlock);
    DType *wmat_dptr = wmat.dptr_;
    DType *wbmat_dptr = wbmat.dptr_;

    cuda::BinaryConvolution<DType><<<numBlocks, threadsPerBlock, 0, stream>>>(wmat_dptr, wbmat_dptr,
                                                                                wmat.shape_[1], wmat.stride_, wbmat.stride_, size);
    cudaStreamSynchronize(stream);

    out = dot(wbmat, data);
    FreeSpace(&wbmat);
}
template <typename DType>
void BinaryConvolutionBackward(Tensor<gpu, 2, DType> wmat, Tensor<gpu, 2, DType> data, Tensor<gpu, 2, DType> out) {
    Tensor<gpu, 2, DType> wbmat(wmat.shape_);
    AllocSpace(&wbmat);

    int size = wmat.shape_[0] * wmat.shape_[1];
    cudaStream_t stream = Stream<gpu>::GetStream(out.stream_);
    dim3 numBlocks((size + kMaxThreadsPerBlock - 1)/kMaxThreadsPerBlock);
    dim3 threadsPerBlock(kMaxThreadsPerBlock);
    DType *wmat_dptr = wmat.dptr_;
    DType *wbmat_dptr = wbmat.dptr_;

    cuda::BinaryConvolution<DType><<<numBlocks, threadsPerBlock, 0, stream>>>(wmat_dptr, wbmat_dptr,
                                                                                wmat.shape_[1], wmat.stride_, wbmat.stride_, size);
    cudaStreamSynchronize(stream);

    out = dot(wbmat.T(), data);
    FreeSpace(&wbmat);
}
}
template <typename DType>
void BinaryConvolutionForward(Tensor<gpu, 2, DType> wmat, Tensor<gpu, 2, DType> data, Tensor<gpu, 2, DType> out) {
    cuda::BinaryConvolutionForward(wmat, data, out);
}
template <typename DType>
void BinaryConvolutionBackward(Tensor<gpu, 2, DType> wmat, Tensor<gpu, 2, DType> data, Tensor<gpu, 2, DType> out) {
    cuda::BinaryConvolutionBackward(wmat, data, out);
}
}


namespace mxnet {
namespace op {
template<>
Operator* CreateOp<gpu>(BinaryConvolutionParam param, int dtype,
                        std::vector<TShape> *in_shape,
                        std::vector<TShape> *out_shape,
                        Context ctx) {
    Operator *op = NULL;
    MSHADOW_REAL_TYPE_SWITCH(dtype, DType, {
            op = new BinaryConvolutionOp<gpu, DType>(param);
    })
    return op;
}

}  // namespace op
}  // namespace mxnet

