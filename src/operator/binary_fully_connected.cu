/*!
 * Copyright (c) 2015 by Contributors
 * \file fully_connected.cu
 * \brief fully connect operator
*/
#include "./binary_fully_connected.h"

namespace mshadow {
namespace cuda {
template <typename DType>
__global__ void BinaryFullyConnectedKernel(DType *wmat, DType *wbmat, int Y, int wmat_stride, int wbmat_stride, int N) {
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
void BinaryFullyConnectedForward(Tensor<gpu, 2, DType> &data,
                            Tensor<gpu, 2, DType> &wmat,
                            Tensor<gpu, 2, DType> &out) {
    //printf("enter FC Forward\n");
    DType *data_ptr = data.dptr_;
    DType *wmat_ptr = wmat.dptr_;
    DType *out_ptr = out.dptr_;
    int size = wmat.shape_[0] * wmat.shape_[1];
    cudaStream_t stream = Stream<gpu>::GetStream(out.stream_);
    dim3 numBlocks((size + kMaxThreadsPerBlock - 1)/kMaxThreadsPerBlock);
    dim3 threadsPerBlock(kMaxThreadsPerBlock);
    //debugOutput<DType>(data_ptr, data.shape_[0] * data.shape_[1]);
    Tensor<gpu, 2, DType> wbmat(wmat.shape_); AllocSpace(&wbmat);
    DType *wbmat_ptr = wbmat.dptr_;
    //printf("BinaryForward 1\n");
    BinaryFullyConnectedKernel<DType><<<numBlocks, threadsPerBlock, 0, stream>>>(wmat_ptr, wbmat_ptr,
                                                                                        wmat.shape_[1], wmat.stride_, wbmat.stride_, size);
    //printf("BinaryForward 2\n");
    cudaStreamSynchronize(stream);

    out = dot(data, wbmat.T());
    FreeSpace(&wbmat);
}
template <typename DType>
void BinaryFullyConnectedBackward(Tensor<gpu, 2, DType> &data,
                            Tensor<gpu, 2, DType> &grad,
                            Tensor<gpu, 2, DType> &wmat,
                            Tensor<gpu, 2, DType> &gwmat,
                            Tensor<gpu, 2, DType> &gdata) {
    DType *data_ptr = data.dptr_;
    DType *grad_ptr = grad.dptr_;
    DType *wmat_ptr = wmat.dptr_;
    DType *gwmat_ptr = gwmat.dptr_;
    DType *gdata_ptr = gdata.dptr_;
    int size = wmat.shape_[0] * wmat.shape_[1];
    cudaStream_t stream = Stream<gpu>::GetStream(gdata.stream_);

    dim3 numBlocks((size + kMaxThreadsPerBlock - 1)/kMaxThreadsPerBlock);
    dim3 threadsPerBlock(kMaxThreadsPerBlock);

    Tensor<gpu, 2, DType> wbmat(wmat.shape_); AllocSpace(&wbmat);
    DType *wbmat_ptr = wbmat.dptr_;
    //printf("BinaryBackward 1\n");
    BinaryFullyConnectedKernel<DType><<<numBlocks, threadsPerBlock, 0, stream>>>(wmat_ptr, wbmat_ptr,
                                                                                wmat.shape_[1], wmat.stride_, wbmat.stride_, size);
    //printf("BinaryBackward 2\n");
    cudaStreamSynchronize(stream);
    gwmat = dot(grad.T(), data);
    gdata = dot(grad, wbmat);
    FreeSpace(&wbmat);
}
}//cuda
template <typename DType>
void BinaryFullyConnectedForward(Tensor<gpu, 2, DType> &data,
                            Tensor<gpu, 2, DType> &wmat,
                            Tensor<gpu, 2, DType> &out) {
    cuda::BinaryFullyConnectedForward(data, wmat, out);
}
template <typename DType>
void BinaryFullyConnectedBackward(Tensor<gpu, 2, DType> &data,
                            Tensor<gpu, 2, DType> &grad,
                            Tensor<gpu, 2, DType> &wmat,
                            Tensor<gpu, 2, DType> &gwmat,
                            Tensor<gpu, 2, DType> &gdata) {
    cuda::BinaryFullyConnectedBackward(data, grad, wmat, gwmat, gdata);
}
}


namespace mxnet {
namespace op {
template<>
Operator* CreateOp<gpu>(BinaryFullyConnectedParam param, int dtype) {
    Operator *op = NULL;
    MSHADOW_REAL_TYPE_SWITCH(dtype, DType, {
            op = new BinaryFullyConnectedOp<gpu, DType>(param);
    })
    return op;
}
}  // namespace op
}  // namespace mxnet
