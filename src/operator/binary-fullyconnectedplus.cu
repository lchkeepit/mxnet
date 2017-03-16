/*!
 * Copyright (c) 2015 by Contributors
 * \file fully_connected.cu
 * \brief fully connect operator
*/
#include "./binary-fullyconnectedplus.h"
#include "cuda.h"
#include "../common/cuda_utils.h"
namespace mshadow {
namespace cuda {
template <typename DType>
__global__ void BinaryFullyConnectedPlusCudaBinarize(DType *src, DType *dst, int Y, int src_stride, int dst_stride, int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) {
        int idx1 = i / Y * src_stride + i % Y;
        int idx2 = i / Y * dst_stride + i % Y;
        if (*(src + idx1) >= 0)
            *(dst + idx2) = 1;
        else
            *(dst + idx2) = -1;
    }
}

template <typename DType>
__global__ void BinaryFullyConnectedPlusCudaInit(DType *src, int Y, int src_stride, int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) return;
    int idx = i / Y * src_stride + i % Y;

    *(src + idx) = 0;
}
template <typename DType>
void debugOutput(DType *src, int size);
template<typename DType>
__global__ void BinaryFullyConnectedPlusCudaSumKernel(DType *src, DType *dst, int Y, int src_stride, int N) {
    __shared__ DType box[1024];
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int threadid = threadIdx.x;
    DType temp = 0;
    while (tid < N) {
        DType x = src[tid / Y * src_stride + tid % Y];
        if (x < 0) x *= -1.;
        temp += x;
        tid += blockDim.x * gridDim.x;
        //printf("%d %d %d waiting\n", blockIdx.x, threadid, tid);
    }
    box[threadid] = temp;
    //printf("%d %d waiting\n", blockIdx.x, threadid);
    __syncthreads();
    int i = blockDim.x / 2;
    while (i != 0) {
        //printf("%d : %d\n", blockIdx.x, i);
        if (threadid < i) {
            box[threadid] += box[threadid + i];
        }
        __syncthreads();
        i /= 2;
    }
    if (threadid == 0) {
        dst[blockIdx.x] = box[0];
        //printf("fc %d %d\n", gridDim.x, blockDim.x);
    }
}
template <typename DType>
void BinaryFullyConnectedPlusCudaSum(DType *src, DType *dst, int Y, int src_stride, int N, cudaStream_t &stream) {
    //printf("start cudasum\n");
    int threads = 1024;
    dim3 numBlocks(32);
    dim3 threadsPerBlock(threads);
    DType *val;
    auto msg = cudaMalloc(&val, 32 * sizeof(DType));
    BinaryFullyConnectedPlusCudaInit<DType><<<dim3(1), dim3(32), 0, stream>>>(val, 1, 1, 32);
    //cudaMemset(val, 0, 32 * sizeof(DType));
    //printf("cudasum Init\n");
    BinaryFullyConnectedPlusCudaSumKernel<DType><<<numBlocks, threadsPerBlock, 0, stream>>>(src, val, Y, src_stride, N);
    cudaStreamSynchronize(stream);
    //printf("kernel 1\n");
    BinaryFullyConnectedPlusCudaSumKernel<DType><<<dim3(1), dim3(32), 0, stream>>>(val, dst, 1, 1, 32);
    //printf("wait\n");
    cudaStreamSynchronize(stream);
    //printf("kernel 2\n");
    //printf("here\n");
    //debugOutput<DType>(src, 10);
    //debugOutput<DType>(dst, 1);
    //cudaStreamSynchronize(stream);
    //printf("3\n");
    cudaFree(val);
}

template <typename DType>
__global__ void BinaryFullyConnectedPlusCudaCopy(DType *src, DType *dst, int Y, int src_stride, int dst_stride, int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) return;
    int idx1 = i / Y * src_stride + i % Y;
    int idx2 = i / Y * dst_stride + i % Y;
    *(dst + idx2) = *(src + idx1);
}
template <typename DType>
__global__ void BinaryFullyConnectedPlusCudaAddTo(DType *src, DType *dst, int Y, int src_stride, int dst_stride, int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) return;
    int idx1 = i / Y * src_stride + i % Y;
    int idx2 = i / Y * dst_stride + i % Y;
    *(dst + idx2) += *(src + idx1);
}
template <typename DType>
__global__ void BinaryFullyConnectedPlusCudaMinus(DType *src, DType *dst, int Y, int src_stride, int dst_stride, int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) return;
    int idx1 = i / Y * src_stride + i % Y;
    int idx2 = i / Y * dst_stride + i % Y;
    *(dst + idx2) -= *(src + idx1);
}
template <typename DType>
__global__ void BinaryFullyConnectedPlusData(DType *data, DType *average, DType *average_val,
                                             int Y, int data_stride, int average_stride, int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) return;
    int idx1 = i / Y * data_stride + i % Y;
    int idx2 = i / Y * average_stride + i % Y;
    DType x = *(data + idx1);
    DType val = (*average_val) / N;
    if (x >= 0)
        *(average + idx2) = val;
    else
        *(average + idx2) = -1. * val;
}

template <typename DType>
void debugOutput(DType *src, int size) {
    static int flag = 0;
    if (flag < 1) {

        flag ++;
        DType *hostval = static_cast<DType*>(malloc(size * sizeof(DType)));
        printf("run here\n");
#include <iostream>
        std::cout << cudaMemcpy(hostval, src, size * sizeof(DType),cudaMemcpyDeviceToHost) << std::endl;
        printf("data\n");
        printf("%f\n", *hostval);
        printf("%d\n", size);
        for (int i = 0; i < min(1000, size); ++ i) {
            printf("%f ", *(hostval + i));
            if ((i + 1) % 8 == 0) printf("\n");
        }
        free(hostval);
    }
}
template <typename DType>
void BinaryFullyConnectedPlusForward(Tensor<gpu, 2, DType> &data,
                                 Tensor<gpu, 2, DType> &wmat,
                                 Tensor<gpu, 2, DType> &out) {
    using namespace mshadow;
    using namespace mshadow::expr;
    //printf("enter FC Forward\n");
    DType *data_ptr = data.dptr_;
    DType *wmat_ptr = wmat.dptr_;
    DType *out_ptr  = out.dptr_;
    int wmat_size   = wmat.shape_[0] * wmat.shape_[1];
    cudaStream_t stream = Stream<gpu>::GetStream(out.stream_);
    dim3 numBlocks((wmat_size + kMaxThreadsPerBlock - 1)/kMaxThreadsPerBlock);
    dim3 threadsPerBlock(kMaxThreadsPerBlock);

    Tensor<gpu, 2, DType> wbmat(wmat.shape_); AllocSpace(&wbmat);wbmat.set_stream(out.stream_);
    Tensor<gpu, 2, DType> average(data.shape_); AllocSpace(&average);average.set_stream(out.stream_);
    Tensor<gpu, 2, DType> residual_data(data.shape_); AllocSpace(&residual_data);residual_data.set_stream(out.stream_);
    Tensor<gpu, 2, DType> temp_res(out.shape_); AllocSpace(&temp_res);temp_res.set_stream(out.stream_);

    DType *average_val;
    auto msg = cudaMalloc(&average_val, sizeof(DType));

    DType *residual_data_ptr = residual_data.dptr_;
    DType *average_ptr = average.dptr_;
    DType *wbmat_ptr = wbmat.dptr_;
    DType *temp_res_ptr = temp_res.dptr_;
    int data_size = data.shape_[0] * data.shape_[1];
    int out_size = out.shape_[0] * out.shape_[1];
    dim3 dataNumBlocks((data_size + kMaxThreadsPerBlock - 1)/kMaxThreadsPerBlock);
    dim3 outNumBlocks((out_size + kMaxThreadsPerBlock - 1)/kMaxThreadsPerBlock);

    BinaryFullyConnectedPlusCudaBinarize<DType><<<numBlocks, threadsPerBlock, 0, stream>>>(wmat_ptr, wbmat_ptr, wmat.shape_[1], wmat.stride_, wbmat.stride_,wmat_size);
    cudaStreamSynchronize(stream);
    //debugOutput<DType>(wbmat_ptr, wmat_size);


    BinaryFullyConnectedPlusCudaCopy<DType><<<dataNumBlocks, threadsPerBlock, 0, stream>>>(data_ptr, residual_data_ptr, data.shape_[1], data.stride_, residual_data.stride_, data_size);
    cudaStreamSynchronize(stream);
    //debugOutput<DType>(data_ptr, data_size);
    //debugOutput<DType>(residual_data_ptr, data_size);
    BinaryFullyConnectedPlusCudaInit<DType><<<dataNumBlocks, threadsPerBlock, 0, stream>>>(average_ptr, data.shape_[1], average.stride_, data_size);
    //cudaMemset(average_ptr, 0, data_size * sizeof(DType));
    cudaStreamSynchronize(stream);
    BinaryFullyConnectedPlusCudaInit<DType><<<outNumBlocks, threadsPerBlock, 0, stream>>>(out_ptr, out.shape_[1], out.stride_, out_size);
    //cudaMemset(out_ptr, 0, out_size * sizeof(DType));
    cudaStreamSynchronize(stream);

    static int debug = 0;
    for (int i = 0; i < 2; ++ i) {
       // printf("Enter FC loop\n");
        //residual_data = residual_data - average;//
        BinaryFullyConnectedPlusCudaMinus<DType><<<dataNumBlocks, threadsPerBlock, 0, stream>>>(average_ptr, residual_data_ptr,
                                                                        data.shape_[1], average.stride_, residual_data.stride_, data_size);
        cudaStreamSynchronize(stream);
       // printf("End minus\n");
        BinaryFullyConnectedPlusCudaInit<DType><<<1, 1, 0, stream>>>(average_val, 1, 1, 1);
        cudaStreamSynchronize(stream);
       // printf("End Init\n");
        BinaryFullyConnectedPlusCudaSum<DType>(residual_data_ptr, average_val, data.shape_[1], residual_data.stride_, data_size, stream);
        //printf("-.-\n");
        cudaStreamSynchronize(stream);
       // printf("End Sum\n");

        BinaryFullyConnectedPlusData<DType><<<dataNumBlocks, threadsPerBlock, 0, stream>>>(residual_data_ptr,
                                                                                            average_ptr, average_val,
                                                                                            data.shape_[1], residual_data.stride_,
                                                                                            average.stride_, data_size);
        cudaStreamSynchronize(stream);
       // printf("End Data\n");
        temp_res = dot(average, wbmat.T());
        cudaStreamSynchronize(stream);
       // printf("End dot\n");
        //printf("%d %d\n", average.shape_[0], average.shape_[1]);
        //printf("%d %d\n", wbmat.shape_[0], wbmat.shape_[1]);
        //CudaAddTo<DType><<<outNumBlocks, threadsPerBlock, 0, stream>>>(temp_res_ptr, out_ptr, out_size);
        //debugOutput<DType>(average_ptr, data_size);

        out = out + temp_res;
        //temp_res = out.T();
        //debugOutput<DType>(temp_res_ptr, out_size);
        cudaStreamSynchronize(stream);
      //  printf("End FC loop\n");
    }

    //free space
    cudaFree(average_val);
    FreeSpace(&wbmat);
    FreeSpace(&average);
    FreeSpace(&residual_data);
    FreeSpace(&temp_res);
   // printf("end FC forward\n");
    //debugOutput<DType>(out_ptr, out_size);
}
template <typename DType>
__global__ void BinaryFullyConnectedPlusBackwardKernel(DType *wbmat, DType *wmat, int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) {
        if (*(wmat + i) >= 0)
            *(wbmat + i) = 1;
        else
            *(wbmat + i) = -1;
    }
}
template <typename DType>
__global__ void BinaryFullyConnectedPlusClip(DType *data, DType *out, int Y, int data_stride, int out_stride, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N) return;
    int idx1 = idx / Y * data_stride + idx % Y;
    DType x = *(data + idx1);
    if (x < -1. || x > 1.) {
        int idx2 = idx / Y * out_stride + idx % Y;
        *(out + idx2) = 0;
    }
}
template <typename DType>
void BinaryFullyConnectedPlusBackward(Tensor<gpu, 2, DType> &data,
                                  Tensor<gpu, 2, DType> &grad,
                                  Tensor<gpu, 2, DType> &wmat,
                                  Tensor<gpu, 2, DType> &gwmat,
                                  Tensor<gpu, 2, DType> &gdata) {
    DType *data_ptr = data.dptr_;
    DType *grad_ptr = grad.dptr_;
    DType *wmat_ptr = wmat.dptr_;
    DType *gwmat_ptr = gwmat.dptr_;
    DType *gdata_ptr = gdata.dptr_;
   //printf("check\n");
    int size = wmat.shape_[0] * wmat.shape_[1];
    cudaStream_t stream = Stream<gpu>::GetStream(gdata.stream_);

    dim3 numBlocks((size + kMaxThreadsPerBlock - 1)/kMaxThreadsPerBlock);
    dim3 threadsPerBlock(kMaxThreadsPerBlock);

    Tensor<gpu, 2, DType> wbmat(wmat.shape_); AllocSpace(&wbmat);wbmat.set_stream(gdata.stream_);
    DType *wbmat_ptr = wbmat.dptr_;
    //printf("BinaryBackward 1\n");
    BinaryFullyConnectedPlusCudaBinarize<DType><<<numBlocks, threadsPerBlock, 0, stream>>>(wmat_ptr, wbmat_ptr, wmat.shape_[1], wmat.stride_, wbmat.stride_, size);
    //printf("BinaryBackward 2\n");
    cudaStreamSynchronize(stream);
    //A to Ab

    DType *average_val;
    auto msg = cudaMalloc(&average_val, sizeof(DType));
    Tensor<gpu, 2, DType> average(data.shape_); AllocSpace(&average);average.set_stream(gdata.stream_);
    Tensor<gpu, 2, DType> residual_data(data.shape_); AllocSpace(&residual_data);residual_data.set_stream(gdata.stream_);
    Tensor<gpu, 2, DType> temp_res(data.shape_); AllocSpace(&temp_res);temp_res.set_stream(gdata.stream_);
    int data_size = data.shape_[0] * data.shape_[1];

    DType *residual_data_ptr = residual_data.dptr_;
    DType *average_ptr = average.dptr_;
    DType *temp_res_ptr = temp_res.dptr_;
    //int data_size = data.shape_[0] * data.shape_[1];
    dim3 dataNumBlocks((data_size + kMaxThreadsPerBlock - 1)/kMaxThreadsPerBlock);

    BinaryFullyConnectedPlusCudaCopy<DType><<<dataNumBlocks, threadsPerBlock, 0, stream>>>
            (data_ptr, residual_data_ptr, data.shape_[1], data.stride_, residual_data.stride_, data_size);
    cudaStreamSynchronize(stream);

    BinaryFullyConnectedPlusCudaInit<DType><<<dataNumBlocks, threadsPerBlock, 0, stream>>>(average_ptr, data.shape_[1], average.stride_, data_size);
    cudaStreamSynchronize(stream);
    //CudaInit<DType><<<outNumBlocks, threadsPerBlock, 0, stream>>>(out_ptr, out.shape_[1], out.stride_, out_size);
    //cudaStreamSynchronize(stream);
    BinaryFullyConnectedPlusCudaInit<DType><<<dataNumBlocks, threadsPerBlock, 0, stream>>>(temp_res_ptr, temp_res.shape_[1], temp_res.stride_, data_size);
    cudaStreamSynchronize(stream);

    for (int i = 0; i < 2; ++ i) {
        BinaryFullyConnectedPlusCudaMinus<DType><<<dataNumBlocks, threadsPerBlock, 0, stream>>>(average_ptr, residual_data_ptr,
                data.shape_[1], average.stride_, residual_data.stride_, data_size);
        cudaStreamSynchronize(stream);
        BinaryFullyConnectedPlusCudaInit<DType><<<1, 1, 0, stream>>>(average_val, 1, 1, 1);
        cudaStreamSynchronize(stream);

        BinaryFullyConnectedPlusCudaSum<DType>(residual_data_ptr, average_val, data.shape_[1], residual_data.stride_, data_size, stream);
        cudaStreamSynchronize(stream);
        //debugOutput<DType>(average_val, 1);

        BinaryFullyConnectedPlusData<DType><<<dataNumBlocks, threadsPerBlock, 0, stream>>>(residual_data_ptr,
                average_ptr, average_val,
                data.shape_[1], residual_data.stride_,
                average.stride_, data_size);
        cudaStreamSynchronize(stream);

        temp_res = temp_res + average;
        cudaStreamSynchronize(stream);

    }
    //end of A to Ab
    //printf("check1\n");
    gwmat = dot(grad.T(), data);
    //cudaStreamSynchronize(stream);
   // printf("check2\n");
    gdata = dot(grad, wbmat);
    cudaStreamSynchronize(stream);
   // printf("check3\n");
    //pass paragrams with Tensor type failed!
    BinaryFullyConnectedPlusClip<DType><<<numBlocks, threadsPerBlock, 0, stream>>>
            (data_ptr, gdata_ptr, data.shape_[1], data.stride_, gdata.stride_, data_size);
    cudaStreamSynchronize(stream);
  //  printf("check4\n");
    FreeSpace(&wbmat);

    //printf("check5\n");
    cudaFree(average_val);
    //printf("check6\n");
    FreeSpace(&average);
    //printf("check7\n");
    FreeSpace(&residual_data);
    //printf("check8\n");
    FreeSpace(&temp_res);
    //printf("check9\n");

}
}//cuda
template <typename DType>
void BinaryFullyConnectedPlusForward(Tensor<gpu, 2, DType> &data,
                                 Tensor<gpu, 2, DType> &wmat,
                                 Tensor<gpu, 2, DType> &out) {
    cuda::BinaryFullyConnectedPlusForward(data, wmat, out);
}
template <typename DType>
void BinaryFullyConnectedPlusBackward(Tensor<gpu, 2, DType> &data,
                                  Tensor<gpu, 2, DType> &grad,
                                  Tensor<gpu, 2, DType> &wmat,
                                  Tensor<gpu, 2, DType> &gwmat,
                                  Tensor<gpu, 2, DType> &gdata) {
    cuda::BinaryFullyConnectedPlusBackward(data, grad, wmat, gwmat, gdata);
}
}


namespace mxnet {
namespace op {
template<>
Operator* CreateOp<gpu>(BinaryFullyConnectedPlusParam param, int dtype) {
    Operator *op = NULL;
    MSHADOW_REAL_TYPE_SWITCH(dtype, DType, {
            op = new BinaryFullyConnectedPlusOp<gpu, DType>(param);
    })
    return op;
}
}  // namespace op
}  // namespace mxnet
