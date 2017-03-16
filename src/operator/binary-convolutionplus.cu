/*!
 * created by chenhao li
*/

#include "./binary-convolutionplus.h"
#include <vector>
#include "cuda.h"
#include "../common/cuda_utils.h"

namespace mshadow {
namespace cuda {
template <typename DType>
void debugOutput(DType *src, int size) {
    static int flag = 0;
    if (flag < 10) {
        flag ++;
        printf("convolution\n");
        DType *hostval = static_cast<DType*>(malloc(size * sizeof(DType)));
        //printf("run here\n");
#include <iostream>
        //std::cout << cudaMemcpy(hostval, src, size * sizeof(DType),cudaMemcpyDeviceToHost) << std::endl;
        //printf("data\n");
        //printf("%f\n", *hostval);
        //printf("%d\n", size);
        for (int i = 0; i < min(1024, size); ++ i) {
            printf("%f ", *(hostval + i));
            if ((i + 1) % 8 == 0) printf("\n");
        }
        puts("");
        free(hostval);
    }
}
template <typename DType>
__global__ void BinaryConvolutionPlusCudaBinarize(DType *src, DType *dst, int Y, int src_stride, int dst_stride, int N) {
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
__global__ void BinaryConvolutionPlusCudaInit(DType *src, int Y, int src_stride, int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) return;
    int idx = i / Y * src_stride + i % Y;

    *(src + idx) = 0;
}
template<typename DType>
__global__ void BinaryConvolutionPlusCudaSumKernel(DType *src, DType *dst, int Y, int src_stride, int N) {
    __shared__ DType box[1024];
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int threadid = threadIdx.x;
    DType temp = 0;
    while (tid < N) {
        DType x = src[tid / Y * src_stride + tid % Y];
        if (x < 0) x *= -1.;
        temp += x;
        tid += blockDim.x * gridDim.x;
    }
    box[threadid] = temp;

    __syncthreads();

    int i = blockDim.x / 2;
    while (i != 0) {
        if (threadid < i) {
            box[threadid] += box[threadid + i];
        }
        __syncthreads();
        i /= 2;
    }
    if (threadid == 0) {
        dst[blockIdx.x] = box[0];

    }
}
template <typename DType>
void BinaryConvolutionPlusCudaSum(DType *src, DType *dst, int Y, int src_stride, int N, cudaStream_t &stream) {
    //printf("convolution cudasum\n");
    int threads = 1024;
    dim3 numBlocks(32);
    dim3 threadsPerBlock(threads);

    DType *val;
    auto msg = cudaMalloc(&val, 32 * sizeof(DType));
    BinaryConvolutionPlusCudaInit<DType><<<1, 32, 0, stream>>>(val, 1, 1, 32);
    cudaStreamSynchronize(stream);
    //printf("kernel 1\n");
    BinaryConvolutionPlusCudaSumKernel<DType><<<32, 1024, 0, stream>>>(src, val, Y, src_stride, N);
    cudaStreamSynchronize(stream);
    //printf("kernel 2\n");
    BinaryConvolutionPlusCudaSumKernel<DType><<<1, 32, 0, stream>>>(val, dst, 1, 1, 32);
    cudaStreamSynchronize(stream);
    //debugOutput<DType>(dst, 1);
    //cudaStreamSynchronize(stream);
    //printf("3\n");
    cudaFree(val);
}
template<typename DType>
__global__ void BinaryConvolutionPlusCudaSumTo(DType *src, DType *dst, int Y, int src_stride, int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) return;
    int idx1 = i / Y * src_stride + i % Y;

    DType x = *(src + idx1);
    if (x < 0) x *= -1.;
    atomicAdd(dst, x); // too slow
    //*dst += x;
}
template <typename DType>
__global__ void BinaryConvolutionPlusCudaCopy(DType *src, DType *dst, int Y, int src_stride, int dst_stride, int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) return;
    while (i < N) {
        int idx1 = i / Y * src_stride + i % Y;
        int idx2 = i / Y * dst_stride + i % Y;
        *(dst + idx2) = *(src + idx1);
        i += blockDim.x * gridDim.x;
    }

}
template <typename DType>
__global__ void BinaryConvolutionPlusCudaAddTo(DType *src, DType *dst, int Y, int src_stride, int dst_stride, int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) return;
    int idx1 = i / Y * src_stride + i % Y;
    int idx2 = i / Y * dst_stride + i % Y;
    *(dst + idx2) += *(src + idx1);
}
template <typename DType>
__global__ void BinaryConvolutionPlusCudaMinus(DType *src, DType *dst, int Y, int src_stride, int dst_stride, int N) {
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
void BinaryConvolutionPlusApproximate(Tensor<gpu, 2, DType> data) {
    cudaStream_t stream = Stream<gpu>::GetStream(data.stream_);

    Tensor<gpu, 2, DType> average(data.shape_); AllocSpace(&average);average.set_stream(data.stream_);
    Tensor<gpu, 2, DType> residual_data(data.shape_); AllocSpace(&residual_data);residual_data.set_stream(data.stream_);
    Tensor<gpu, 2, DType> temp_res(data.shape_); AllocSpace(&temp_res);temp_res.set_stream(data.stream_);

    DType *average_val;
    auto msg = cudaMalloc(&average_val, sizeof(DType));

    DType *residual_data_ptr = residual_data.dptr_;
    DType *average_ptr = average.dptr_;
    DType *temp_res_ptr = temp_res.dptr_;
    DType *data_ptr = data.dptr_;
    int data_size = data.shape_[0] * data.shape_[1];

    dim3 dataNumBlocks((data_size + kMaxThreadsPerBlock - 1)/kMaxThreadsPerBlock);
    dim3 threadsPerBlock(kMaxThreadsPerBlock);

    BinaryConvolutionPlusCudaCopy<DType><<<dataNumBlocks, threadsPerBlock, 0, stream>>>
            (data_ptr, residual_data_ptr, data.shape_[1], data.stride_, residual_data.stride_, data_size);
    cudaStreamSynchronize(stream);
    BinaryConvolutionPlusCudaInit<DType><<<dataNumBlocks, threadsPerBlock, 0, stream>>>
            (average_ptr, data.shape_[1], average.stride_, data_size);
    cudaStreamSynchronize(stream);
    BinaryConvolutionPlusCudaInit<DType><<<dataNumBlocks, threadsPerBlock, 0, stream>>>
            (temp_res_ptr, temp_res.shape_[1], temp_res.stride_, data_size);
    cudaStreamSynchronize(stream);

    for (int i = 0; i < 2; ++ i) {
        BinaryConvolutionPlusCudaMinus<DType><<<dataNumBlocks, threadsPerBlock, 0, stream>>>(average_ptr, residual_data_ptr,
                data.shape_[1], average.stride_, residual_data.stride_, data_size);
        cudaStreamSynchronize(stream);
        BinaryConvolutionPlusCudaInit<DType><<<1, 1, 0, stream>>>(average_val, 1, 1, 1);
        cudaStreamSynchronize(stream);
        BinaryConvolutionPlusCudaSum<DType>(residual_data_ptr, average_val, data.shape_[1], residual_data.stride_, data_size, stream);
        cudaStreamSynchronize(stream);

        BinaryFullyConnectedPlusData<DType><<<dataNumBlocks, threadsPerBlock, 0, stream>>>(residual_data_ptr,
                average_ptr, average_val,
                data.shape_[1], residual_data.stride_,
                average.stride_, data_size);
        cudaStreamSynchronize(stream);

        temp_res = temp_res + average;
        cudaStreamSynchronize(stream);
    }
    BinaryConvolutionPlusCudaCopy<DType><<<dataNumBlocks, threadsPerBlock, 0, stream>>>
            (temp_res_ptr, data_ptr, data.shape_[1], temp_res.stride_, data.stride_, data_size);
    cudaStreamSynchronize(stream);

    //freespace
    cudaFree(average_val);
    FreeSpace(&average);
    FreeSpace(&residual_data);
    FreeSpace(&temp_res);
}
template <typename DType>
void BinaryConvolutionPlusForward(Tensor<gpu, 2, DType> &wmat,
                                     Tensor<gpu, 2, DType> &data,
                                     Tensor<gpu, 2, DType> &out) {
    using namespace mshadow;
    using namespace mshadow::expr;
    //printf("enter convolution Forward\n");
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

    BinaryConvolutionPlusCudaBinarize<DType><<<numBlocks, threadsPerBlock, 0, stream>>>(wmat_ptr, wbmat_ptr, wmat.shape_[1], wmat.stride_, wbmat.stride_,wmat_size);
    cudaStreamSynchronize(stream);
    //debugOutput<DType>(wbmat_ptr, wmat_size);
    dim3 dataNumBlocks((data_size + kMaxThreadsPerBlock - 1)/kMaxThreadsPerBlock);
    dim3 outNumBlocks((out_size + kMaxThreadsPerBlock - 1)/kMaxThreadsPerBlock);

    BinaryConvolutionPlusCudaCopy<DType><<<dataNumBlocks, threadsPerBlock, 0, stream>>>(data_ptr, residual_data_ptr, data.shape_[1], data.stride_, residual_data.stride_, data_size);
    cudaStreamSynchronize(stream);
    //debugOutput<DType>(data_ptr, data_size);
    //debugOutput<DType>(residual_data_ptr, data_size);
    BinaryConvolutionPlusCudaInit<DType><<<dataNumBlocks, threadsPerBlock, 0, stream>>>(average_ptr, data.shape_[1], average.stride_, data_size);
    //cudaMemset(average_ptr, 0, data_size * sizeof(DType));
    cudaStreamSynchronize(stream);
    BinaryConvolutionPlusCudaInit<DType><<<outNumBlocks, threadsPerBlock, 0, stream>>>(out_ptr, out.shape_[1], out.stride_, out_size);
    //cudaMemset(out_ptr, 0, out_size * sizeof(DType));
    cudaStreamSynchronize(stream);

    static int debug = 0;
    for (int i = 0; i < 2; ++ i) {
        //printf("loop %d\n", i);
        BinaryConvolutionPlusCudaMinus<DType><<<dataNumBlocks, threadsPerBlock, 0, stream>>>(average_ptr, residual_data_ptr,
                data.shape_[1], average.stride_, residual_data.stride_, data_size);
        cudaStreamSynchronize(stream);
        BinaryConvolutionPlusCudaInit<DType><<<1, 1, 0, stream>>>(average_val, 1, 1, 1);
        //cudaMemset(average_val, 0, 1 * sizeof(DType));

        cudaStreamSynchronize(stream);

        //printf("sum begin\n");
        BinaryConvolutionPlusCudaSum<DType>(residual_data_ptr, average_val, data.shape_[1], residual_data.stride_, data_size, stream);
        cudaStreamSynchronize(stream);
        //printf("sum end\n");

        BinaryFullyConnectedPlusData<DType><<<dataNumBlocks, threadsPerBlock, 0, stream>>>(residual_data_ptr,
                average_ptr, average_val,
                data.shape_[1], residual_data.stride_,
                average.stride_, data_size);
        cudaStreamSynchronize(stream);

        temp_res = dot(wbmat, average);
        cudaStreamSynchronize(stream);

        out = out + temp_res;
        cudaStreamSynchronize(stream);

    }
    //out = dot(wbmat, data);
    //debugOutput<DType>(out_ptr, out_size);
    //debugOutput<DType>(wmat_ptr, wmat_size);
    //printf("count");
    //free space
    cudaFree(average_val);
    FreeSpace(&wbmat);
    FreeSpace(&average);
    FreeSpace(&residual_data);
    FreeSpace(&temp_res);
    //debugOutput<DType>(out_ptr, out_size);
}
template <typename DType>
__global__ void BinaryConvolutionPlusClip(DType *data, DType *out, int Y, int data_stride, int out_stride, int N) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    while (idx < N) {
        int idx1 = idx / Y * data_stride + idx % Y;
        int idx2 = idx / Y * out_stride + idx % Y;
        if (data[idx1] < -1. || data[idx1] > 1.) {
            out[idx2] = 0;
        }
        idx += blockDim.x * gridDim.x;
    }
}
template <typename DType>
void BinaryConvolutionPlusBackward(Tensor<gpu, 2, DType> wmat, Tensor<gpu, 2, DType> grad, Tensor<gpu, 2, DType> out) {
    Tensor<gpu, 2, DType> wbmat(wmat.shape_);wbmat.set_stream(out.stream_);
    AllocSpace(&wbmat);

    int size = wmat.shape_[0] * wmat.shape_[1];
    cudaStream_t stream = Stream<gpu>::GetStream(out.stream_);
    dim3 numBlocks((size + kMaxThreadsPerBlock - 1)/kMaxThreadsPerBlock);
    dim3 threadsPerBlock(kMaxThreadsPerBlock);
    DType *wmat_dptr = wmat.dptr_;
    DType *wbmat_dptr = wbmat.dptr_;

    BinaryConvolutionPlusCudaBinarize<DType><<<numBlocks, threadsPerBlock, 0, stream>>>(wmat_dptr, wbmat_dptr, wmat.shape_[1], wmat.stride_, wbmat.stride_, size);
    cudaStreamSynchronize(stream);

    int data_size = out.shape_[0] * out.shape_[1];
    Tensor<gpu, 2, DType> data(out.shape_); AllocSpace(&data); data.set_stream(out.stream_);
    BinaryConvolutionPlusCudaCopy<DType><<<numBlocks, threadsPerBlock, 0, stream>>>
            (out.dptr_, data.dptr_, data.shape_[1], out.stride_, data.stride_, data_size);
    cudaStreamSynchronize(stream);

    out = dot(wbmat.T(), grad);
    BinaryConvolutionPlusClip<DType><<<numBlocks, threadsPerBlock, 0, stream>>>
            (data.dptr_, out.dptr_, data.shape_[1], data.stride_, out.stride_, data_size);
    FreeSpace(&wbmat);
    FreeSpace(&data);
    //
}
}

template <typename DType>
void BinaryConvolutionPlusForward(Tensor<gpu, 2, DType> wmat, Tensor<gpu, 2, DType> data, Tensor<gpu, 2, DType> out) {
    cuda::BinaryConvolutionPlusForward(wmat, data, out);
}
template <typename DType>
void BinaryConvolutionPlusApproximate(Tensor<gpu, 2, DType> data) {
    cuda::BinaryConvolutionPlusApproximate(data);
}
template <typename DType>
void BinaryConvolutionPlusBackward(Tensor<gpu, 2, DType> wmat, Tensor<gpu, 2, DType> grad, Tensor<gpu, 2, DType> out) {
    cuda::BinaryConvolutionPlusBackward(wmat, grad, out);
}
}


namespace mxnet {
namespace op {
template<>
Operator* CreateOp<gpu>(BinaryConvolutionPlusParam param, int dtype,
                        std::vector<TShape> *in_shape,
                        std::vector<TShape> *out_shape,
                        Context ctx) {
    Operator *op = NULL;
    MSHADOW_REAL_TYPE_SWITCH(dtype, DType, {
            op = new BinaryConvolutionPlusOp<gpu, DType>(param);
    })
    return op;
}

}  // namespace op
}  // namespace mxnet

