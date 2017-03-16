/*!
 * Copyright (c) 2015 by Contributors
 * \file convolution.cc
 * \brief
 * \author Bing Xu
*/

#include "./binary-convolution.h"
#if MXNET_USE_MKL2017 == 1
#include <mkl_memory.h>
#include "./mkl/mkl_memory-inl.h"
#include "./mkl/mkl_convolution-inl.h"
#endif  // MXNET_USE_MKL2017
#if MXNET_USE_NNPACK == 1
#include "./nnpack/nnpack_convolution-inl.h"
#endif  // MXNET_USE_NNPACK

namespace mshadow {
template <typename DType>
void BinaryConvolutionForward(Tensor<cpu, 2, DType> wmat, Tensor<cpu, 2, DType> &data, Tensor<cpu, 2, DType> out) {
    using namespace mshadow::expr;
    Tensor<cpu, 2, DType> wbmat(wmat.shape_);
    AllocSpace(&wbmat);

    for (int i = 0; i < wmat.shape_[0]; ++ i) {
        for (int j = 0; j < wmat.shape_[1]; ++ j) {
            DType x = *(wmat.dptr_ + i * wmat.stride_ + j);

            if (x >= 0)
                *(wbmat.dptr_ + i * wbmat.stride_ + j) = 1;
            else
                *(wbmat.dptr_ + i * wbmat.stride_ + j) = -1;
        }
    }
    out = dot(data, wbmat.T());
    FreeSpace(&wbmat);
}
template <typename DType>
void BinaryConvolutionBackward(Tensor<cpu, 2, DType> wmat, Tensor<cpu, 2, DType> data, Tensor<cpu, 2, DType> out) {
    using namespace mshadow::expr;
    Tensor<cpu, 2, DType> wbmat(wmat.shape_);
    AllocSpace(&wbmat);

    for (int i = 0; i < wmat.shape_[0]; ++ i) {
        for (int j = 0; j < wmat.shape_[1]; ++ j) {
            DType x = *(wmat.dptr_ + i * wmat.stride_ + j);

            if (x >= 0)
                *(wbmat.dptr_ + i * wbmat.stride_ + j) = 1;
            else
                *(wbmat.dptr_ + i * wbmat.stride_ + j) = -1;
        }
    }
    out = dot(data, wbmat.T());
    FreeSpace(&wbmat);
}
}

namespace mxnet {
namespace op {
DMLC_REGISTER_PARAMETER(BinaryConvolutionParam);

template<>
Operator* CreateOp<cpu>(BinaryConvolutionParam param, int dtype,
                        std::vector<TShape> *in_shape,
                        std::vector<TShape> *out_shape,
                        Context ctx) {
    Operator *op = NULL;

    MSHADOW_REAL_TYPE_SWITCH(dtype, DType, {
            op = new BinaryConvolutionOp<cpu, DType>(param);
    })
    return op;
}

// DO_BIND_DISPATCH comes from operator_common.h
Operator *BinaryConvolutionProp::CreateOperatorEx(Context ctx,
                                            std::vector<TShape> *in_shape,
                                            std::vector<int> *in_type) const {
    std::vector<TShape> out_shape, aux_shape;
    std::vector<int> out_type, aux_type;
    CHECK(InferType(in_type, &out_type, &aux_type));
    CHECK(InferShape(in_shape, &out_shape, &aux_shape));
    DO_BIND_DISPATCH(CreateOp, param_, (*in_type)[0], in_shape, &out_shape, ctx);
}

MXNET_REGISTER_OP_PROPERTY(BinaryConvolution, BinaryConvolutionProp)
.add_argument("data", "Symbol", "Input data to the ConvolutionOp.")
.add_argument("weight", "Symbol", "Weight matrix.")
.add_argument("bias", "Symbol", "Bias parameter.")
.add_arguments(BinaryConvolutionParam::__FIELDS__())
.describe("Apply convolution to input then add a bias.");

}  // namespace op
}  // namespace mxnet

