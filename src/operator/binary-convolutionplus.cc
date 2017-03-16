/*!
 * Copyright (c) 2015 by Contributors
 * \file convolution.cc
 * \brief
 * \author Bing Xu
*/

#include "./binary-convolutionplus.h"
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
void BinaryConvolutionPlusForward(Tensor<cpu, 2, DType> wmat, Tensor<cpu, 2, DType> &data, Tensor<cpu, 2, DType> out) {
    using namespace mshadow::expr;
    Tensor<cpu, 2, DType> wbmat(wmat.shape_);
    AllocSpace(&wbmat);
    Tensor<cpu, 2, DType> average(data.shape_);
    AllocSpace(&average);
    Tensor<cpu, 2, DType> residual_data(data.shape_);
    AllocSpace(&residual_data);
    Tensor<cpu, 2, DType> temp_res(out.shape_);
    AllocSpace(&temp_res);
    DType average_val = 0;
    int size = data.shape_[0] * data.shape_[1];
    out = 0;
    for (int i = 0; i < wmat.shape_[0]; ++ i){
        for (int j = 0; j < wmat.shape_[1]; ++ j) {
            DType x = *(wmat.dptr_ + i * wmat.stride_ + j);

            if (x >= 0)
                *(wbmat.dptr_ + i * wbmat.stride_ + j) = 1;
            else
                *(wbmat.dptr_ + i * wbmat.stride_ + j) = -1;
        }
    }
    //debugOutput<DType>(wbmat.dptr_, wmat.shape_[0] * wmat.shape_[1]);
    for (int i = 0; i < data.shape_[0]; ++ i) {
        for (int j = 0; j < data.shape_[1]; ++ j) {
            DType x = *(data.dptr_ + i * data.stride_ + j);
            *(residual_data.dptr_ + i * residual_data.stride_ + j) = x;
        }
    }
    //debugOutput<DType>(data.dptr_, data.shape_[0] * data.shape_[1]);
    //debugOutput<DType>(residual_data.dptr_, data.shape_[0] * data.shape_[1]);
    for (int m = 0; m < 2; ++ m) {
        residual_data = residual_data - average;
        average_val = 0;
        for (int i = 0; i < data.shape_[0]; ++ i) {
            for (int j = 0; j < data.shape_[1]; ++ j) {
                DType x = *(residual_data.dptr_ + i * residual_data.stride_ + j);
                if (x < 0) x *= -1.;
                average_val += x;
            }
        }

        //debugOutput<DType>(&average_val, 1);
        average_val /= static_cast<float>(size);
        for (int i = 0; i < data.shape_[0]; ++ i) {
            for (int j = 0; j < data.shape_[1]; ++ j) {
                int idx = i * residual_data.stride_ + j;
                int idx2 = i * average.stride_ + j;
                if (*(residual_data.dptr_ + idx) >= 0)
                    *(average.dptr_ + idx2) = average_val;
                else
                    *(average.dptr_ + idx2) = -1. * average_val;
            }
        }

        temp_res = dot(average, wbmat.T());

        out = out + temp_res;
        //out = dot(data, wbmat.T());
    }
    //out = dot(data, wbmat.T());
    //debugOutput<DType>(out.dptr_, out.shape_[0] * out.shape_[1]);

    //FreeSpace
    FreeSpace(&temp_res);
    FreeSpace(&wbmat);
    FreeSpace(&average);
    FreeSpace(&residual_data);
}
template <typename DType>
void BinaryConvolutionPlusApproximate(Tensor<cpu, 2, DType> data) {
    //not implemented
}
template <typename DType>
void BinaryConvolutionPlusBackward(Tensor<cpu, 2, DType> wmat, Tensor<cpu, 2, DType> data, Tensor<cpu, 2, DType> out) {
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
    //out = dot(data, wbmat.T());
    out = dot(wbmat.T(), data);
    FreeSpace(&wbmat);
}
}

namespace mxnet {
namespace op {
DMLC_REGISTER_PARAMETER(BinaryConvolutionPlusParam);

template<>
Operator* CreateOp<cpu>(BinaryConvolutionPlusParam param, int dtype,
                        std::vector<TShape> *in_shape,
                        std::vector<TShape> *out_shape,
                        Context ctx) {
    Operator *op = NULL;

    MSHADOW_REAL_TYPE_SWITCH(dtype, DType, {
            op = new BinaryConvolutionPlusOp<cpu, DType>(param);
    })
    return op;
}

// DO_BIND_DISPATCH comes from operator_common.h
Operator *BinaryConvolutionPlusProp::CreateOperatorEx(Context ctx,
                                                  std::vector<TShape> *in_shape,
                                                  std::vector<int> *in_type) const {
    std::vector<TShape> out_shape, aux_shape;
    std::vector<int> out_type, aux_type;
    CHECK(InferType(in_type, &out_type, &aux_type));
    CHECK(InferShape(in_shape, &out_shape, &aux_shape));
    DO_BIND_DISPATCH(CreateOp, param_, (*in_type)[0], in_shape, &out_shape, ctx);
}

MXNET_REGISTER_OP_PROPERTY(BinaryConvolutionPlus, BinaryConvolutionPlusProp)
.add_argument("data", "Symbol", "Input data to the ConvolutionOp.")
.add_argument("weight", "Symbol", "Weight matrix.")
.add_argument("bias", "Symbol", "Bias parameter.")
.add_arguments(BinaryConvolutionPlusParam::__FIELDS__())
.describe("Apply convolution to input then add a bias.");

}  // namespace op
}  // namespace mxnet

