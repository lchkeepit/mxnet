/*!
 * Copyright (c) 2015 by Contributors
 * \file fully_connected.cc
 * \brief fully connect operator
*/
#include "./binary_fully_connected.h"
#if MXNET_USE_MKL2017 == 1
#include <mkl_memory.h>
#include "./mkl/mkl_memory-inl.h"
#include "./mkl/mkl_fully_connected-inl.h"
#endif  // MXNET_USE_MKL2017

namespace mshadow {
template <typename DType>
void debugOutput(DType *src, int size) {
    static int flag = 0;
    if (flag == 0) {
        flag ++;
#include <iostream>
        printf("data\n");
        for (int i = 0; i < std::min(1000, size); ++ i) {
            printf("%f ", *(src + i));
            if ((i + 1) % 10 == 0) printf("\n");
        }
    }
}
template<typename DType>
void BinaryFullyConnectedForward(Tensor<cpu, 2, DType> &data,
                                Tensor<cpu, 2, DType> &wmat,
                                Tensor<cpu, 2, DType> &out) {
        using namespace mshadow::expr;
        Tensor<cpu, 2, DType> wbmat(wmat.shape_);
        AllocSpace(&wbmat);
        debugOutput<DType>(data.dptr_, data.shape_[0] * data.shape_[1]);
        for (int i = 0; i < wmat.shape_[0]; ++ i){
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
template<typename DType>
void BinaryFullyConnectedBackward(Tensor<cpu, 2, DType> &data,
                                Tensor<cpu, 2, DType> &grad,
                                Tensor<cpu, 2, DType> &wmat,
                                Tensor<cpu, 2, DType> &gwmat,
                                Tensor<cpu, 2, DType> &gdata) {
        gwmat = dot(grad.T(), data);
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
        //printf("binary_fully_connected 1\n");
        gdata = dot(grad, wbmat);
        //printf("binary_fully_connected 1\n");
        FreeSpace(&wbmat);
    }

}

namespace mxnet {
namespace op {
template<>
Operator* CreateOp<cpu>(BinaryFullyConnectedParam param, int dtype) {
    Operator *op = NULL;
#if MXNET_USE_MKL2017 == 1
    switch (dtype) {
  case mshadow::kFloat32:
    return new MKLFullyConnectedOp<cpu, float>(param);
  case mshadow::kFloat64:
    return new MKLFullyConnectedOp<cpu, double>(param);
  default:
    if (enableMKLWarnGenerated())
      LOG(INFO) << MKLFullyConnectedOp<cpu, float>::getName() << " Skip MKL optimization";
    break;
  }
#endif
    switch (dtype) {
        case mshadow::kFloat32:
            op = new BinaryFullyConnectedOp<cpu, float>(param);
            break;
        case mshadow::kFloat64:
            op = new BinaryFullyConnectedOp<cpu, double>(param);
            break;
        case mshadow::kFloat16:
            LOG(FATAL) << "float16 fully connected layer is currently"
                    "only supported by CuDNN version.";
            break;
        default:
            LOG(FATAL) << "Unsupported type " << dtype;
    }

    return op;
}

// DO_BIND_DISPATCH comes from operator_common.h
Operator *BinaryFullyConnectedProp::CreateOperatorEx(Context ctx, std::vector<TShape> *in_shape,
                                               std::vector<int> *in_type) const {
    std::vector<TShape> out_shape, aux_shape;
    std::vector<int> out_type, aux_type;
    CHECK(InferType(in_type, &out_type, &aux_type));
    CHECK(InferShape(in_shape, &out_shape, &aux_shape));
    DO_BIND_DISPATCH(CreateOp, param_, (*in_type)[0]);
}

DMLC_REGISTER_PARAMETER(BinaryFullyConnectedParam);

MXNET_REGISTER_OP_PROPERTY(BinaryFullyConnected, BinaryFullyConnectedProp)
.describe(R"(Apply matrix multiplication to input then add a bias.
It maps the input of shape `(batch_size, input_dim)` to the shape of
`(batch_size, num_hidden)`. Learnable parameters include the weights
of the linear transform and an optional bias vector.)")
.add_argument("data", "Symbol", "Input data to the FullyConnectedOp.")
.add_argument("weight", "Symbol", "Weight matrix.")
.add_argument("bias", "Symbol", "Bias parameter.")
.add_arguments(BinaryFullyConnectedParam::__FIELDS__());
}  // namespace op
}  // namespace mxnet
