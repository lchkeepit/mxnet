/*!
 * Copyright (c) 2015 by Contributors
 * \file fully_connected.cc
 * \brief fully connect operator
*/
#include "./binary-fullyconnectedplus.h"
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
        Tensor <cpu, 2, DType> a(Shape2(3, 1)); AllocSpace(&a);
        Tensor <cpu, 2, DType> b(Shape2(1, 3)); AllocSpace(&b);
        Tensor <cpu, 2, DType> c(Shape2(3, 3)); AllocSpace(&c);
        Tensor <cpu, 2, DType> d(Shape2(1, 1)); AllocSpace(&d);
        printf("%d\n", a.stride_);
        *(a.dptr_ + 0) = 1; *(a.dptr_ + a.stride_) = 2; *(a.dptr_ + 2 * a.stride_) = 3;
        *(b.dptr_ + 0) = 1; *(b.dptr_ + 1) = 2; *(b.dptr_ + 2) = 3;
        c = dot(a, b);
        d = dot(b, a);
        printf("%f\n", *d.dptr_);
        for (int i = 0; i < 3; ++ i) {
            printf("%f ", *(a.dptr_ + i * a.stride_));
        }
        puts("");
        for (int i = 0; i < 3; ++ i) {
            printf("%f ", *(b.dptr_ + i));
        }
        puts("");
        for (int i = 0; i < 3; ++ i) {
            for (int j = 0; j < 3; ++ j){
                printf("%f ", *(c.dptr_ + i * c.stride_ + j));
            }
            printf("\n");
        }
        FreeSpace(&a);
        FreeSpace(&b);
        FreeSpace(&c);
        FreeSpace(&d);
        flag ++;
#include <iostream>
        printf("data\n");

        printf("%f\n", *src);
        printf("%d\n", size);
        for (int i = 0; i < std::min(1000, size); ++ i) {
            DType x = *(src + i);
            printf("%f ", x);
            if ((i + 1) % 8 == 0) printf("\n");
        }
    }
}
template<typename DType>
void BinaryFullyConnectedPlusForward(Tensor<cpu, 2, DType> &data,
                                 Tensor<cpu, 2, DType> &wmat,
                                 Tensor<cpu, 2, DType> &out) {
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
    static int debug = 0;
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
    for (int i = 0; i < average.shape_[0]; ++ i) {
        for (int j = 0; j < average.shape_[1]; ++ j) {
            *(average.dptr_ + i * average.stride_ + j) = 0;
        }
    }
    //debugOutput<DType>(data.dptr_, data.shape_[0] * data.shape_[1]);
    //debugOutput<DType>(residual_data.dptr_, data.shape_[0] * data.shape_[1]);
    for (int m = 0; m < 2; ++ m) {
        //residual_data = residual_data - average;
        for (int i = 0; i < average.shape_[0]; ++ i) {
            for (int j = 0; j < average.shape_[1]; ++ j) {
                *(residual_data.dptr_ + i * residual_data.stride_ + j) -= *(average.dptr_ + i * average.stride_ + j);
            }
        }
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
        //out = dot(average, wbmat.T());
        //debugOutput<DType>(wmat.dptr_, wbmat.shape_[0] * wbmat.shape_[1]);
        temp_res = dot(average, wbmat.T());

        //debugOutput<DType>(average.dptr_, average.shape_[0] * average.shape_[1]);
        //debugOutput<DType>(temp_res.dptr_, temp_res.shape_[0] * temp_res.shape_[1]);
        out = out + temp_res;
        //printf("%d %d\n", average.shape_[0], average.shape_[1]);
        //printf("%d %d\n", wbmat.shape_[0], wbmat.shape_[1]);
        //debugOutput<DType>(out.dptr_, out.shape_[0] * out.shape_[1]);

        //out = dot(data, wbmat.T());
    }
    //out = dot(data, wbmat.T());
    debugOutput<DType>(out.dptr_, out.shape_[0] * out.shape_[1]);
    //FreeSpace
    FreeSpace(&temp_res);
    FreeSpace(&wbmat);
    FreeSpace(&average);
    FreeSpace(&residual_data);
}
template<typename DType>
void BinaryFullyConnectedPlusBackward(Tensor<cpu, 2, DType> &data,
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
Operator* CreateOp<cpu>(BinaryFullyConnectedPlusParam param, int dtype) {
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
            op = new BinaryFullyConnectedPlusOp<cpu, float>(param);
            break;
        case mshadow::kFloat64:
            op = new BinaryFullyConnectedPlusOp<cpu, double>(param);
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
Operator *BinaryFullyConnectedPlusProp::CreateOperatorEx(Context ctx, std::vector<TShape> *in_shape,
                                                     std::vector<int> *in_type) const {
    std::vector<TShape> out_shape, aux_shape;
    std::vector<int> out_type, aux_type;
    CHECK(InferType(in_type, &out_type, &aux_type));
    CHECK(InferShape(in_shape, &out_shape, &aux_shape));
    DO_BIND_DISPATCH(CreateOp, param_, (*in_type)[0]);
}

DMLC_REGISTER_PARAMETER(BinaryFullyConnectedPlusParam);

MXNET_REGISTER_OP_PROPERTY(BinaryFullyConnectedPlus, BinaryFullyConnectedPlusProp)
.describe(R"(Apply matrix multiplication to input then add a bias.
It maps the input of shape `(batch_size, input_dim)` to the shape of
`(batch_size, num_hidden)`. Learnable parameters include the weights
of the linear transform and an optional bias vector.)")
.add_argument("data", "Symbol", "Input data to the FullyConnectedOp.")
.add_argument("weight", "Symbol", "Weight matrix.")
.add_argument("bias", "Symbol", "Bias parameter.")
.add_arguments(BinaryFullyConnectedPlusParam::__FIELDS__());
}  // namespace op
}  // namespace mxnet
