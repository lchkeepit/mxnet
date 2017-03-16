/*!
* created by chenhao li
*/

#include "./binary-activation.h"
#include <cstdlib>
#if MXNET_USE_MKL2017 == 1
#include <mkl_memory.h>
#include "./mkl/mkl_memory-inl.h"
#include "./mkl/mkl_convolution-inl.h"
#endif  // MXNET_USE_MKL2017
#if MXNET_USE_NNPACK == 1
#include "./nnpack/nnpack_convolution-inl.h"
#endif  // MXNET_USE_NNPACK

namespace mshadow {
/*template<typename DType>
void BinaryActivationForward(Tensor<cpu, 2, DType> &data, Tensor<cpu, 2, DType> &out) {
        Tensor<cpu, 2, DType> ShipTensor(data.shape_);
        AllocSpace(&ShipTensor);
        for (int i = 0; i < data.shape_[0]; ++ i) {
            for (int j = 0; j < data.shape_[1]; ++ j) {
                DType x = *(data.dptr_ + data.stride_ * i + j);
                *(ShipTensor.dptr_ + i * ShipTensor.stride_ + j) = (x + 1) / 2.;
            }
        }

        for (int i = 0; i < data.shape_[0]; ++ i) {
            for (int j = 0; j < data.shape_[1]; ++ j) {
                DType x = *(ShipTensor.dptr_ + i * ShipTensor.stride_ + j)
                          - static_cast <float> (rand()) / static_cast <float> (RAND_MAX);
                if (x >= 0)
                    *(out.dptr_ + i * out.stride_ + j) = 1;
                else
                    *(out.dptr_ + i * out.stride_ + j) = -1;
            }
        }

        FreeSpace(&ShipTensor);
    }
*/
template <typename DType>
void BinaryActivationForward(Tensor<cpu, 2, DType> &data, Tensor<cpu, 2, DType> &out) {
    for (int i = 0; i < data.shape_[0]; ++ i) {
        for (int j = 0; j < data.shape_[1]; ++ j) {
            DType x = *(data.dptr_ + data.stride_ * i + j);
            if (x >= 0) {
                *(out.dptr_ + i * out.stride_ + j) = 1;
            } else {
                *(out.dptr_ + i * out.stride_ + j) = -1;
            }
        }
    }
}
template<typename DType>
void BinaryActivationBackward(Tensor<cpu, 2, DType> &grad, Tensor<cpu, 2, DType> &gdata) {
        for (int i = 0; i < grad.shape_[0]; ++ i) {
            for (int j = 0; j < grad.shape_[1]; ++ j) {
                *(gdata.dptr_ + gdata.stride_ * i + j) = *(grad.dptr_ + grad.stride_ * i + j);
            }
        }
    }
}


namespace mxnet {
namespace op {
DMLC_REGISTER_PARAMETER(BinaryActivationParam);

template<>
Operator* CreateOp<cpu>(BinaryActivationParam param, int dtype) {
    Operator *op = NULL;

    MSHADOW_REAL_TYPE_SWITCH(dtype, DType, {
            op = new BinaryActivationOp<cpu, DType>(param);
    })
    return op;
}

// DO_BIND_DISPATCH comes from operator_common.h
Operator *BinaryActivationProp::CreateOperatorEx(Context ctx,
                                                  std::vector<TShape> *in_shape,
                                                  std::vector<int> *in_type) const {
    std::vector<TShape> out_shape, aux_shape;
    std::vector<int> out_type, aux_type;
    CHECK(InferType(in_type, &out_type, &aux_type));
    CHECK(InferShape(in_shape, &out_shape, &aux_shape));
    DO_BIND_DISPATCH(CreateOp, param_, (*in_type)[0]);
}

MXNET_REGISTER_OP_PROPERTY(BinaryActivation, BinaryActivationProp)
.add_argument("data", "Symbol", "Input data to the BinaryActivationOp.")
.add_arguments(BinaryActivationParam::__FIELDS__())
.describe("binarize activation.");

}  // namespace op
}  // namespace mxnet

