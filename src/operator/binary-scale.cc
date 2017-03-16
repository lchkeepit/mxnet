/*!
* created by chenhao li
*/

#include "./binary-scale.h"
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

template <typename DType>
void BinaryScaleForward(Tensor<cpu, 2, DType> &data, Tensor<cpu, 2, DType> &wValue, Tensor<cpu, 2, DType> &out) {
    for (int i = 0; i < data.shape_[0]; ++ i) {
        for (int j = 0; j < data.shape_[1]; ++ j) {
            int idx = i * data.stride_ + j;
            int idx2 = i * out.stride_ + j;
            *(out.dptr_ + idx2) = (*(data.dptr_ + idx)) * (*wValue.dptr_);
        }
    }
}
template<typename DType>
void BinaryScaleBackward(Tensor<cpu, 2, DType> &data, Tensor<cpu, 2, DType> &grad, Tensor<cpu, 2, DType> &wValue,
                         Tensor<cpu, 2, DType> &gwValue, Tensor<cpu, 2, DType> &gdata) {
    *(gwValue.dptr_) = 0;
    for (int i = 0; i < data.shape_[0]; ++ i) {
        for (int j = 0; j < data.shape_[1]; ++ j) {
            int idx = i * data.stride_ + j;
            int idx2 = i * grad.stride_ + j;
            int idx3 = i * gdata.stride_ + j;
            *(gwValue.dptr_) += (*(grad.dptr_ + idx2)) * (*(data.dptr_ + idx));
            *(gdata.dptr_ + idx3) = (*(grad.dptr_ + idx2)) * (*wValue.dptr_);
        }
    }
}
}


namespace mxnet {
namespace op {
DMLC_REGISTER_PARAMETER(BinaryScaleParam);

template<>
Operator* CreateOp<cpu>(BinaryScaleParam param, int dtype) {
    Operator *op = NULL;

    MSHADOW_REAL_TYPE_SWITCH(dtype, DType, {
            op = new BinaryScaleOp<cpu, DType>(param);
    })
    return op;
}

// DO_BIND_DISPATCH comes from operator_common.h
Operator *BinaryScaleProp::CreateOperatorEx(Context ctx,
                                                 std::vector<TShape> *in_shape,
                                                 std::vector<int> *in_type) const {
    std::vector<TShape> out_shape, aux_shape;
    std::vector<int> out_type, aux_type;
    CHECK(InferType(in_type, &out_type, &aux_type));
    CHECK(InferShape(in_shape, &out_shape, &aux_shape));
    DO_BIND_DISPATCH(CreateOp, param_, (*in_type)[0]);
}

MXNET_REGISTER_OP_PROPERTY(BinaryScale, BinaryScaleProp)
.add_argument("data", "Symbol", "Input data to the BinaryScaleOp.")
.add_arguments(BinaryScaleParam::__FIELDS__())
.describe("binarize scale.");

}  // namespace op
}  // namespace mxnet

