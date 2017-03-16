/*!
* created by chenhao li
*/

#include "./hardtanh.h"

namespace mshadow{
template<typename DType>
void HardTanhForward(Tensor<cpu, 2, DType> &data, Tensor<cpu, 2, DType> &out){
        for (int i = 0; i < data.shape_[0]; ++ i) {
            for (int j = 0; j < data.shape_[1]; ++ j) {
                DType x = *(data.dptr_ + i * data.stride_ + j);
                if (x > 1)
                    *(out.dptr_ + i * out.stride_ + j) = 1;
                else if (x < -1)
                    *(out.dptr_ + i * out.stride_ + j) = -1;
                else
                    *(out.dptr_ + i * out.stride_ + j) = x;
            }
        }
    }
template<typename DType>
void HardTanhBackward(Tensor<cpu, 2, DType> &data, Tensor<cpu, 2, DType> &grad, Tensor<cpu, 2, DType> &gdata){
        for (int i = 0; i < grad.shape_[0]; ++ i) {
            for (int j = 0; j < grad.shape_[1]; ++ j) {
                DType x = *(grad.dptr_ + grad.stride_ * i + j);
                DType y = *(data.dptr_ + data.stride_ * i + j);
                if (y < -1. || y > 1.)
                    *(gdata.dptr_ + i * gdata.stride_ + j) = 0;
                else
                    *(gdata.dptr_ + i * gdata.stride_ + j) = x;
            }
        }
    }
}

namespace mxnet {
namespace op {
DMLC_REGISTER_PARAMETER(HardTanhParam);

template<>
Operator* CreateOp<cpu>(HardTanhParam param, int dtype) {
    Operator *op = NULL;

    MSHADOW_REAL_TYPE_SWITCH(dtype, DType, {
            op = new HardTanhOp<cpu, DType>(param);
    })
    return op;
}

// DO_BIND_DISPATCH comes from operator_common.h
Operator *HardTanhProp::CreateOperatorEx(Context ctx,
                                                 std::vector<TShape> *in_shape,
                                                 std::vector<int> *in_type) const {
    std::vector<TShape> out_shape, aux_shape;
    std::vector<int> out_type, aux_type;
    CHECK(InferType(in_type, &out_type, &aux_type));
    CHECK(InferShape(in_shape, &out_shape, &aux_shape));
    DO_BIND_DISPATCH(CreateOp, param_, (*in_type)[0]);
}

MXNET_REGISTER_OP_PROPERTY(HardTanh, HardTanhProp)
.add_argument("data", "Symbol", "Input data to the BinaryActivationOp.")
.add_arguments(HardTanhParam::__FIELDS__())
.describe("binarize activation.");

}  // namespace op
}  // namespace mxnet

