//
// Created by 算法S100 on 2017/1/13.
//

#ifndef MXNET_HARDTANH_H
#define MXNET_HARDTANH_H
/*!
* created by chenhao li
*/

#include <iostream>
#include <cmath>
#include <cstdlib>
#include <dmlc/logging.h>
#include <dmlc/parameter.h>
#include <mxnet/operator.h>
#include <map>
#include <vector>
#include <string>
#include <utility>
#include "./operator_common.h"
#include "./mshadow_op.h"

#define lchdebug0
namespace mxnet {
namespace op {

// Declare enumeration of input order to make code more intuitive.
// These enums are only visible within this header
namespace bactc {
enum HardTanhOpInputs {kData};
enum HardTanhOpOutputs {kOut};
}  // fullc

struct HardTanhParam : public dmlc::Parameter<HardTanhParam> {
    DMLC_DECLARE_PARAMETER(HardTanhParam) {

    }
};

/**
 * \brief This is the implementation of fully connected operator.
 * \tparam xpu The device that the op will be executed on.
 */
template<typename xpu, typename DType>
class HardTanhOp : public Operator {
public:
    explicit HardTanhOp(HardTanhParam p) {
        this->param_ = p;
    }


    virtual void Forward(const OpContext &ctx,
                         const std::vector<TBlob> &in_data,
                         const std::vector<OpReqType> &req,
                         const std::vector<TBlob> &out_data,
                         const std::vector<TBlob> &aux_args) {
        using namespace mshadow;
        using namespace mshadow::expr;
        //using namespace mshadow_op;
        if (req[bactc::kOut] == kNullOp) return;

        Stream<xpu> *s = ctx.get_stream<xpu>();
        const TShape& ishape = in_data[bactc::kData].shape_;
        const TShape& oshape = out_data[bactc::kOut].shape_;

        Tensor<xpu, 2, DType> data = in_data[bactc::kData].FlatTo2D<xpu, DType>(s);
        Tensor<xpu, 2, DType> out = out_data[bactc::kOut].FlatTo2D<xpu, DType>(s);

        HardTanhForward(data, out);
        /*
        Tensor<xpu, 2, DType> res = ForwardCalcTensor(data);
        Assign(out, req[bactc::kOut], F<mshadow_op::identity>(res));
        FreeSpace(&res);
         */
    }

    mshadow::Tensor<xpu, 2, DType> BinaryBackward(mshadow::Tensor<xpu, 2, DType> &tensor) {
        using namespace mshadow;
        Tensor<xpu, 2, DType> ntensor(tensor.shape_);
        AllocSpace(&ntensor);
        for (int i = 0; i < tensor.shape_[0]; ++ i) {
            for (int j = 0; j < tensor.shape_[1]; ++ j) {
                DType x = *(tensor.dptr_ + i * tensor.stride_ + j);
                if (x > 1.) {
                    *(ntensor.dptr_ + i * ntensor.stride_ + j) = 0;
                } else if (x < -1.) {
                    *(ntensor.dptr_ + i * ntensor.stride_ + j) = 0;
                } else {
                    *(ntensor.dptr_ + i * ntensor.stride_ + j) = x;
                }
            }
        }
        return ntensor;
    }
    virtual void Backward(const OpContext &ctx,
                          const std::vector<TBlob> &out_grad,
                          const std::vector<TBlob> &in_data,
                          const std::vector<TBlob> &out_data,
                          const std::vector<OpReqType> &req,
                          const std::vector<TBlob> &in_grad,
                          const std::vector<TBlob> &aux_args) {
        using namespace mshadow;
        using namespace mshadow::expr;

        Stream<xpu> *s = ctx.get_stream<xpu>();
        const TShape& ishape = in_data[bactc::kData].shape_;
        const TShape& oshape = out_grad[bactc::kOut].shape_;

        Tensor<xpu, 2, DType> data = in_data[bactc::kData].FlatTo2D<xpu, DType>(s);
        Tensor<xpu, 2, DType> grad = out_grad[bactc::kOut].FlatTo2D<xpu, DType>(s);
        Tensor<xpu, 2, DType> gdata = in_grad[bactc::kData].FlatTo2D<xpu, DType>(s);
        using namespace mshadow_op;

        HardTanhBackward(data, grad, gdata);
        /*
        Tensor<xpu, 2, DType> ntensor = BinaryBackward(grad);
        Assign(gdata, req[bactc::kData], F<identity>(ntensor));
        FreeSpace(&ntensor);
        */
    }

private:
    HardTanhParam param_;
};  // class BinaryActivationOp

// Decalre Factory function, used for dispatch specialization
template<typename xpu>
Operator* CreateOp(HardTanhParam param, int dtype);

#if DMLC_USE_CXX11
class HardTanhProp : public OperatorProperty {
 public:

  void Init(const std::vector<std::pair<std::string, std::string> >& kwargs) override {
    param_.Init(kwargs);
  }

  std::map<std::string, std::string> GetParams() const override {
    return param_.__DICT__();
  }

  bool InferShape(std::vector<TShape> *in_shape,
                  std::vector<TShape> *out_shape,
                  std::vector<TShape> *aux_shape) const override {
    using namespace mshadow;

    const TShape &dshape = (*in_shape)[bactc::kData];
    // require data to be known
    if (dshape.ndim() ==  0) return false;

    //index_t num_input = dshape.ProdShape(1, dshape.ndim());
    //SHAPE_ASSIGN_CHECK(*in_shape, fullc::kWeight, Shape2(param_.num_hidden, num_input));

    out_shape->clear();
    out_shape->push_back(dshape);
    return true;
  }

  bool InferType(std::vector<int> *in_type,
                 std::vector<int> *out_type,
                 std::vector<int> *aux_type) const override {
    CHECK_GE(in_type->size(), 1);
    int dtype = (*in_type)[0];
    CHECK_NE(dtype, -1) << "First input must have specified type";
    for (index_t i = 0; i < in_type->size(); ++i) {
      if ((*in_type)[i] == -1) {
        (*in_type)[i] = dtype;
      } else {
        CHECK_EQ((*in_type)[i], dtype) << "This layer requires uniform type. "
                                       << "Expected " << dtype << " v.s. given "
                                       << (*in_type)[i] << " at " << ListArguments()[i];
      }
    }
    out_type->clear();
    out_type->push_back(dtype);
    return true;
  }

  OperatorProperty* Copy() const override {
    HardTanhProp* bac_sym = new HardTanhProp();
    bac_sym->param_ = this->param_;
    return bac_sym;
  }

  std::string TypeString() const override {
    return "HardTanh";
  }
/*
  // decalre dependency and inplace optimization options
  std::vector<int> DeclareBackwardDependency(
    const std::vector<int> &out_grad,
    const std::vector<int> &in_data,
    const std::vector<int> &out_data) const override {
    return {out_grad[bactc::kOut], out_data[bactc::kOut], in_data[bactc::kData]};
  }

  std::vector<std::pair<int, void*> > BackwardInplaceOption(
    const std::vector<int> &out_grad,
    const std::vector<int> &in_data,
    const std::vector<int> &out_data,
    const std::vector<void*> &in_grad) const override {
    return {{out_grad[bactc::kOut], in_grad[bactc::kData]}};
  }
  std::vector<std::pair<int, void*> > ForwardInplaceOption(
    const std::vector<int> &in_data,
    const std::vector<void*> &out_data) const override {
    return {{in_data[bactc::kData], out_data[bactc::kOut]}};
  }
*/
  Operator* CreateOperator(Context ctx) const override {
    LOG(FATAL) << "Not Implemented.";
    return NULL;
  }

  Operator* CreateOperatorEx(Context ctx, std::vector<TShape> *in_shape,
                             std::vector<int> *in_type) const override;

 private:
  HardTanhParam param_;
};  // class FullyConnectedSymbol
#endif
}  // namespace op
}  // namespace mxnet

#endif //MXNET_BINARY_ACTIVATION_H
