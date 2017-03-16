//
// Created by 算法S100 on 2017/1/13.
//

#ifndef MXNET_BINARY_ACTIVATION_H
#define MXNET_BINARY_ACTIVATION_H
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
#include <cassert>
#include "./operator_common.h"
#include "./mshadow_op.h"

#define lchdebug0
namespace mxnet {
namespace op {

// Declare enumeration of input order to make code more intuitive.
// These enums are only visible within this header
namespace bactc {
enum BinaryActivationOpInputs {kData};
enum BinaryActivationOpOutputs {kOut};
}  // fullc

struct BinaryActivationParam : public dmlc::Parameter<BinaryActivationParam> {
    DMLC_DECLARE_PARAMETER(BinaryActivationParam) {

    }
};

/**
 * \brief This is the implementation of fully connected operator.
 * \tparam xpu The device that the op will be executed on.
 */
template<typename xpu, typename DType>
class BinaryActivationOp : public Operator {
public:
    explicit BinaryActivationOp(BinaryActivationParam p) {
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


        BinaryActivationForward(data, out);
        /*
        Tensor<xpu, 2, DType> res = CalcTensor(data); // (x + 1) /2
        Tensor<xpu, 2, DType> randTensor = RandomTensor(data); // random
        Tensor<xpu, 2, DType> minusTensor = minus(res, randTensor);
        Tensor<xpu, 2, DType> midres = binary(minusTensor, true);
        Assign(out, req[bactc::kOut], F<mshadow_op::identity>(midres));
        FreeSpace(&res);
        FreeSpace(&randTensor);
        FreeSpace(&minusTensor);
        FreeSpace(&midres);
        */
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
        /*
        Tensor<xpu, 2, DType> data = in_data[bactc::kData].get_with_shape<xpu, 2, DType>(
                Shape2(ishape[0], ishape.ProdShape(1, ishape.ndim())), s);
        Tensor<xpu, 2, DType> grad = out_grad[bactc::kOut].get_with_shape<xpu, 2, DType>(
                Shape2(oshape[0], oshape.ProdShape(1, oshape.ndim())), s);
        Tensor<xpu, 2, DType> gdata = in_grad[bactc::kData].get_with_shape<xpu, 2, DType>(
                Shape2(ishape[0], ishape.ProdShape(1, ishape.ndim())), s);
        */
        //std::cout << "bact backward" << std::endl;
        Tensor<xpu, 2, DType> data = in_data[bactc::kData].FlatTo2D<xpu, DType>(s);
        Tensor<xpu, 2, DType> grad = out_grad[bactc::kOut].FlatTo2D<xpu, DType>(s);
        Tensor<xpu, 2, DType> gdata = in_grad[bactc::kData].FlatTo2D<xpu, DType>(s);
        using namespace mshadow_op;

        BinaryActivationBackward(grad, gdata);

        //Assign(gdata, req[bactc::kData], F<identity>(grad));

    }

private:
    BinaryActivationParam param_;
};  // class BinaryActivationOp

// Decalre Factory function, used for dispatch specialization
template<typename xpu>
Operator* CreateOp(BinaryActivationParam param, int dtype);

#if DMLC_USE_CXX11
class BinaryActivationProp : public OperatorProperty {
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
    BinaryActivationProp* bac_sym = new BinaryActivationProp();
    bac_sym->param_ = this->param_;
    return bac_sym;
  }

  std::string TypeString() const override {
    return "BinaryActivation";
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
  BinaryActivationParam param_;
};  // class FullyConnectedSymbol
#endif
}  // namespace op
}  // namespace mxnet

#endif //MXNET_BINARY_ACTIVATION_H
