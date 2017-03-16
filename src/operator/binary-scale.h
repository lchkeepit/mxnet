//
// Created by ChenhaoLi on 2017/2/20.
//

#ifndef MXNET_BINARY_SCALE_H_H
#define MXNET_BINARY_SCALE_H_H

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

namespace mxnet{
namespace op{

namespace bscalec {
enum BinaryScaleOpInputs {kData, kWeight};
enum BinaryScaleOpOutputs {kOut};
}

struct BinaryScaleParam : public dmlc::Parameter<BinaryScaleParam> {
    float init_value;
    bool initialized;
    DMLC_DECLARE_PARAMETER(BinaryScaleParam) {
        DMLC_DECLARE_FIELD(init_value).set_default(0.001)
        .describe("init_value");
        DMLC_DECLARE_FIELD(initialized).set_default(false)
        .describe("initialized");
    }
};

template <typename xpu, typename DType>
class BinaryScaleOp : public Operator {
public:
    explicit BinaryScaleOp(BinaryScaleParam p) {
        this->param_ = p;
    }
    virtual void Forward(const OpContext &ctx,
                        const std::vector<TBlob> &in_data,
                        const std::vector<OpReqType> &req,
                        const std::vector<TBlob> &out_data,
                        const std::vector<TBlob> &aux_args) {
        using namespace mshadow;
        using namespace mshadow::expr;

        if (req[bscalec::kOut] == kNullOp) return;

        Stream<xpu> *s = ctx.get_stream<xpu>();
        const TShape& ishape = in_data[bscalec::kData].shape_;
        const TShape& oshape = out_data[bscalec::kOut].shape_;

        Tensor<xpu, 2, DType> data = in_data[bscalec::kData].FlatTo2D<xpu, DType>(s);
        Tensor<xpu, 2, DType> wValue = in_data[bscalec::kWeight].FlatTo2D<xpu, DType>(s);
        Tensor<xpu, 2, DType> out = out_data[bscalec::kOut].FlatTo2D<xpu, DType>(s);

        if (this->param_.initialized == false) {
            this->param_.initialized = true;
            wValue = this->param_.init_value;
        }
        BinaryScaleForward(data, wValue, out);
        //out = data * broadcast<1>(wValue, ishape);
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
        const TShape& ishape = in_data[bscalec::kData].shape_;
        const TShape& oshape = out_grad[bscalec::kOut].shape_;

        Tensor<xpu, 2, DType> data = in_data[bscalec::kData].FlatTo2D<xpu, DType>(s);
        Tensor<xpu, 2, DType> wValue = in_data[bscalec::kWeight].FlatTo2D<xpu, DType>(s);
        Tensor<xpu, 2, DType> grad = out_grad[bscalec::kOut].FlatTo2D<xpu, DType>(s);
        Tensor<xpu, 2, DType> gdata = in_grad[bscalec::kData].FlatTo2D<xpu, DType>(s);
        Tensor<xpu, 2, DType> gwValue = in_grad[bscalec::kWeight].FlatTo2D<xpu, DType>(s);
        //BinaryScaleBackward();
        //gdata = grad * broadcast<1>(wValue, grad.shape_);
        BinaryScaleBackward(data, grad, wValue, gwValue, gdata);
        //gwValue = sumall(F<mshadow_op::multiply>(grad, data));
    }
private:
    BinaryScaleParam param_;
};

template <typename xpu>
Operator* CreateOp(BinaryScaleParam param, int dtype) ;

#if DMLC_USE_CXX11
class BinaryScaleProp : public OperatorProperty {
    public:
    std::vector<std::string> ListArguments() const override {
        return {"data", "weight"};
    }
    void Init(const std::vector<std::pair<std::string, std::string>> &kwargs) override {
        param_.Init(kwargs);
    }

    std::map<std::string, std::string> GetParams() const override {
        return param_.__DICT__();
    }

    bool InferShape(std::vector<TShape> *in_shape,
                  std::vector<TShape> *out_shape,
                  std::vector<TShape> *aux_shape) const override {
    using namespace mshadow;

    const TShape &dshape = (*in_shape)[bscalec::kData];
    // require data to be known
    if (dshape.ndim() ==  0) return false;
    //in_shape->at(1) = TShape(Shape1(1));
    SHAPE_ASSIGN_CHECK(*in_shape, bscalec::kWeight, TShape(Shape2(1, 1)));
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
    BinaryScaleProp* bscale_sym = new BinaryScaleProp();
    bscale_sym->param_ = this->param_;
    return bscale_sym;
  }

  std::string TypeString() const override {
    return "BinaryScale";
  }
  Operator* CreateOperator(Context ctx) const override {
    LOG(FATAL) << "Not Implemented.";
    return NULL;
  }

  Operator* CreateOperatorEx(Context ctx, std::vector<TShape> *in_shape,
                             std::vector<int> *in_type) const override;

 private:
  BinaryScaleParam param_;
};
#endif

}
}

#endif //MXNET_BINARY_SCALE_H_H
