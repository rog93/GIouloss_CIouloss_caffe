// ------------------------------------------------------------------
// Fast R-CNN
// Copyright (c) 2015 Microsoft
// Licensed under The MIT License [see fast-rcnn/LICENSE for details]
// Written by Ross Girshick
// ------------------------------------------------------------------

#include <algorithm>
#include <cfloat>
#include <vector>

#include "caffe/util/math_functions.hpp"
#include "caffe/layers/ciou_loss_layer.hpp"


namespace caffe {

template <typename Dtype>
void CIouLossLayer<Dtype>::LayerSetUp(
  const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  CHECK_EQ(bottom.size(), 4) << "must specify both "
    "inside and outside weights";
  CHECK_EQ(top.size(), 1) << "must be 1 top";
  CIouLossParameter loss_param = this->layer_param_.ciou_loss_param();
  clip_ = loss_param.clip();
  clip_bound_ = loss_param.clip_bound();
  x_std_ = loss_param.x_std();
  y_std_ = loss_param.y_std();
  w_std_ = loss_param.w_std();
  h_std_ = loss_param.h_std();
  channel_ = bottom[0]->channels();
}

template <typename Dtype>
void CIouLossLayer<Dtype>::Reshape(
  const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  LossLayer<Dtype>::Reshape(bottom, top);
  CHECK_EQ(bottom[0]->channels(), bottom[1]->channels());
  CHECK_EQ(bottom[0]->height(), bottom[1]->height());
  CHECK_EQ(bottom[0]->width(), bottom[1]->width());
  CHECK_EQ(bottom[0]->channels(), bottom[2]->channels());
  CHECK_EQ(bottom[0]->height(), bottom[2]->height());
  CHECK_EQ(bottom[0]->width(), bottom[2]->width());
  CHECK_EQ(bottom[0]->channels(), bottom[3]->channels());
  CHECK_EQ(bottom[0]->height(), bottom[3]->height());
  CHECK_EQ(bottom[0]->width(), bottom[3]->width());

  pred_.Reshape(bottom[0]->num(), bottom[0]->channels() * 2,
      bottom[0]->height(), bottom[0]->width());
  gt_.Reshape(bottom[0]->num(), bottom[0]->channels() * 2,
      bottom[0]->height(), bottom[0]->width());
  diff_.Reshape(bottom[0]->num(), bottom[0]->channels(),
      1, 1);
}

template <typename Dtype>
void CIouLossLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  NOT_IMPLEMENTED;
}

template <typename Dtype>
void CIouLossLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  NOT_IMPLEMENTED;
}

#ifdef CPU_ONLY
STUB_GPU(CIouLossLayer);
#endif

INSTANTIATE_CLASS(CIouLossLayer);
REGISTER_LAYER_CLASS(CIouLoss);

}  // namespace caffe
