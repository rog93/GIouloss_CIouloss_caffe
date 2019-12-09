#ifndef CAFFE_CIOU_LOSS_LAYERS_HPP_
#define CAFFE_CIOU_L1_LOSS_LAYERS_HPP_

#include <vector>

#include "caffe/blob.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"

#include "caffe/layers/loss_layer.hpp"

namespace caffe {

template <typename Dtype>
class CIouLossLayer : public LossLayer<Dtype> {
 public:
  explicit CIouLossLayer(const LayerParameter& param)
      : LossLayer<Dtype>(param), diff_() {}
  virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);

  virtual inline const char* type() const { return "CIouLoss"; }

  virtual inline int ExactNumBottomBlobs() const { return 4; }
  virtual inline int ExactNumTopBlobs() const { return 1; }

  /**
   * Unlike most loss layers, in the CIouLossLayer we can backpropagate
   * to both inputs -- override to return true and always allow force_backward.
   */
  virtual inline bool AllowForceBackward(const int bottom_index) const {
    return true;
  }

 protected:
  virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);
  virtual void Forward_gpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);

  virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);
  virtual void Backward_gpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);

  Blob<Dtype> diff_;
  Blob<Dtype> pred_;
  Blob<Dtype> gt_;
  Dtype norm_count_;
  Dtype x_std_;
  Dtype y_std_;
  Dtype w_std_;
  Dtype h_std_;
  Dtype clip_bound_;
  bool clip_;
  int channel_;
};

}  // namespace caffe

#endif  // CAFFE_CIOU_LOSS_LAYERS_HPP_
