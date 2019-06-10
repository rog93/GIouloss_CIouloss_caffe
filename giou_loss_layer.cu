#include <algorithm>
#include <cfloat>
#include <vector>
#include <cmath>
#include "caffe/util/math_functions.hpp"
#include "caffe/layers/giou_loss_layer.hpp"

namespace caffe {

template <typename Dtype>
__global__ void bbox_transform_kernel(const int nthreads, const Dtype* deltas, Dtype* out, const bool clip, const Dtype clip_bound, 
  const Dtype x_std, const Dtype y_std, const Dtype w_std, const Dtype h_std) {
  CUDA_KERNEL_LOOP(index, nthreads) {
    int i = index;
    Dtype dx = deltas[i * 4 + 0] * x_std;
    Dtype dy = deltas[i * 4 + 1] * y_std;
    
    Dtype dw = deltas[i * 4 + 2] * w_std;
    Dtype dh = deltas[i * 4 + 3] * h_std;
     
    if (clip){
      dw = min(dw, clip_bound);
      dh = min(dh, clip_bound);
    }
    
    Dtype x1 = dx - 0.5 * exp(dw);
    Dtype y1 = dy - 0.5 * exp(dh);
    Dtype x2 = dx + 0.5 * exp(dw);
    Dtype y2 = dy + 0.5 * exp(dh);


    out[i * 4 + 0] = x1;
    out[i * 4 + 1] = y1;
    out[i * 4 + 2] = x2;
    out[i * 4 + 3] = y2;

  }
}

template <typename Dtype>
__global__ void GIouForward(const int n, const Dtype* pred, const Dtype* gt, 
  const Dtype* bbox_inside_weights, const Dtype* bbox_outside_weights, Dtype* loss_data) {
  CUDA_KERNEL_LOOP(index, n) {
    int i = index;
    
    Dtype x1_orig = pred[i * 4 + 0];
    Dtype y1_orig = pred[i * 4 + 1];
    Dtype x2_orig = pred[i * 4 + 2];
    Dtype y2_orig = pred[i * 4 + 3];

    Dtype x1 = min(x1_orig, x2_orig);
    Dtype y1 = min(y1_orig, y2_orig);
    Dtype x2 = max(x1_orig, x2_orig);
    Dtype y2 = max(y1_orig, y2_orig);

    
    Dtype gx1 = gt[i * 4 + 0];
    Dtype gy1 = gt[i * 4 + 1];
    Dtype gx2 = gt[i * 4 + 2];
    Dtype gy2 = gt[i * 4 + 3];

    
    Dtype xkis1 = max(x1, gx1);
    Dtype ykis1 = max(y1, gy1);
    Dtype xkis2 = min(x2, gx2);
    Dtype ykis2 = min(y2, gy2);


    Dtype xc1 = min(x1, gx1);
    Dtype yc1 = min(y1, gy1);
    Dtype xc2 = max(x2, gx2);
    Dtype yc2 = max(y2, gy2);

    
    Dtype intsctk = 0;
    int mask = int(ykis2 > ykis1) * int(xkis2 > xkis1);
    if (mask > 0){
        intsctk = (xkis2 - xkis1) * (ykis2 - ykis1);
    }
    Dtype unionk = (x2 - x1) * (y2 - y1) + (gx2 - gx1) * (gy2 - gy1) - intsctk + 1e-7;
    
    
    Dtype iouk = intsctk / unionk;
    Dtype area = (xc2 - xc1) * (yc2 - yc1) + 1e-7;


    Dtype miouk = iouk - ((area - unionk) / area);
    Dtype iou_weights = (bbox_inside_weights[i * 4 + 0] + bbox_inside_weights[i * 4 + 1] + 
			bbox_inside_weights[i * 4 + 2] + bbox_inside_weights[i * 4 + 3]) * 
			(bbox_outside_weights[i * 4 + 0] + bbox_outside_weights[i * 4 + 1] + 
			bbox_outside_weights[i * 4 + 2] + bbox_outside_weights[i * 4 + 3]) / Dtype(16);


    loss_data[i] = (Dtype(1) - miouk) * iou_weights;

  }
}


template <typename Dtype>
void GIouLossLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
    const vector<Blob<Dtype>*>& top) {
  norm_count_ = bottom[0]->num();
  int cls_num = channel_ / 4;
  int loop_count = norm_count_ * cls_num;
  CHECK_GE(norm_count_, 0);
  if (norm_count_ == 0) {
    norm_count_ = Dtype(1);
    loop_count = Dtype(2);
  }
  bbox_transform_kernel<Dtype><<<CAFFE_GET_BLOCKS(loop_count), CAFFE_CUDA_NUM_THREADS>>>(
      loop_count, bottom[0]->gpu_data(), pred_.mutable_gpu_data(), clip_, clip_bound_, 
      x_std_, y_std_, w_std_, h_std_);
  
  bbox_transform_kernel<Dtype><<<CAFFE_GET_BLOCKS(loop_count), CAFFE_CUDA_NUM_THREADS>>>(
      loop_count, bottom[1]->gpu_data(), gt_.mutable_gpu_data(), clip_, clip_bound_, 
      x_std_, y_std_, w_std_, h_std_);

  GIouForward<Dtype><<<CAFFE_GET_BLOCKS(loop_count), CAFFE_CUDA_NUM_THREADS>>>(
      loop_count, pred_.gpu_data(), gt_.gpu_data(), bottom[2]->gpu_data(), bottom[3]->gpu_data(), diff_.mutable_gpu_data());

  CUDA_POST_KERNEL_CHECK;

  Dtype loss;

  caffe_gpu_asum(loop_count, diff_.gpu_data(), &loss);

  top[0]->mutable_cpu_data()[0] = loss / norm_count_;
}

template <typename Dtype>
__global__ void GIouBackward(const int n, const Dtype* bbox_pred, const Dtype* pred, const Dtype* gt, 
  const Dtype* bbox_inside_weights, const Dtype* bbox_outside_weights, Dtype* bottom_diff, const bool clip, 
  const Dtype clip_bound, const Dtype x_std, const Dtype y_std, const Dtype w_std, const Dtype h_std, 
  const Dtype loss_weight, int cls_num) {
  CUDA_KERNEL_LOOP(index, n) {
    int i = index;
    Dtype x1_orig = pred[i * 4 + 0];
    Dtype y1_orig = pred[i * 4 + 1];
    Dtype x2_orig = pred[i * 4 + 2];
    Dtype y2_orig = pred[i * 4 + 3];

    Dtype x1 = min(x1_orig, x2_orig);
    Dtype y1 = min(y1_orig, y2_orig);
    Dtype x2 = max(x1_orig, x2_orig);
    Dtype y2 = max(y1_orig, y2_orig);

    Dtype gx1 = gt[i * 4 + 0];
    Dtype gy1 = gt[i * 4 + 1];
    Dtype gx2 = gt[i * 4 + 2];
    Dtype gy2 = gt[i * 4 + 3];


    Dtype X = (y2 - y1) * (x2 - x1);
    Dtype Xhat = (gy2 - gy1) * (gx2 - gx1);

    
    Dtype xkis1 = max(x1, gx1);
    Dtype ykis1 = max(y1, gy1);
    Dtype xkis2 = min(x2, gx2);
    Dtype ykis2 = min(y2, gy2);

    Dtype I = 0;
    Dtype Ih = 0;
    Dtype Iw = 0;


    int mask = int(ykis2 > ykis1) * int(xkis2 > xkis1);
    
    if (mask > 0){
        Ih = ykis2 - ykis1;
        Iw = xkis2 - xkis1;
        I = Iw * Ih;
    }    
    
    Dtype U = X + Xhat - I  + 1e-7;

    
    Dtype Cw = max(x2, gx2) - min(x1, gx1);
    Dtype Ch = max(y2, gy2) - min(y1, gy1);

    Dtype C = Cw * Ch + 1e-7;

    
    Dtype dX_wrt_t = -1 * (x2 - x1);
    Dtype dX_wrt_b = x2 - x1;
    Dtype dX_wrt_l = -1 * (y2 - y1);
    Dtype dX_wrt_r = y2 - y1;
    

    Dtype dI_wrt_t = y1 > gy1 ? (-1 * Iw) : 0;
    Dtype dI_wrt_b = y2 < gy2 ? Iw : 0;
    Dtype dI_wrt_l = x1 > gx1 ? (-1 * Ih) : 0;
    Dtype dI_wrt_r = x2 < gx2 ? Ih : 0;
    
 
    Dtype dU_wrt_t = dX_wrt_t - dI_wrt_t;
    Dtype dU_wrt_b = dX_wrt_b - dI_wrt_b;
    Dtype dU_wrt_l = dX_wrt_l - dI_wrt_l;
    Dtype dU_wrt_r = dX_wrt_r - dI_wrt_r;
    

    Dtype dC_wrt_t = y1 < gy1 ? (-1 * Cw) : 0;
    Dtype dC_wrt_b = y2 > gy2 ? Cw : 0;
    Dtype dC_wrt_l = x1 < gx1 ? (-1 * Ch) : 0;
    Dtype dC_wrt_r = x2 > gx2 ? Ch : 0;
    
    Dtype p_dt = 0;
    Dtype p_db = 0;
    Dtype p_dl = 0;
    Dtype p_dr = 0;
    

    if (U > 0) {
      p_dt = ((U * dI_wrt_t) - (I * dU_wrt_t)) / (U * U);
      p_db = ((U * dI_wrt_b) - (I * dU_wrt_b)) / (U * U);
      p_dl = ((U * dI_wrt_l) - (I * dU_wrt_l)) / (U * U);
      p_dr = ((U * dI_wrt_r) - (I * dU_wrt_r)) / (U * U);
    }
    
    if (C > 0) {
      p_dt += ((C * dU_wrt_t) - (U * dC_wrt_t)) / (C * C);
      p_db += ((C * dU_wrt_b) - (U * dC_wrt_b)) / (C * C);
      p_dl += ((C * dU_wrt_l) - (U * dC_wrt_l)) / (C * C);
      p_dr += ((C * dU_wrt_r) - (U * dC_wrt_r)) / (C * C);
    }
    
    
    Dtype dt = y1_orig < y2_orig ? p_dt : p_db;
    Dtype db = y1_orig < y2_orig ? p_db : p_dt;
    Dtype dl = x1_orig < x2_orig ? p_dl : p_dr;
    Dtype dr = x1_orig < x2_orig ? p_dr : p_dl;  

    Dtype iou_weights = (bbox_inside_weights[i * 4 + 0] + bbox_inside_weights[i * 4 + 1] + 
			bbox_inside_weights[i * 4 + 2] + bbox_inside_weights[i * 4 + 3]) * 
			(bbox_outside_weights[i * 4 + 0] + bbox_outside_weights[i * 4 + 1] + 
			bbox_outside_weights[i * 4 + 2] + bbox_outside_weights[i * 4 + 3]) / Dtype(16);
    
    bottom_diff[i * 4 + 0] = 0; 
    bottom_diff[i * 4 + 1] = 0; 
    bottom_diff[i * 4 + 2] = 0; 
    bottom_diff[i * 4 + 3] = 0; 
    
    bottom_diff[i * 4 + 0] = (dl + dr) * static_cast<Dtype>(cls_num) / static_cast<Dtype>(n) * 
        Dtype(-1) * iou_weights * x_std * loss_weight;
    bottom_diff[i * 4 + 1] = (dt + db) * static_cast<Dtype>(cls_num) / static_cast<Dtype>(n) * 
        Dtype(-1) * iou_weights * y_std * loss_weight;
    bottom_diff[i * 4 + 2] = ((-0.5 * dl) + (0.5 * dr)) * exp(bbox_pred[i * 4 + 2] * w_std) * 
        static_cast<Dtype>(cls_num) / static_cast<Dtype>(n) * Dtype(-1) * iou_weights * w_std * loss_weight;
    bottom_diff[i * 4 + 3] = ((-0.5 * dt) + (0.5 * db)) * exp(bbox_pred[i * 4 + 3] * h_std) * 
        static_cast<Dtype>(cls_num) / static_cast<Dtype>(n) * Dtype(-1) * iou_weights * h_std * loss_weight;
    if (clip && bbox_pred[i * 4 + 2] > clip_bound){
        bottom_diff[i * 4 + 2] = 0;
    }
    if (clip && bbox_pred[i * 4 + 3] > clip_bound){
        bottom_diff[i * 4 + 3] = 0;
    }
  
  }
}

template <typename Dtype>
void GIouLossLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
    const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  //NOT_IMPLEMENTED;
  if (propagate_down[0]) {
    norm_count_ = bottom[0]->num();
    int cls_num = channel_ / 4;
    int loop_count = norm_count_ * cls_num;
    CHECK_GE(norm_count_, 0);
    
    if (norm_count_ == 0) {
      norm_count_ = Dtype(1);
      loop_count = Dtype(2);
    }
    GIouBackward<Dtype><<<CAFFE_GET_BLOCKS(loop_count), CAFFE_CUDA_NUM_THREADS>>>(
        loop_count, bottom[0]->gpu_data(), pred_.gpu_data(), gt_.gpu_data(),
        bottom[2]->gpu_data(), bottom[3]->gpu_data(), bottom[0]->mutable_gpu_diff(), 
        clip_, clip_bound_, x_std_, y_std_, w_std_, h_std_, top[0]->cpu_diff()[0], cls_num);
    CUDA_POST_KERNEL_CHECK;
  }
}

INSTANTIATE_LAYER_GPU_FUNCS(GIouLossLayer);

}  // namespace caffe
