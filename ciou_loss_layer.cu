#include <algorithm>
#include <cfloat>
#include <vector>
#include <cmath>
#include "caffe/util/math_functions.hpp"
#include "caffe/layers/ciou_loss_layer.hpp"

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


    out[i * 8 + 0] = dx;
    out[i * 8 + 1] = dy;
    out[i * 8 + 2] = exp(dw);
    out[i * 8 + 3] = exp(dh);
    out[i * 8 + 4] = x1;
    out[i * 8 + 5] = y1;
    out[i * 8 + 6] = x2;
    out[i * 8 + 7] = y2;

  }
}

template <typename Dtype>
__global__ void CIouForward(const int n, const Dtype* pred, const Dtype* gt, 
  const Dtype* bbox_inside_weights, const Dtype* bbox_outside_weights, Dtype* loss_data) {
  const Dtype PI=3.14159265358979f;
  CUDA_KERNEL_LOOP(index, n) {
    int i = index;
    
    Dtype x_orig = pred[i * 8 + 0];
    Dtype y_orig = pred[i * 8 + 1];
    Dtype w_orig = pred[i * 8 + 2];
    Dtype h_orig = pred[i * 8 + 3];

    Dtype x1_orig = pred[i * 8 + 4];
    Dtype y1_orig = pred[i * 8 + 5];
    Dtype x2_orig = pred[i * 8 + 6];
    Dtype y2_orig = pred[i * 8 + 7];

    Dtype x1 = min(x1_orig, x2_orig);
    Dtype y1 = min(y1_orig, y2_orig);
    Dtype x2 = max(x1_orig, x2_orig);
    Dtype y2 = max(y1_orig, y2_orig);

    
    Dtype gx = gt[i * 8 + 0];
    Dtype gy = gt[i * 8 + 1];
    Dtype gw = gt[i * 8 + 2];
    Dtype gh = gt[i * 8 + 3];

    Dtype gx1 = gt[i * 8 + 4];
    Dtype gy1 = gt[i * 8 + 5];
    Dtype gx2 = gt[i * 8 + 6];
    Dtype gy2 = gt[i * 8 + 7];

    
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

    Dtype c = (xc2 - xc1)*(xc2 - xc1) + (yc2 - yc1) *(yc2 - yc1) + 1e-7;
    Dtype u = (x_orig - gx)*(x_orig - gx) + (y_orig - gy)*(y_orig - gy) ;
    Dtype d = u / c;
    Dtype ar_gt = gw / gh;
    Dtype ar_pred = w_orig / h_orig;

    Dtype ar_loss = Dtype(4) / (PI * PI) * (atan(ar_gt) - atan(ar_pred)) * (atan(ar_gt) - atan(ar_pred));
    Dtype alpha = ar_loss / (1 - iouk + ar_loss + 1e-7); 
    
    Dtype ciou_term = d + alpha * ar_loss; 
    Dtype miouk = iouk - ciou_term; 

    Dtype iou_weights = (bbox_inside_weights[i * 4 + 0] + bbox_inside_weights[i * 4 + 1] + 
			bbox_inside_weights[i * 4 + 2] + bbox_inside_weights[i * 4 + 3]) * 
			(bbox_outside_weights[i * 4 + 0] + bbox_outside_weights[i * 4 + 1] + 
			bbox_outside_weights[i * 4 + 2] + bbox_outside_weights[i * 4 + 3]) / Dtype(16);


    loss_data[i] = (Dtype(1) - miouk) * iou_weights;

  }
}


template <typename Dtype>
void CIouLossLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
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

  CIouForward<Dtype><<<CAFFE_GET_BLOCKS(loop_count), CAFFE_CUDA_NUM_THREADS>>>(
      loop_count, pred_.gpu_data(), gt_.gpu_data(), bottom[2]->gpu_data(), bottom[3]->gpu_data(), diff_.mutable_gpu_data());

  CUDA_POST_KERNEL_CHECK;

  Dtype loss;

  caffe_gpu_asum(loop_count, diff_.gpu_data(), &loss);

  top[0]->mutable_cpu_data()[0] = loss / norm_count_;
}

template <typename Dtype>
__global__ void CIouBackward(const int n, const Dtype* bbox_pred, const Dtype* pred, const Dtype* gt, 
  const Dtype* bbox_inside_weights, const Dtype* bbox_outside_weights, Dtype* bottom_diff, const bool clip, 
  const Dtype clip_bound, const Dtype x_std, const Dtype y_std, const Dtype w_std, const Dtype h_std, 
  const Dtype loss_weight, int cls_num) {
  const Dtype PI=3.14159265358979f;
  CUDA_KERNEL_LOOP(index, n) {
    int i = index;
    Dtype x_orig = pred[i * 8 + 0];
    Dtype y_orig = pred[i * 8 + 1];
    Dtype w_orig = pred[i * 8 + 2];
    Dtype h_orig = pred[i * 8 + 3];

    Dtype x1_orig = pred[i * 8 + 4];
    Dtype y1_orig = pred[i * 8 + 5];
    Dtype x2_orig = pred[i * 8 + 6];
    Dtype y2_orig = pred[i * 8 + 7];

    
    Dtype gx = gt[i * 8 + 0];
    Dtype gy = gt[i * 8 + 1];
    Dtype gw = gt[i * 8 + 2];
    Dtype gh = gt[i * 8 + 3];

    Dtype gx1 = gt[i * 8 + 4];
    Dtype gy1 = gt[i * 8 + 5];
    Dtype gx2 = gt[i * 8 + 6];
    Dtype gy2 = gt[i * 8 + 7];

    Dtype x1 = min(x1_orig, x2_orig);
    Dtype y1 = min(y1_orig, y2_orig);
    Dtype x2 = max(x1_orig, x2_orig);
    Dtype y2 = max(y1_orig, y2_orig);


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
    Dtype S = (x_orig - gx) *(x_orig - gx) + (y_orig - gy) * (y_orig - gy);
    
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
    
    Dtype dt = 0;
    Dtype db = 0;
    Dtype dl = 0;
    Dtype dr = 0;

    dt = y1_orig < y2_orig ? p_dt : p_db;
    db = y1_orig < y2_orig ? p_db : p_dt;
    dl = x1_orig < x2_orig ? p_dl : p_dr;
    dr = x1_orig < x2_orig ? p_dr : p_dl;  
 
    Dtype p_dx = 0;
    Dtype p_dy = 0;
    Dtype p_dw = 0;
    Dtype p_dh = 0;
    
    p_dx = dl + dr;         
    p_dy = dt + db;
    p_dw = (dr - dl) * Dtype(0.5);
    p_dh = (db - dt) * Dtype(0.5);  

    Dtype Cl = min(x1, gx1);
    Dtype Ct = min(y1, gy1);
    Dtype Cr = max(x2, gx2);
    Dtype Cb = max(y2, gy2);  
    
    Dtype Cw = Cr - Cl;
    Dtype Ch = Cb - Ct;
    
    Dtype C = Cw * Cw + Ch * Ch;

    Dtype dCt_dy = y1 < gy1 ? Dtype(-1) : 0;
    Dtype dCb_dy = y2 > gy2 ? Dtype(1) : 0;
    Dtype dCl_dx = x1 < gx1 ? Dtype(-1) : 0;
    Dtype dCr_dx = x2 > gx2 ? Dtype(1) : 0;
    
    Dtype dt_dy = 0;
    Dtype db_dy = 0;

    Dtype dl_dx = 0;
    Dtype dr_dx = 0;

    dt_dy = y1_orig < y2_orig ? dCt_dy : dCb_dy;
    db_dy = y1_orig < y2_orig ? dCb_dy : dCt_dy;

    dl_dx = x1_orig < x2_orig ? dCl_dx : dCr_dx;
    dr_dx = x1_orig < x2_orig ? dCr_dx : dCl_dx;

    Dtype dCw_dx = dr_dx + dl_dx;
    Dtype dCh_dy = db_dy + dt_dy;
    
    Dtype dCw_dw = Dtype(0.5) * (dr_dx - dl_dx);
    Dtype dCh_dh = Dtype(0.5) * (db_dy - dt_dy);   

    Dtype ar_gt = gw / gh;
    Dtype ar_pred = w_orig / h_orig;

    Dtype ar_loss = Dtype(4) / (PI * PI) * (atan(ar_gt) - atan(ar_pred)) * (atan(ar_gt) - atan(ar_pred));
    Dtype alpha = ar_loss / (Dtype(1) - I/U + ar_loss + 1e-7); //no grad

    Dtype ar_dw = Dtype(8)/(PI*PI)*(atan(ar_gt)-atan(ar_pred))*h_orig;
    Dtype ar_dh = -Dtype(8)/(PI*PI)*(atan(ar_gt)-atan(ar_pred))*w_orig;

    if (C > 0) {
      p_dx += (Dtype(2)*(gx -x_orig)*C+(Dtype(2)*Cw*dCw_dx)*S) / (C * C);
      p_dy += (Dtype(2)*(gy -y_orig)*C+(Dtype(2)*Ch*dCh_dy)*S) / (C * C);
      p_dw += (Dtype(2)*Cw*dCw_dw)*S / (C * C) + alpha * ar_dw;
      p_dh += (Dtype(2)*Ch*dCh_dh)*S / (C * C) + alpha * ar_dh;
    }
    
    Dtype iou_weights = (bbox_inside_weights[i * 4 + 0] + bbox_inside_weights[i * 4 + 1] + 
			bbox_inside_weights[i * 4 + 2] + bbox_inside_weights[i * 4 + 3]) * 
			(bbox_outside_weights[i * 4 + 0] + bbox_outside_weights[i * 4 + 1] + 
			bbox_outside_weights[i * 4 + 2] + bbox_outside_weights[i * 4 + 3]) / Dtype(16);
    
    bottom_diff[i * 4 + 0] = 0; 
    bottom_diff[i * 4 + 1] = 0; 
    bottom_diff[i * 4 + 2] = 0; 
    bottom_diff[i * 4 + 3] = 0; 
    
    bottom_diff[i * 4 + 0] = (p_dx) * static_cast<Dtype>(cls_num) / static_cast<Dtype>(n) * 
        Dtype(-1) * iou_weights * x_std * loss_weight;
    bottom_diff[i * 4 + 1] = (p_dy) * static_cast<Dtype>(cls_num) / static_cast<Dtype>(n) * 
        Dtype(-1) * iou_weights * y_std * loss_weight;
    bottom_diff[i * 4 + 2] = (p_dw) * exp(bbox_pred[i * 4 + 2] * w_std) * 
        static_cast<Dtype>(cls_num) / static_cast<Dtype>(n) * Dtype(-1) * iou_weights * w_std * loss_weight;
    bottom_diff[i * 4 + 3] = (p_dh) * exp(bbox_pred[i * 4 + 3] * h_std) * 
        static_cast<Dtype>(cls_num) / static_cast<Dtype>(n) * Dtype(-1) * iou_weights * h_std * loss_weight;
    if (clip && bbox_pred[i * 4 + 2] * w_std > clip_bound){
        bottom_diff[i * 4 + 2] = 0;
    }
    if (clip && bbox_pred[i * 4 + 3] * h_std > clip_bound){
        bottom_diff[i * 4 + 3] = 0;
    }
  }
}

template <typename Dtype>
void CIouLossLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
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
    CIouBackward<Dtype><<<CAFFE_GET_BLOCKS(loop_count), CAFFE_CUDA_NUM_THREADS>>>(
        loop_count, bottom[0]->gpu_data(), pred_.gpu_data(), gt_.gpu_data(),
        bottom[2]->gpu_data(), bottom[3]->gpu_data(), bottom[0]->mutable_gpu_diff(), 
        clip_, clip_bound_, x_std_, y_std_, w_std_, h_std_, top[0]->cpu_diff()[0], cls_num);
    CUDA_POST_KERNEL_CHECK;
  }
}

INSTANTIATE_LAYER_GPU_FUNCS(CIouLossLayer);

}  // namespace caffe
