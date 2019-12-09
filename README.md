# GIou_loss_caffe
Caffe version Generalized Iou loss and Complete Iou loss Implementation for Faster RCNN/FPN bbox regression. (https://arxiv.org/abs/1902.09630)
(https://github.com/generalized-iou)
(https://arxiv.org/abs/1911.08287)
(https://github.com/Zzh-tju/DIoU)
# Usage
### caffe.proto
```
optional GIouLossParameter giou_loss_param = 1490;
message GIouLossParameter {
  optional float x_std = 1 [default = 0.1]; // BBOX_NORMALIZE_STDS
  optional float y_std = 2 [default = 0.1]; 
  optional float w_std = 3 [default = 0.2]; 
  optional float h_std = 4 [default = 0.2]; 
  optional float clip_bound = 5 [default = 4.1352]; // log(1000/ 16)
  optional bool clip = 6 [default = true]; // Clip bounding box transformation predictions to prevent exp() from overflowing
}
```
### train.prototxt
Simply replace
```
layer {
  bottom: "bbox_pred"
  bottom: "proposal_targets"
  bottom: "box_inside_weights"
  bottom: "box_outside_weights"
  top: "loss_bbox"
  name: "loss_bbox"
  type: "SmoothL1Loss"
  loss_weight: 1
  propagate_down: 1
  propagate_down: 0
  propagate_down: 0
  propagate_down: 0
}
```
By
```
layer {
  bottom: "bbox_pred"
  bottom: "proposal_targets"
  bottom: "box_inside_weights"
  bottom: "box_outside_weights"
  top: "loss_bbox"
  name: "loss_bbox"
  type: "GIouLoss"
  giou_loss_param{
    clip: true # or false
    x_std: 0.1 # __C.TRAIN.BBOX_NORMALIZE_STDS = (0.1, 0.1, 0.2, 0.2) in py-faster-rcnn
    y_std: 0.1
    w_std: 0.2
    h_std: 0.2
  }
  loss_weight: 10
  propagate_down: 1
  propagate_down: 0
  propagate_down: 0
  propagate_down: 0
}
```
# COCO2017 validation Results
### Baseline model (SmoothL1Loss)
```
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.290
 Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.458
 Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.311
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.127
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.320
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.413
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ] = 0.271
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ] = 0.388
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.394
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.202
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.437
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.540
```
### Current model (GIouLoss)
```
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.292
 Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.460
 Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.308
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.126
 Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.328
 Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.413
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ] = 0.275
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ] = 0.401
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.408
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.211
 Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.462
 Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.553
```
### more experiments
TODO
