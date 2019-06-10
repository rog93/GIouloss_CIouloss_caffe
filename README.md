# GIou_loss_caffe
Caffe version GIou loss Implementation for Faster RCNN/FPN bbox regression
# Usage
### caffe.proto
```
optional GIouLossParameter giou_loss_param = 1490;
message GIouLossParameter {
  optional float x_std = 1 [default = 0.1];
  optional float y_std = 2 [default = 0.1];
  optional float w_std = 3 [default = 0.2];
  optional float h_std = 4 [default = 0.2];
  optional float clip_bound = 5 [default = 4.1352]; // log(1000/ 16)
  optional bool clip = 6 [default = true];
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
    clip: true #or false
    x_std: 0.1
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
# Results
TODO
