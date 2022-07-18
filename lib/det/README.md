# Object Detection Evaluation

## Usage

```python

from ap_helper import APCalculator, parse_predictions, parse_groundtruths

# config
POST_DICT = {
    "remove_empty_box": True, 
    "use_3d_nms": True, 
    "nms_iou": 0.25,
    "use_old_type_nms": False, 
    "cls_nms": True, 
    "per_class_proposal": True,
    "conf_thresh": 0.05,
    "dataset_config": DC
}
AP_IOU_THRESHOLDS = [0.25, 0.5]
AP_CALCULATOR_LIST = [APCalculator(iou_thresh, DC.class2type) for iou_thresh in AP_IOU_THRESHOLDS]

for data_dict in tqdm(dataloader):
    for key in data_dict:
        data_dict[key] = data_dict[key].cuda()

    # feed
    with torch.no_grad():
        data_dict = model(data_dict) # your network

    # NOTE the following data must be included after the steps above
    #
    # predictions ->
    # "point_clouds": point cloud coordinates consistent with the predictions, BxNx3
    # "bbox_corners": bounding box corners, BxKx8x3
    # "bbox_masks": bounding box objectness scores, BxKx2
    # "sem_cls_scores": bounding box semantic predictions, BxKxCls
    #
    # ground truth ->
    # "bbox_corner_labels: GT bounding box corners, BxMx8x3
    # "bbox_mask_labels": GT objectness masks (in case there are zero-paddings), BxM
    # "sem_cls_labels": GT bounding box semantic labels, BxM

    batch_pred_map_cls = parse_predictions(data_dict, POST_DICT) 
    batch_gt_map_cls = parse_groundtruths(data_dict, POST_DICT) 
    for ap_calculator in AP_CALCULATOR_LIST:
        ap_calculator.step(batch_pred_map_cls, batch_gt_map_cls)

# aggregate object detection results and report
for i, ap_calculator in enumerate(AP_CALCULATOR_LIST):
    print()
    print("-"*10, "iou_thresh: %f"%(AP_IOU_THRESHOLDS[i]), "-"*10)
    metrics_dict = ap_calculator.compute_metrics()
    for key in metrics_dict:
        print("eval %s: %f"%(key, metrics_dict[key]))

```