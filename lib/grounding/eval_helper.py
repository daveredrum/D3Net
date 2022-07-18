# Copyright (c) Facebook, Inc. and its affiliates.
# 
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import torch
import torch.nn as nn
import numpy as np
import sys
import os

from lib.utils.bbox import get_aabb3d_iou

def eval_ref_one_sample(pred_bbox, gt_bbox):
    """ Evaluate one grounding prediction

    Args:
        pred_bbox: 8 corners of prediction bounding box, (8, 3)
        gt_bbox: 8 corners of ground truth bounding box, (8, 3)
    Returns:
        iou: intersection over union score
    """

    iou = get_aabb3d_iou(pred_bbox.detach().cpu().numpy(), gt_bbox.detach().cpu().numpy())

    return iou

def get_eval(data_dict, grounding=True, use_lang_classifier=False):
    """ Loss functions

    Args:
        data_dict: dict
        config: dataset config instance
        grounding: flag (False/True)
        post_processing: config dict
    Returns:
        loss: pytorch scalar tensor
        data_dict: dict
    """

    objectness_mask_preds = data_dict["proposal_batch_mask"].long()

    # construct valid mask
    pred_masks = (objectness_mask_preds == 1).float()

    # expand
    cluster_preds = data_dict["cluster_ref"]
    batch_size, num_proposals = cluster_preds.shape
    chunk_size = batch_size // objectness_mask_preds.shape[0]
    pred_masks = pred_masks.unsqueeze(1).repeat(1, chunk_size, 1).reshape(-1, num_proposals)

    cluster_preds = torch.argmax(cluster_preds, 1).float().unsqueeze(1).repeat(1, num_proposals)
    preds = torch.zeros(pred_masks.shape).type_as(cluster_preds)
    preds = preds.scatter_(1, cluster_preds.long(), 1)
    cluster_preds = preds
    cluster_labels = data_dict["cluster_labels"].float()
    # cluster_labels *= label_masks.unsqueeze(1).repeat(1, chunk_size, 1).reshape(-1, num_proposals)
    
    # compute classification scores
    corrects = torch.sum((cluster_preds == 1) * (cluster_labels == 1), dim=1).float()
    labels = torch.ones(corrects.shape[0]).type_as(cluster_labels)
    ref_acc = corrects / (labels + 1e-8)
    
    # store
    data_dict["ref_acc"] = ref_acc.cpu().numpy().tolist()
    data_dict["ref_acc_mean"] = ref_acc.mean()

    # compute localization metrics
    pred_ref = torch.argmax(data_dict["cluster_ref"] * pred_masks, 1) # (B,)
    # store the calibrated predictions and masks
    data_dict["cluster_ref"] = data_dict["cluster_ref"] * pred_masks

    pred_bbox_corners = data_dict["proposal_bbox_batched"] # (B, num_proposal, 8, 3)

    pred_bbox_corners = pred_bbox_corners.unsqueeze(1).repeat(1, chunk_size, 1, 1, 1).reshape(-1, num_proposals, 8, 3)
    
    gt_ref = torch.argmax(cluster_labels, 1)
    gt_bbox_corners = data_dict["ref_box_corner_label"] # (B, C, 8, 3)
    gt_bbox_corners = gt_bbox_corners.reshape(-1, 8, 3)

    # eval pred
    ious = []
    best_ious = [] # best bbox during training, i.e. pseudo-GT
    multiple = []
    others = []
    pred_bboxes = []
    gt_bboxes = []
    for i in range(pred_ref.shape[0]):
        # compute the iou
        pred_ref_idx = pred_ref[i]
        pred_bbox = pred_bbox_corners[i, pred_ref_idx]
        gt_bbox = gt_bbox_corners[i]
        iou = eval_ref_one_sample(pred_bbox, gt_bbox)
        ious.append(iou)

        gt_ref_idx = gt_ref[i]
        best_bbox = pred_bbox_corners[i, gt_ref_idx]
        iou = eval_ref_one_sample(best_bbox, gt_bbox)
        best_ious.append(iou)

        # NOTE: get_3d_box() will return problematic bboxes
        pred_bboxes.append(pred_bbox.unsqueeze(0))
        gt_bboxes.append(gt_bbox.unsqueeze(0))

        # construct the multiple mask
        unique_multiple_labels = data_dict["unique_multiple"].reshape(-1)
        multiple.append(unique_multiple_labels[i].item())

        # construct the others mask
        object_cat_labels = data_dict["object_cat"].reshape(-1)
        flag = 1 if object_cat_labels[i] == 17 else 0
        others.append(flag)

    # lang
    if grounding and use_lang_classifier:
        data_dict["lang_acc"] = (torch.argmax(data_dict["lang_scores"], 1) == object_cat_labels).float().mean()
    else:
        data_dict["lang_acc"] = torch.zeros(1)[0].type_as(cluster_preds)

    ious = torch.tensor(ious).type_as(cluster_preds)
    best_ious = torch.tensor(best_ious).type_as(cluster_preds)
    pred_bboxes = torch.cat(pred_bboxes, dim=0)
    gt_bboxes = torch.cat(gt_bboxes, dim=0)

    # store
    data_dict["ref_iou"] = ious
    data_dict["best_ious"] = best_ious
    data_dict["ref_iou_mean"] = ious.mean()
    data_dict["best_ious_mean"] = best_ious.mean()
    data_dict["ref_iou_rate_0.25"] = ious[ious >= 0.25].shape[0] / ious.shape[0]
    data_dict["ref_iou_rate_0.5"] = ious[ious >= 0.5].shape[0] / ious.shape[0]
    data_dict["ref_multiple_mask"] = multiple
    data_dict["ref_others_mask"] = others
    data_dict["pred_bboxes"] = pred_bboxes
    data_dict["gt_bboxes"] = gt_bboxes

    return data_dict
