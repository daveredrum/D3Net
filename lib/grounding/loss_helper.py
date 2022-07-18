import os
import sys
import torch

import torch.nn as nn
import numpy as np

from lib.grounding.loss import SoftmaxRankingLoss, ContrastiveLoss
from lib.utils.bbox import get_aabb3d_iou, get_aabb3d_iou_batch


def get_grounding_loss(data_dict, grounding=True, use_oracle=False, is_frozen=False, use_rl=False, loss="cross_entropy"):
    """ Compute cluster grounding loss
    Args:
        data_dict: dict (read-only)
    Returns:
        ref_loss, lang_loss, cluster_probs, cluster_labels
    """

    if grounding:

        # unpack
        if use_rl:
            sampled_preds = data_dict["cluster_ref"]["sampled"] # (B, num_proposal)
            baseline_preds = data_dict["cluster_ref"]["baseline"] # (B, num_proposal)
            sampled_topn = data_dict["sampled_topn"] # e.g. 3

            bbox_features = data_dict["proposal_feats_batched"]
            
            # predicted bbox
            pred_bbox_corners = data_dict["proposal_bbox_batched"] # (B, num_proposal, 8, 3)

            # expand
            batch_size, num_proposals = sampled_preds.shape
            chunk_size = batch_size // (pred_bbox_corners.shape[0] * sampled_topn)
            pred_bbox_corners = pred_bbox_corners.unsqueeze(1).repeat(1, sampled_topn, 1, 1, 1)
            pred_bbox_corners = pred_bbox_corners.reshape(-1, num_proposals, 8, 3)
            pred_bbox_corners = pred_bbox_corners.unsqueeze(1).repeat(1, chunk_size, 1, 1, 1)
            pred_bbox_corners = pred_bbox_corners.reshape(-1, num_proposals, 8, 3)

            # ground truth bbox
            gt_bbox_corners = data_dict["ref_box_corner_label"] # (B, C, 8, 3)
            gt_bbox_corners = gt_bbox_corners.reshape(-1, 8, 3)

            assert pred_bbox_corners.shape[0] == gt_bbox_corners.shape[0]

            # compute the iou score for all predictd positive ref
            labels = np.zeros((batch_size, num_proposals))
            for i in range(batch_size):
                # convert the bbox parameters to bbox corners
                ious = get_aabb3d_iou_batch(
                    pred_bbox_corners[i].detach().cpu().numpy(), 
                    gt_bbox_corners[i].unsqueeze(0).repeat(num_proposals, 1, 1).detach().cpu().numpy()
                )
                labels[i, ious.argmax()] = 1 # treat the bbox with highest iou score as the gt

            cluster_labels = torch.FloatTensor(labels).type_as(sampled_preds)

            sampled_ious = []
            baseline_ious = []
            best_ious = []
            
            sampled_ref = sampled_preds.argmax(-1) # batch_size
            baseline_ref = baseline_preds.argmax(-1) # batch_size
            label_ref = cluster_labels.argmax(-1) # batch_size
            for i in range(batch_size):
                gt_bbox = gt_bbox_corners[i] # 8, 3

                sampled_ref_idx = sampled_ref[i]
                sampled_bbox = pred_bbox_corners[i, sampled_ref_idx] # 8, 3
                sampled_ious.append(get_aabb3d_iou(sampled_bbox.detach().cpu().numpy(), gt_bbox.detach().cpu().numpy()))

                baseline_ref_idx = baseline_ref[i]
                baseline_bbox = pred_bbox_corners[i, baseline_ref_idx] # 8, 3
                baseline_ious.append(get_aabb3d_iou(baseline_bbox.detach().cpu().numpy(), gt_bbox.detach().cpu().numpy()))

                label_ref_idx = label_ref[i]
                best_bbox = pred_bbox_corners[i, label_ref_idx] # 8, 3
                best_ious.append(get_aabb3d_iou(best_bbox.detach().cpu().numpy(), gt_bbox.detach().cpu().numpy()))

            
            data_dict["cluster_labels"] = cluster_labels

            if loss == "cross_entropy":
                # grounding loss
                criterion = SoftmaxRankingLoss(is_reduce=False)

                # loss without reduction
                sampled_loss = criterion(sampled_preds, cluster_labels.float().clone())
                baseline_loss = criterion(baseline_preds, cluster_labels.float().clone())
                # loss = loss_all.mean() 
            elif loss == "contrastive":
                criterion = ContrastiveLoss(margin=0.2, gamma=5)

                # loss without reduction
                sampled_loss = torch.zeros(batch_size).type_as(sampled_preds)
                baseline_loss = torch.zeros(batch_size).type_as(sampled_preds)

                for batch_id in range(batch_size):
                    batch_label = cluster_labels[batch_id]
                    batch_sampled_pred = sampled_preds[batch_id]
                    batch_baseline_pred = baseline_preds[batch_id]

                    sampled_loss[batch_id] = criterion(batch_sampled_pred, batch_label)
                    baseline_loss[batch_id] = criterion(batch_baseline_pred, batch_label)

            # grounding acc
            sampled_acc = (sampled_ref == label_ref).sum().float() / label_ref.shape[0]
            baseline_acc = (baseline_ref == label_ref).sum().float() / label_ref.shape[0]

            # iou rate
            sampled_ious = torch.tensor(sampled_ious).type_as(bbox_features)
            ref_acc_0_25 = sampled_ious[sampled_ious >= 0.25].shape[0] / sampled_ious.shape[0]
            ref_acc_0_5 = sampled_ious[sampled_ious >= 0.5].shape[0] / sampled_ious.shape[0]

            best_ious = torch.tensor(best_ious).type_as(bbox_features)

            # store
            data_dict["ref_loss"] = sampled_loss.mean()
            data_dict["ref_sampled_loss"] = sampled_loss
            data_dict["ref_baseline_loss"] = baseline_loss
            data_dict["ref_acc_mean"] = sampled_acc
            data_dict["ref_sampled_acc"] = sampled_acc
            data_dict["ref_sampled_acc_all"] = (sampled_ref == label_ref).float() # 0 and 1
            data_dict["ref_baseline_acc"] = baseline_acc
            data_dict["ref_baseline_acc_all"] = (baseline_ref == label_ref).float() # 0 and 1
            data_dict["ref_iou_mean"] = sampled_ious.mean()
            data_dict["best_ious_mean"] = best_ious.mean()
            data_dict["ref_iou_rate_0.25"] = ref_acc_0_25
            data_dict["ref_iou_rate_0.5"] = ref_acc_0_5

        else:
            bbox_features = data_dict["proposal_feats_batched"]
            cluster_preds = data_dict["cluster_ref"] # (B*C, num_proposal)

            # predicted bbox
            pred_bbox_corners = data_dict["proposal_bbox_batched"] # (B, num_proposal, 8, 3)

            # expand
            batch_size, num_proposals = cluster_preds.shape
            chunk_size = batch_size // pred_bbox_corners.shape[0]
            pred_bbox_corners = pred_bbox_corners.unsqueeze(1).repeat(1, chunk_size, 1, 1, 1).reshape(batch_size, num_proposals, 8, 3)

            # ground truth bbox
            gt_bbox_corners = data_dict["ref_box_corner_label"] # (B, 8, 3)
            gt_bbox_corners = gt_bbox_corners.reshape(batch_size, 8, 3)

            # compute the iou score for all predictd positive ref
            labels = np.zeros((batch_size, num_proposals))
            for i in range(batch_size):
                # convert the bbox parameters to bbox corners
                ious = get_aabb3d_iou_batch(
                    pred_bbox_corners[i].detach().cpu().numpy(), 
                    gt_bbox_corners[i].unsqueeze(0).repeat(num_proposals, 1, 1).detach().cpu().numpy()
                )
                labels[i, ious.argmax()] = 1 # treat the bbox with highest iou score as the gt

            cluster_labels = torch.FloatTensor(labels).type_as(cluster_preds)

            # grounding loss
            if loss == "cross_entropy":
                criterion = SoftmaxRankingLoss()
                loss = criterion(cluster_preds, cluster_labels.float().clone())
            elif loss == "contrastive":
                criterion = ContrastiveLoss()
                loss = torch.zeros(batch_size).type_as(cluster_preds)

                for i in range(batch_size):
                    label = cluster_labels[i]
                    pred = cluster_preds[i]
                    loss[i] = criterion(pred, label)

                loss = loss.mean()
            else:
                raise NotImplementedError("invalid loss type")

            data_dict["cluster_labels"] = cluster_labels

            # grounding acc
            cluster_labels = cluster_labels.argmax(-1) # (B,)
            cluster_preds = cluster_preds.argmax(-1) # (B,)
            cluster_acc = (cluster_preds == cluster_labels).sum().float() / cluster_labels.shape[0]

            # eval IoU rate
            ious = []
            best_ious = [] # best bbox during training, i.e. pseudo-GT
            for i in range(cluster_preds.shape[0]):
                # compute the iou
                pred_ref_idx = cluster_preds[i]
                pred_bbox = pred_bbox_corners[i, pred_ref_idx]
                gt_bbox = gt_bbox_corners[i]
                iou = get_aabb3d_iou(pred_bbox.detach().cpu().numpy(), gt_bbox.detach().cpu().numpy())
                ious.append(iou)

                gt_ref_idx = cluster_labels[i]
                best_bbox = pred_bbox_corners[i, gt_ref_idx]
                iou = get_aabb3d_iou(best_bbox.detach().cpu().numpy(), gt_bbox.detach().cpu().numpy())
                best_ious.append(iou)


            ious = torch.FloatTensor(ious).type_as(bbox_features)
            best_ious = torch.FloatTensor(best_ious).type_as(bbox_features)

            # iou rate
            ref_acc_0_25 = ious[ious >= 0.25].shape[0] / ious.shape[0]
            ref_acc_0_5 = ious[ious >= 0.5].shape[0] / ious.shape[0]

            # store
            data_dict["ref_loss"] = loss if not is_frozen else torch.zeros(1)[0].type_as(bbox_features)
            data_dict["ref_acc_mean"] = cluster_acc
            data_dict["ref_iou_mean"] = ious.mean()
            data_dict["best_ious_mean"] = best_ious.mean()
            data_dict["ref_iou_rate_0.25"] = ref_acc_0_25
            data_dict["ref_iou_rate_0.5"] = ref_acc_0_5
    else:
        # pc = data_dict["point_clouds"]
        bbox_features = data_dict["bbox_feature"]

        data_dict["ref_loss"] = torch.zeros(1)[0].type_as(bbox_features)
        data_dict["ref_acc_mean"] = torch.zeros(1)[0].type_as(bbox_features)
        data_dict["ref_iou_mean"] = torch.zeros(1)[0].type_as(bbox_features)
        data_dict["best_iou_mean"] = torch.zeros(1)[0].type_as(bbox_features)
        data_dict["ref_acc_0_25"] = torch.zeros(1)[0].type_as(bbox_features)
        data_dict["ref_acc_0_5"] = torch.zeros(1)[0].type_as(bbox_features)

    # exposed loss
    loss = data_dict["ref_loss"]

    return loss, data_dict

def get_lobjcls_loss(data_dict, lang_cls=True, is_frozen=False, use_rl=False):
    """ Compute object classification loss

    Args:
        data_dict: dict (read-only)

    Returns:
        cls_loss, cls_acc
    """

    if lang_cls:
        
        if use_rl:

            sampled_preds = data_dict["lang_scores"]["sampled"] # (B * chunk_size, num_cls)
            baseline_preds = data_dict["lang_scores"]["baseline"] # (B * chunk_size, num_cls)

            targets = data_dict.get("ref_cat_label", data_dict["object_cat"]) # (B, chunk_size)
            # sampled_topn = data_dict["sampled_topn"] # e.g. 3
            targets = targets.reshape(-1)

            # classification loss
            criterion = nn.CrossEntropyLoss(reduction="none")
            assert targets.shape[0] == sampled_preds.shape[0]
            sampled_cls_loss = criterion(sampled_preds, targets.long())
            baseline_cls_loss = criterion(baseline_preds, targets.long())

            # classification acc
            sampled_preds = sampled_preds.argmax(-1) # (B,)
            sampled_acc = (sampled_preds == targets).sum().float() / targets.shape[0]

            baseline_preds = baseline_preds.argmax(-1) # (B,)
            baseline_acc = (baseline_preds == targets).sum().float() / targets.shape[0]

            # store
            data_dict["lang_loss"] = sampled_cls_loss.mean()
            data_dict["sampled_lang_loss"] = sampled_cls_loss
            data_dict["baseline_lang_loss"] = baseline_cls_loss
            data_dict["lang_acc"] = sampled_acc
            data_dict["lang_sampled_acc"] = sampled_acc
            data_dict["lang_baseline_acc"] = baseline_acc

        else:

            # unpack
            preds = data_dict["lang_scores"] # (B, num_cls)
            targets = data_dict.get("ref_cat_label", data_dict["object_cat"]) # (B,)
            targets = targets.reshape(-1)
            
            # classification loss
            criterion = nn.CrossEntropyLoss()
            cls_loss = criterion(preds, targets)

            # classification acc
            preds = preds.argmax(-1) # (B,)
            cls_acc = (preds == targets).sum().float() / targets.shape[0]

            # store
            data_dict["lang_loss"] = cls_loss if not is_frozen else torch.zeros(1)[0].type_as(preds)
            data_dict["lang_acc"] = cls_acc

    else:
        # pc = data_dict["point_clouds"]
        bbox_features = data_dict["bbox_feature"]

        data_dict["lang_loss"] = torch.zeros(1)[0].type_as(bbox_features)
        data_dict["lang_acc"] = torch.zeros(1)[0].type_as(bbox_features)

    # exposed loss
    loss = data_dict["lang_loss"]

    return loss, data_dict

def get_loss(data_dict, use_oracle, grounding, use_lang_classifier, use_rl):
    """ Loss functions

    Args:
        data_dict: dict
        config: dataset config instance
        grounding: flag (False/True)
    Returns:
        loss: pytorch scalar tensor
        data_dict: dict
    """


    # grounding loss
    _, data_dict = get_grounding_loss(
        data_dict, 
        grounding, 
        use_oracle=use_oracle, 
        is_frozen=False, 
        use_rl=use_rl, 
        loss="cross_entropy"
    )
    _, data_dict = get_lobjcls_loss(
        data_dict, 
        lang_cls=use_lang_classifier, 
        is_frozen=False, 
        use_rl=use_rl
    )

    loss = data_dict["ref_loss"] + data_dict["lang_loss"]

    return loss, data_dict