# Copyright (c) Facebook, Inc. and its affiliates.
# 
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import torch
import torch.nn as nn
import numpy as np
import sys
import os

import lib.capeval.bleu.bleu as capblue
import lib.capeval.cider.cider as capcider

def compute_caption_reward(data_dict, cap_tables, sample_topn, idx2word, dataset_data, organized_data):
    assert len(cap_tables[0]) == sample_topn

    # unpack from dataset
    # organized = dataset.organized
    # vocabulary = dataset.vocabulary
    # raw_data = dataset.scanrefer

    # unpack from data_dict
    dataset_ids = data_dict["id"] # batch_size
    chunk_ids = data_dict["chunk_ids"] # batch_size, chunk_size
    is_annotated = data_dict["annotated"] # batch_size, chunk_size

    _, chunk_size = chunk_ids.shape
    dataset_ids = dataset_ids.unsqueeze(1).repeat(1, chunk_size).reshape(-1)
    chunk_ids = chunk_ids.reshape(-1)
    is_annotated = is_annotated.reshape(-1)
    batch_size = dataset_ids.shape[0] # = batch_size * chunk_size
    
    # batch_size = len(cap_tables)

    scores = torch.zeros(batch_size, sample_topn).type_as(is_annotated).float()

    # query GTs & transform generated candidates
    valid_ids = torch.arange(batch_size).type_as(is_annotated).long()
    valid_ids = valid_ids[is_annotated == 1]

    if valid_ids.shape[0] > 0: # in case all are unannotated
        count = 0
        gts = {}
        cands = {}
        for batch_id in valid_ids:
            dataset_idx = dataset_ids[batch_id].item()
            chunk_idx = chunk_ids[batch_id].item()

            # query raw info
            raw = dataset_data[dataset_idx][chunk_idx]
            scene_id = raw["scene_id"]
            object_id = raw["object_id"]

            for sample_id in range(sample_topn):
                # key = "{}|{}|{}".format(scene_id, object_id, str(sample_id))
                key = str(count)

                # query GTs
                gt_des = []
                gt_data = organized_data[scene_id][object_id]
                for data in gt_data:
                    tokens = data["token"] + ["eos"]
                    des = " ".join(tokens)

                    gt_des.append(des)

                gts[key] = gt_des

                # transform candidates
                raw_candidates = cap_tables[batch_id]
                cand_data = raw_candidates[sample_id]
                tokens = [idx2word[str(t.item())] for t in cand_data]
                if "eos" not in tokens: tokens += ["eos"]
                cands[key] = [" ".join(tokens)]

                count += 1

        # compute scoress
        _, cider = capcider.Cider().compute_score(gts, cands)
        _, (_, _, _, bleu) = capblue.Bleu(4).compute_score(gts, cands)
        cider = np.array(cider)
        bleu = np.array(bleu)

        # aggregate
        cider_weight = 1
        bleu_weight = 0

        combined = cider_weight * cider + bleu_weight * bleu
        # # combined = bleu # HACK bleu only
        # combined = bleu + cider # HACK bleu and cider

        # store
        scores[valid_ids, :] = torch.Tensor(combined).type_as(scores).view(valid_ids.shape[0], sample_topn)

    return scores

def compute_cap_loss(data_dict, loss_opt={}):
    """ Compute cluster caption loss

    Args:
        data_dict: dict (read-only)

    Returns:
        cap_loss, cap_acc
    """

    use_rl = loss_opt.get("use_rl", False)

    if use_rl:
        sample_topn = loss_opt.get("sample_topn", 1)
        idx2word = loss_opt.get("idx2word", None)
        dataset_data = loss_opt.get("train_dataset_data", None)
        organized_data = loss_opt.get("organized_data", None)

        # unpack
        sampled_caps = data_dict["lang_cap"] # [batch_size...[sample_topn...[seq_len...]]]
        sampled_logprobs = data_dict["lang_logprob"] # [batch_size...[sample_topn...[seq_len..]]]
        baseline_caps = data_dict["baseline_cap"] # [batch_size...[seq_len..]]
        good_bbox_masks = data_dict["good_bbox_masks"].long() # batch_size
        annotated_masks = data_dict["annotated"].reshape(-1) # batch_size

        # compute caption rewards
        sampled_logprobs = torch.stack([seqprobs.sum() for beam in sampled_logprobs for seqprobs in beam]) # batch_size * sample_topn
        sampled_scores = compute_caption_reward(data_dict, sampled_caps, sample_topn, idx2word, dataset_data, organized_data).type_as(sampled_logprobs) # batch_size, sample_topn
        baseline_scores = compute_caption_reward(data_dict, baseline_caps, sample_topn, idx2word, dataset_data, organized_data).type_as(sampled_logprobs) # batch_size, sample_topn

        # construct masks
        good_bbox_masks = good_bbox_masks.unsqueeze(1).repeat(1, sample_topn) # batch_size, sample_topn
        annotated_masks = annotated_masks.unsqueeze(1).repeat(1, sample_topn) # batch_size, sample_topn

        # aggregate the caption reward by averaging over the number of samples
        caption_reward = sampled_scores - baseline_scores

        # # unpack listener rewards from loss dict
        # ref_sampled_loss = data_dict["ref_sampled_loss"].view(caption_reward.shape)
        # ref_baseline_loss = data_dict["ref_baseline_loss"].detach().view(caption_reward.shape)
        # lang_sampled_loss = data_dict["sampled_lang_loss"].view(caption_reward.shape)
        # lang_baseline_loss = data_dict["baseline_lang_loss"].detach().view(caption_reward.shape)

        # detach the ref loss from the computational graph
        # NOTE the ref loss must be exposed in the training objective
        ref_sampled_loss = data_dict["ref_sampled_loss"].detach().view(caption_reward.shape)
        ref_baseline_loss = data_dict["ref_baseline_loss"].detach().view(caption_reward.shape)
        lang_sampled_loss = data_dict["sampled_lang_loss"].detach().view(caption_reward.shape)
        lang_baseline_loss = data_dict["baseline_lang_loss"].detach().view(caption_reward.shape)

        # aggregate the listener rewards
        ref_reward = -(ref_sampled_loss - ref_baseline_loss)
        lang_reward = -(lang_sampled_loss - lang_baseline_loss)

        ref_weight = loss_opt.get("ref_reward_weight", 1)
        lang_weight = loss_opt.get("lang_reward_weight", 1)
        listener_reward = ref_weight * ref_reward + lang_weight * lang_reward

        # listener_weight = 0.1 # HACK trial for using ref loss as one of the rewards
        listener_weight = loss_opt.get("listener_reward_weight", 1)
        caption_weight = loss_opt.get("caption_reward_weight", 1)
        rewards = caption_weight * caption_reward + listener_weight * listener_reward

        # baseline = (scores.sum(1, keepdim=True) - scores) / (scores.shape[1] - 1 + 1e-8)
        # rewards = scores - baseline + listener_weight * listener_reward
        
        cap_loss = - rewards.view(-1) * sampled_logprobs * good_bbox_masks.view(-1)
        cap_loss = cap_loss.sum() / (good_bbox_masks.sum() + 1e-8)
        cap_acc = (sampled_scores * good_bbox_masks * annotated_masks).sum() / ((good_bbox_masks * annotated_masks).sum() + 1e-8)
        # cap_acc = (baseline_scores * good_bbox_masks * annotated_masks).sum() / ((good_bbox_masks * annotated_masks).sum() + 1e-8)
        
        cap_rwd = (caption_reward * good_bbox_masks).sum() / (good_bbox_masks.sum() + 1e-8)
        loc_rwd = (listener_reward * good_bbox_masks).sum() / (good_bbox_masks.sum() + 1e-8)
        ttl_rwd = (rewards * good_bbox_masks).sum() / (good_bbox_masks.sum() + 1e-8)

        # store
        data_dict["cap_rwd"] = cap_rwd
        data_dict["loc_rwd"] = loc_rwd
        data_dict["ttl_rwd"] = ttl_rwd

    else:
        max_len = loss_opt.get("max_len", 30)

        # unpack
        pred_caps = data_dict["lang_cap"] # (B, num_words - 1, num_vocabs)
        num_words = data_dict["lang_len"].reshape(-1).max()
        target_caps = data_dict["lang_ids"].reshape(-1, max_len)[:, 1:num_words] # (B, num_words - 1)
        
        good_bbox_masks = data_dict["good_bbox_masks"]
        num_good_bbox = data_dict["good_bbox_masks"].sum()
        if num_good_bbox > 0: 
            _, _, num_vocabs = pred_caps.shape

            # only apply loss on the good boxes
            pred_caps = pred_caps[good_bbox_masks]
            target_caps = target_caps[good_bbox_masks]

            # caption loss
            criterion = nn.CrossEntropyLoss(ignore_index=0)
            cap_loss = criterion(pred_caps.reshape(-1, num_vocabs), target_caps.reshape(-1))

            # caption acc
            pred_caps = pred_caps.reshape(-1, num_vocabs).argmax(-1) # num_good_bbox * (num_words - 1)
            target_caps = target_caps.reshape(-1) # num_good_bbox * (num_words - 1)
            masks = target_caps != 0
            masked_pred_caps = pred_caps[masks]
            masked_target_caps = target_caps[masks]
            cap_acc = (masked_pred_caps == masked_target_caps).sum().float() / masks.sum().float()
        else: # zero placeholder if there is no good box
            cap_loss = torch.zeros(1)[0].type_as(pred_caps)
            cap_acc = torch.zeros(1)[0].type_as(pred_caps)

        # store
        bbox_features = data_dict["bbox_feature"]

        data_dict["cap_rwd"] = torch.zeros(1)[0].type_as(bbox_features)
        data_dict["loc_rwd"] = torch.zeros(1)[0].type_as(bbox_features)
        data_dict["ttl_rwd"] = torch.zeros(1)[0].type_as(bbox_features)

    # store
    data_dict["cap_loss"] = cap_loss
    data_dict["cap_acc"] = cap_acc

    # exposed loss
    loss = data_dict["cap_loss"]
    
    return loss, data_dict

def radian_to_label(radians, num_bins=6):
    """
        convert radians to labels

        Arguments:
            radians: a tensor representing the rotation radians, (batch_size)
            radians: a binary tensor representing the valid masks, (batch_size)
            num_bins: number of bins for discretizing the rotation degrees

        Return:
            labels: a long tensor representing the discretized rotation degree classes, (batch_size)
    """

    boundaries = torch.arange(np.pi / num_bins, np.pi-1e-8, np.pi / num_bins).type_as(radians)
    labels = torch.bucketize(radians, boundaries)

    return labels

def compute_node_orientation_loss(data_dict, num_bins=6):
    object_assignment = data_dict["object_assignment"]
    edge_indices = data_dict["edge_index"]
    edge_preds = data_dict["edge_orientations"]
    num_sources = data_dict["num_edge_source"]
    num_targets = data_dict["num_edge_target"]
    batch_size, num_proposals = object_assignment.shape

    object_rotation_matrices = torch.gather(
        data_dict["scene_object_rotations"], 
        1, 
        object_assignment.view(batch_size, num_proposals, 1, 1).repeat(1, 1, 3, 3)
    ) # batch_size, num_proposals, 3, 3
    object_rotation_masks = torch.gather(
        data_dict["scene_object_rotation_masks"], 
        1, 
        object_assignment
    ) # batch_size, num_proposals
    
    preds = []
    labels = []
    masks = []
    for batch_id in range(batch_size):
        batch_rotations = object_rotation_matrices[batch_id] # num_proposals, 3, 3
        batch_rotation_masks = object_rotation_masks[batch_id] # num_proposals

        batch_num_sources = num_sources[batch_id]
        batch_num_targets = num_targets[batch_id]
        batch_edge_indices = edge_indices[batch_id, :batch_num_sources * batch_num_targets]

        source_indices = edge_indices[batch_id, 0, :batch_num_sources*batch_num_targets].long()
        target_indices = edge_indices[batch_id, 1, :batch_num_sources*batch_num_targets].long()

        source_rot = torch.index_select(batch_rotations, 0, source_indices)
        target_rot = torch.index_select(batch_rotations, 0, target_indices)

        relative_rot = torch.matmul(source_rot, target_rot.transpose(2, 1))
        relative_rot = torch.acos(torch.clamp(0.5 * (torch.diagonal(relative_rot, dim1=-2, dim2=-1).sum(-1) - 1), -1, 1))
        assert torch.isfinite(relative_rot).sum() == source_indices.shape[0]

        source_masks = torch.index_select(batch_rotation_masks, 0, source_indices)
        target_masks = torch.index_select(batch_rotation_masks, 0, target_indices)
        batch_edge_masks = source_masks * target_masks
        
        batch_edge_labels = radian_to_label(relative_rot, num_bins)
        batch_edge_preds = edge_preds[batch_id, :batch_num_sources * batch_num_targets]

        preds.append(batch_edge_preds)
        labels.append(batch_edge_labels)
        masks.append(batch_edge_masks)

    # aggregate
    preds = torch.cat(preds, dim=0)
    labels = torch.cat(labels, dim=0)
    masks = torch.cat(masks, dim=0)

    criterion = nn.CrossEntropyLoss(reduction="none")
    loss = criterion(preds, labels)
    loss = (loss * masks).sum() / (masks.sum() + 1e-8)

    preds = preds.argmax(-1)
    acc = (preds[masks==1] == labels[masks==1]).sum().float() / (masks.sum().float() + 1e-8)

    return loss, acc

def get_loss(data_dict, caption, orientation, num_bins, loss_opt):

    temp = data_dict["lang_feat"]

    if caption:
        _, data_dict = compute_cap_loss(data_dict, loss_opt)
    else:
        # store
        data_dict["cap_loss"] = torch.zeros(1)[0].type_as(temp)
        data_dict["cap_acc"] = torch.zeros(1)[0].type_as(temp)
        data_dict["pred_ious"] =  torch.zeros(1)[0].type_as(temp)

    if orientation:
        ori_loss, ori_acc = compute_node_orientation_loss(data_dict, num_bins)

        # store
        data_dict["ori_loss"] = ori_loss
        data_dict["ori_acc"] = ori_acc
    else:
        # store
        data_dict["ori_loss"] = torch.zeros(1)[0].type_as(temp)
        data_dict["ori_acc"] = torch.zeros(1)[0].type_as(temp)

    loss = data_dict["cap_loss"] + 0.1 * data_dict["ori_loss"]

    return loss, data_dict