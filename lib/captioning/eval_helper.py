import os
import sys
import json
from scipy.spatial.kdtree import distance_matrix
import torch
import pickle
import argparse

import numpy as np

from tqdm import tqdm
from copy import deepcopy
from torch.utils.data import DataLoader
from numpy.linalg import inv

sys.path.append(os.path.join(os.getcwd())) # HACK add the root folder

import lib.capeval.bleu.bleu as capblue
import lib.capeval.cider.cider as capcider
import lib.capeval.rouge.rouge as caprouge
import lib.capeval.meteor.meteor as capmeteor

from data.scannet.model_util_scannet import ScannetDatasetConfig
from lib.det.ap_helper import parse_predictions
from lib.utils.bbox import get_aabb3d_iou_batch


def get_organized(cfg, phase):
    dataset_name = cfg.general.dataset
    raw_data = json.load(open(cfg["{}_PATH".format(dataset_name.upper())]["{}_split".format(phase)]))

    organized = {}
    for data in raw_data:
        scene_id = data["scene_id"]
        object_id = data["object_id"]

        if scene_id not in organized: organized[scene_id] = {}
        if object_id not in organized[scene_id]: organized[scene_id][object_id] = []

        organized[scene_id][object_id].append(data)

    return organized

def prepare_corpus(raw_data, candidates, max_len=30):
    # get involved scene IDs
    scene_list = []
    for key in candidates.keys():
        scene_id, _, _ = key.split("|")
        if scene_id not in scene_list: scene_list.append(scene_id)

    corpus = {}
    for data in raw_data:
        scene_id = data["scene_id"]
        if scene_id not in scene_list: continue
        object_id = data["object_id"]
        object_name = data["object_name"]
        token = data["token"][:max_len]
        description = " ".join(token)

        # add start and end token
        description = "sos " + description
        description += " eos"

        key = "{}|{}|{}".format(scene_id, object_id, object_name)

        if key not in corpus:
            corpus[key] = []

        corpus[key].append(description)

    return corpus

def decode_caption(raw_caption, idx2word):
    decoded = ["sos"]
    for token_idx in raw_caption:
        token_idx = token_idx.item()
        token = idx2word[str(token_idx)]
        decoded.append(token)
        if token == "eos": break

    if "eos" not in decoded: decoded.append("eos")
    decoded = " ".join(decoded)

    return decoded

def filter_candidates(candidates, min_iou):
    new_candidates = {}
    for key, value in candidates.items():
        des, iou = value[0], value[1]
        if iou >= min_iou:
            new_candidates[key] = des

    return new_candidates

def check_candidates(corpus, candidates):
    placeholder = "sos eos"
    corpus_keys = list(corpus.keys())
    candidate_keys = list(candidates.keys())
    missing_keys = [key for key in corpus_keys if key not in candidate_keys]

    if len(missing_keys) != 0:
        for key in missing_keys:
            candidates[key] = [placeholder]

    return candidates

def organize_candidates(corpus, candidates):
    new_candidates = {}
    for key in corpus.keys():
        new_candidates[key] = candidates[key]

    return new_candidates

def eval_caption_step(cfg, data_dict, dataset_chunked_data, dataset_vocabulary, phase="val"):
    candidates = {}

    organized = get_organized(cfg, phase)

    # unpack
    captions = data_dict["lang_cap"] # [batch_size...[num_proposals...[num_words...]]]
    # NOTE the captions are stacked
    bbox_corners = data_dict["proposal_bbox_batched"]
    dataset_ids = data_dict["id"]
    batch_size, num_proposals, _, _ = bbox_corners.shape

    # post-process
    # config
    POST_DICT = {
        "remove_empty_box": False, 
        "use_3d_nms": True, 
        "nms_iou": 0.25,
        "use_old_type_nms": False, 
        "cls_nms": True, 
        "per_class_proposal": True,
        "conf_thresh": 0.09,
        "dataset_config": ScannetDatasetConfig(cfg)
    }

    if cfg.model.no_detection:
        nms_masks = data_dict["proposal_batch_mask"]

        detected_object_ids = data_dict["proposal_object_ids"]
        ious = torch.ones(batch_size, num_proposals).type_as(bbox_corners)
    else:
        # nms mask
        _ = parse_predictions(data_dict, POST_DICT)
        nms_masks = torch.FloatTensor(data_dict["pred_mask"]).type_as(bbox_corners).long()

        # objectness mask
        obj_masks = data_dict["proposal_batch_mask"].long()

        # # final mask
        nms_masks = nms_masks * obj_masks

        # pick out object ids of detected objects
        detected_object_ids = torch.gather(data_dict["scene_object_ids"], 1, data_dict["object_assignment"])

        # bbox corners
        assigned_target_bbox_corners = torch.gather(
            data_dict["gt_bbox"], 
            1, 
            data_dict["object_assignment"].view(batch_size, num_proposals, 1, 1).repeat(1, 1, 8, 3)
        ) # batch_size, num_proposals, 8, 3
        detected_bbox_corners = data_dict["proposal_bbox_batched"] # batch_size, num_proposals, 8, 3
        
        # compute IoU between each detected box and each ground truth box
        ious = get_aabb3d_iou_batch(
            assigned_target_bbox_corners.view(-1, 8, 3).detach().cpu().numpy(), # batch_size * num_proposals, 8, 3
            detected_bbox_corners.view(-1, 8, 3).detach().cpu().numpy() # batch_size * num_proposals, 8, 3
        )
        ious = torch.from_numpy(ious).type_as(bbox_corners).view(batch_size, num_proposals)

        # change shape
        assigned_target_bbox_corners = assigned_target_bbox_corners.view(-1, num_proposals, 8, 3) # batch_size, num_proposals, 8, 3
        detected_bbox_corners = detected_bbox_corners.view(-1, num_proposals, 8, 3) # batch_size, num_proposals, 8, 3

    # dump generated captions
    for batch_id in range(batch_size):
        dataset_idx = dataset_ids[batch_id].item()
        scene_id = dataset_chunked_data[dataset_idx][0]["scene_id"]
        for prop_id in range(num_proposals):
            if nms_masks[batch_id, prop_id] == 1:
                object_id = str(detected_object_ids[batch_id, prop_id].item())
                caption_decoded = decode_caption(captions[batch_id][prop_id], dataset_vocabulary["idx2word"])

                entry = [
                    [caption_decoded],
                    ious[batch_id, prop_id].item(),
                    detected_bbox_corners[batch_id, prop_id].detach().cpu().numpy().tolist(),
                    assigned_target_bbox_corners[batch_id, prop_id].detach().cpu().numpy().tolist()
                ]

                try:
                    object_name = organized[scene_id][object_id][0]["object_name"]

                    # store
                    key = "{}|{}|{}".format(scene_id, object_id, object_name)

                    if key not in candidates:
                        candidates[key] = entry
                    else:
                        # update the stored prediction if IoU is higher
                        if ious[batch_id, prop_id].item() > candidates[key][1]:
                            candidates[key] = entry

                except KeyError:
                    continue

    return candidates

def eval_caption_epoch(candidates, cfg, device, phase="val", force=False, max_len=30, min_iou=0.5):
    experiment_root = os.path.join(cfg.OUTPUT_PATH, cfg.general.experiment.upper())
    
    # corpus
    corpus_path = os.path.join(experiment_root, "corpus_{}_{}.json".format(phase, str(device.index)))
    
    dataset_name = cfg.general.dataset
    raw_data = json.load(open(cfg["{}_PATH".format(dataset_name.upper())]["{}_split".format(phase)]))

    corpus = prepare_corpus(raw_data, candidates, max_len)
    
    if not os.path.exists(corpus_path) or force:
        with open(corpus_path, "w") as f:
            json.dump(corpus, f, indent=4)

    pred_path = os.path.join(experiment_root, "pred_{}_{}.json".format(phase, str(device.index)))
    with open(pred_path, "w") as f:
        json.dump(candidates, f, indent=4)

    # check candidates
    # NOTE: make up the captions for the undetected object by "sos eos"
    candidates = filter_candidates(candidates, min_iou)
    candidates = check_candidates(corpus, candidates)
    candidates = organize_candidates(corpus, candidates)

    # candidates for evaluation -> debug
    temp_path = os.path.join(experiment_root, "pred_processed_{}_{}_{}.json".format(phase, str(device.index), str(min_iou)))
    with open(temp_path, "w") as f:
        json.dump(candidates, f, indent=4)

    # compute scores
    # print("computing scores...")
    bleu = capblue.Bleu(4).compute_score(corpus, candidates)
    cider = capcider.Cider().compute_score(corpus, candidates)
    rouge = caprouge.Rouge().compute_score(corpus, candidates)
    meteor = capmeteor.Meteor().compute_score(corpus, candidates)

    # save results
    result_path = os.path.join(experiment_root, "eval_{}_{}.txt".format(phase, str(device.index)))
    with open(result_path, "w") as f:
        f.write("----------------------Evaluation-----------------------\n")
        f.write("[BLEU-1] Mean: {:.4f}, Max: {:.4f}, Min: {:.4f}\n".format(bleu[0][0], max(bleu[1][0]), min(bleu[1][0])))
        f.write("[BLEU-2] Mean: {:.4f}, Max: {:.4f}, Min: {:.4f}\n".format(bleu[0][1], max(bleu[1][1]), min(bleu[1][1])))
        f.write("[BLEU-3] Mean: {:.4f}, Max: {:.4f}, Min: {:.4f}\n".format(bleu[0][2], max(bleu[1][2]), min(bleu[1][2])))
        f.write("[BLEU-4] Mean: {:.4f}, Max: {:.4f}, Min: {:.4f}\n".format(bleu[0][3], max(bleu[1][3]), min(bleu[1][3])))
        f.write("[CIDEr] Mean: {:.4f}, Max: {:.4f}, Min: {:.4f}\n".format(cider[0], max(cider[1]), min(cider[1])))
        f.write("[ROUGE-L] Mean: {:.4f}, Max: {:.4f}, Min: {:.4f}\n".format(rouge[0], max(rouge[1]), min(rouge[1])))
        f.write("[METEOR] Mean: {:.4f}, Max: {:.4f}, Min: {:.4f}\n".format(meteor[0], max(meteor[1]), min(meteor[1])))

    return bleu, cider, rouge, meteor

