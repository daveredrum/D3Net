import os
import sys
import json
import torch

from scipy.optimize import linear_sum_assignment

sys.path.append(os.path.join(os.getcwd())) # HACK add the root folder

import lib.capeval.bleu.bleu as capblue
import lib.capeval.cider.cider as capcider
import lib.capeval.rouge.rouge as caprouge
import lib.capeval.meteor.meteor as capmeteor

from lib.utils.bbox import box3d_iou_batch_tensor, generalized_box3d_iou


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
        scene_id, _ = key.split("|")
        if scene_id not in scene_list: scene_list.append(scene_id)

    corpus = {}
    for data in raw_data:
        scene_id = data["scene_id"]
        if scene_id not in scene_list: continue
        object_id = data["object_id"]
        token = data["token"][:max_len]
        description = " ".join(token)

        # add start and end token
        description = "sos " + description
        description += " eos"

        key = "{}|{}".format(scene_id, object_id)

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
        if value["iou"] >= min_iou:
            new_candidates[key] = [value["caption"]]

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

def assign_dense_caption(pred_captions, pred_boxes, gt_boxes, gt_box_ids, gt_box_masks, gt_scene_list, idx2word, special_tokens, strategy="giou"):
    """assign the densely predicted captions to GT boxes

    Args:
        pred_captions (torch.Tensor): predicted captions for all boxes, shape: (B, K1, L)
        pred_boxes (torch.Tensor): predicted bounding boxes, shape: (B, K1, 8, 3)
        gt_boxes (torch.Tensor): GT bounding boxes, shape: (B, K2)
        gt_box_ids (torch.Tensor): GT bounding boxes object IDs, shape: (B, K2)
        gt_box_masks (torch.Tensor): GT bounding boxes masks in the batch, shape: (B, K2)
        gt_scene_list (list): scene list in the batch, length: B
        idx2word (dict): vocabulary dictionary of all words, idx -> str
        special_tokens (dict): vocabulary dictionary of special tokens, e.g. [SOS], [PAD], etc.
        strategy ("giou" or "center"): assignment strategy, default: "giou"

    Returns:
        Dict: dictionary of assigned boxes and captions
    """

    def box_assignment(pred_boxes, gt_boxes, gt_masks):
        """assign GT boxes to predicted boxes

        Args:
            pred_boxes (torch.Tensor): predicted boxes, shape: (B, K1, 8, 3)
            gt_boxes (torch.Tensor): GT boxes, shape: (B, K2, 8, 3)
        """

        batch_size, nprop, *_ = pred_boxes.shape
        _, ngt, *_ = gt_boxes.shape
        nactual_gt = gt_masks.sum(1).long()

        # assignment
        if strategy == "giou":
            # gious
            gious = generalized_box3d_iou(
                pred_boxes,
                gt_boxes,
                nactual_gt,
                rotated_boxes=False,
                needs_grad=False,
            ) # B, K1, K2

            # hungarian assignment
            final_cost = -gious.detach().cpu().numpy()
        elif strategy == "center":
            # center distance
            dist = torch.cdist(pred_boxes.mean(2).float(), gt_boxes.mean(2).float())

            # hungarian assignment
            final_cost = dist.detach().cpu().numpy()
        else:
            raise ValueError("invalid strategy.")

        assignments = []

        # assignments from GTs to proposals
        per_gt_prop_inds = torch.zeros(
            [batch_size, ngt], dtype=torch.int64, device=pred_boxes.device
        )
        gt_matched_mask = torch.zeros(
            [batch_size, ngt], dtype=torch.float32, device=pred_boxes.device
        )

        for b in range(batch_size):
            assign = []
            if nactual_gt[b] > 0:
                assign = linear_sum_assignment(final_cost[b, :, : nactual_gt[b]])
                assign = [
                    torch.from_numpy(x).long().to(device=pred_boxes.device)
                    for x in assign
                ]

                per_gt_prop_inds[b, assign[1]] = assign[0]
                gt_matched_mask[b, assign[1]] = 1

            assignments.append(assign)

        return {
            "assignments": assignments,
            "per_gt_prop_inds": per_gt_prop_inds,
            "gt_matched_mask": gt_matched_mask
        }

    def decode_caption(raw_caption, idx2word, special_tokens):
        decoded = [special_tokens["bos_token"]]
        for token_idx in raw_caption:
            token_idx = token_idx.item()
            token = idx2word[str(token_idx)]
            decoded.append(token)
            if token == special_tokens["eos_token"]: break

        if special_tokens["eos_token"] not in decoded: decoded.append(special_tokens["eos_token"])
        decoded = " ".join(decoded)

        return decoded
    
    candidates = {}

    # assign GTs to predicted boxes
    assignments = box_assignment(pred_boxes, gt_boxes, gt_box_masks)

    batch_size, num_gts = gt_box_ids.shape
    per_gt_prop_inds = assignments["per_gt_prop_inds"]
    matched_prop_box_corners = torch.gather(
        pred_boxes, 1, per_gt_prop_inds[:, :, None, None].repeat(1, 1, 8, 3)
    ) # batch_size, num_gts, 8, 3 
    matched_ious = box3d_iou_batch_tensor(
        matched_prop_box_corners.reshape(-1, 8, 3), 
        gt_boxes.reshape(-1, 8, 3)
    ).reshape(batch_size, num_gts)

    candidates = {}
    for batch_id in range(batch_size):
        scene_id = gt_scene_list[batch_id]
        for gt_id in range(num_gts):
            if gt_box_masks[batch_id, gt_id] == 0: continue

            object_id = str(gt_box_ids[batch_id, gt_id].item())
            caption_decoded = decode_caption(pred_captions[batch_id, per_gt_prop_inds[batch_id, gt_id]], idx2word, special_tokens)
            iou = matched_ious[batch_id, gt_id].item()
            box = matched_prop_box_corners[batch_id, gt_id].detach().cpu().numpy().tolist()
            gt_box = gt_boxes[batch_id, gt_id].detach().cpu().numpy().tolist()

            # store
            key = "{}|{}".format(scene_id, object_id)
            entry = {
                "caption": caption_decoded,
                "iou": iou,
                "box": box,
                "gt_box": gt_box
            }

            if key not in candidates:
                candidates[key] = entry
            else:
                # update the stored prediction if IoU is higher
                if iou > candidates[key][1]:
                    candidates[key] = entry

    return candidates

@torch.no_grad()
def eval_caption_step(cfg, data_dict, dataset_chunked_data, dataset_vocabulary, phase="val"):
    
    candidates = assign_dense_caption(
        pred_captions=data_dict["lang_cap"], # batch_size, num_proposals, num_words - 1/max_len
        pred_boxes=data_dict["proposal_bbox_batched"], 
        gt_boxes=data_dict["gt_bbox"], 
        gt_box_ids=data_dict["gt_bbox_object_id"], 
        gt_box_masks=data_dict["gt_bbox_label"], 
        gt_scene_list=data_dict["scene_id"],
        idx2word=dataset_vocabulary["idx2word"],
        special_tokens=dataset_vocabulary["special_tokens"]
    )

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

