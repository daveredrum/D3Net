import warnings
warnings.filterwarnings("ignore")


import os
import sys
import json
import torch
import pickle
import argparse

import numpy as np

from copy import deepcopy
from omegaconf import OmegaConf
from importlib import import_module
from tqdm import tqdm

from torch.utils.data.dataloader import DataLoader

sys.path.append(".")

from data.scannet.model_util_scannet import ScannetDatasetConfig

from lib.det.ap_helper import APCalculator, parse_predictions, parse_groundtruths
from lib.grounding.loss_helper import get_loss
from lib.grounding.eval_helper import get_eval
from lib.captioning.eval_helper import eval_caption_step, eval_caption_epoch

def load_conf(args):
    base_cfg = OmegaConf.load("conf/path.yaml")
    cfg_path = os.path.join(base_cfg.OUTPUT_PATH, args.folder, "config.yaml")
    cfg = OmegaConf.load(cfg_path)
    cfg = OmegaConf.merge(base_cfg, cfg)
    
    root = os.path.join(cfg.general.output_root, args.folder)
    os.makedirs(root, exist_ok=True)

    # HACK manually setting those properties
    cfg.data.split = args.split
    cfg.general.task = args.task
    cfg.general.root = root
    cfg.data.num_des_per_scene = 8 # NOTE set to 1 for accurate evaluation - but it could take a long time
    cfg.cluster.prepare_epochs = -1

    return cfg

def init_data(cfg):
    DATA_MODULE = import_module(cfg.data.module)
    Dataset = getattr(DATA_MODULE, cfg.data.dataset)
    collate_fn = getattr(DATA_MODULE, "sparse_collate_fn")

    SCAN2CAD = json.load(open(cfg.SCAN2CAD))

    raw_train = json.load(open(cfg["{}_PATH".format(cfg.general.dataset.upper())].train_split))
    raw_val = json.load(open(cfg["{}_PATH".format(cfg.general.dataset.upper())].val_split))

    raw_train_scan_list = sorted(list(set([data["scene_id"] for data in raw_train])))
    raw_val_scan_list = sorted(list(set([data["scene_id"] for data in raw_val])))
    det_val_scan_list = sorted([line.rstrip() for line in open(cfg.SCANNETV2_PATH.val_list)])

    det_val = []
    for scene_id in det_val_scan_list:
        data = deepcopy(raw_val[0])
        data["scene_id"] = scene_id
        det_val.append(data)

    if cfg.general.task == "captioning":
        mode = "speaker"

        new_val = []
        for scene_id in raw_val_scan_list:
            data = deepcopy(raw_val[0])
            data["scene_id"] = scene_id
            new_val.append(data)

        raw_val = new_val
    else:
        mode = "listener"

    print("=> loading train split...")
    cap_train_dataset = Dataset(cfg, cfg.general.dataset, mode, "train", raw_train, raw_train_scan_list, SCAN2CAD)

    print("=> loading val split...")
    cap_val_dataset = Dataset(cfg, cfg.general.dataset, mode, "val", raw_val, raw_val_scan_list, SCAN2CAD)
    
    print("=> loading val split for detection...")
    det_val_dataset = Dataset(cfg, cfg.general.dataset, mode, "val", det_val, det_val_scan_list)

    print("=> loading complete")

    train_dataloader = DataLoader(cap_train_dataset, batch_size=cfg.data.batch_size, \
        shuffle=True, pin_memory=True, num_workers=cfg.data.num_workers, collate_fn=collate_fn)
    val_dataloader = DataLoader(cap_val_dataset, batch_size=cfg.data.batch_size, \
        shuffle=False, pin_memory=True, num_workers=cfg.data.num_workers, collate_fn=collate_fn)
    det_dataloader = DataLoader(det_val_dataset, batch_size=cfg.data.batch_size, \
        shuffle=False, pin_memory=True, num_workers=cfg.data.num_workers, collate_fn=collate_fn)

    dataset = {
        "train": cap_train_dataset,
        "val": cap_val_dataset,
        "det": det_val_dataset
    }

    dataloader = {
        "train": train_dataloader,
        "val": val_dataloader,
        "det": det_dataloader
    }

    return dataset, dataloader

def init_model(cfg, dataset):
    PipelineNet = getattr(import_module("model.pipeline"), "PipelineNet")
    model = PipelineNet(cfg, dataset)

    checkpoint_name = "model.ckpt"
    checkpoint_path = os.path.join(cfg.general.root, checkpoint_name)

    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint["state_dict"], strict=False)

    model.cuda()
    model.eval()

    return model

def eval_detection(cfg, dataloader, model):
    DC = ScannetDatasetConfig(cfg)

    # config
    POST_DICT = {
        "remove_empty_box": False, 
        "use_3d_nms": True, 
        "nms_iou": 0.25,
        "use_old_type_nms": False, 
        "cls_nms": True, 
        "per_class_proposal": True,
        "conf_thresh": 0.09,
        "dataset_config": DC
    }
    AP_IOU_THRESHOLDS = [0.25, 0.5]
    AP_CALCULATOR_LIST = [APCalculator(iou_thresh, DC.class2type) for iou_thresh in AP_IOU_THRESHOLDS]

    with torch.no_grad():
        for data_dict in tqdm(dataloader):
            for key in data_dict.keys():
                data_dict[key] = data_dict[key].cuda()

            torch.cuda.empty_cache()

            with torch.no_grad():
                data_dict = model.detector.feed(data_dict, epoch=1)
            
            batch_pred_map_cls = parse_predictions(data_dict, POST_DICT) 
            batch_gt_map_cls = parse_groundtruths(data_dict, POST_DICT) 
            for ap_calculator in AP_CALCULATOR_LIST:
                ap_calculator.step(batch_pred_map_cls, batch_gt_map_cls)
            
    for i, ap_calculator in enumerate(AP_CALCULATOR_LIST):
        print()
        print("-"*10, "iou_thresh: %f"%(AP_IOU_THRESHOLDS[i]), "-"*10)
        metrics_dict = ap_calculator.compute_metrics()
        for key in metrics_dict:
            print("eval %s: %f"%(key, metrics_dict[key]))

def eval_grounding(cfg, dataset, dataloader, model):
    # random seeds
    seeds = [cfg.general.manual_seed] + [2 * i for i in range(cfg.eval.repeat - 1)]

    # evaluate
    print("evaluating...")
    score_path = os.path.join(cfg.general.root, "scores.p")
    pred_path = os.path.join(cfg.general.root, "predictions.p")
    gen_flag = (not os.path.exists(score_path)) or cfg.eval.repeat > 1 or cfg.eval.force
    if gen_flag:
        ref_acc_all = []
        ious_all = []
        masks_all = []
        others_all = []
        lang_acc_all = []
        for seed in seeds:
            # reproducibility
            torch.manual_seed(seed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
            np.random.seed(seed)

            print("generating the scores for seed {}...".format(seed))
            ref_acc = []
            ious = []
            masks = []
            others = []
            lang_acc = []
            predictions = {}
            for data_dict in tqdm(dataloader):
                for key in data_dict:
                    data_dict[key] = data_dict[key].cuda()

                torch.cuda.empty_cache()

                # feed
                data_dict = model.detector.feed(data_dict)
                _, data_dict = model.detector.parse_feed_ret(data_dict)
                data_dict = model.detector.loss(data_dict, epoch=1)

                data_dict = model.listener(data_dict)

                _, data_dict = get_loss(
                    data_dict,
                    use_oracle=model.no_detection,
                    grounding=not model.no_grounding,
                    use_lang_classifier=model.use_lang_classifier,
                    use_rl=False
                )
                data_dict = get_eval(
                    data_dict,
                    grounding=not model.no_grounding,
                    use_lang_classifier=model.use_lang_classifier
                )

                ref_acc += data_dict["ref_acc"]
                ious += data_dict["ref_iou"].cpu().numpy().tolist()
                masks += data_dict["ref_multiple_mask"]
                others += data_dict["ref_others_mask"]
                lang_acc.append(data_dict["lang_acc"].item())

                # store predictions
                batch_ids = data_dict["id"]
                chunk_ids = data_dict["chunk_ids"]
                _, chunk_size = chunk_ids.shape
                batch_ids = batch_ids.unsqueeze(1).repeat(1, chunk_size).reshape(-1)
                chunk_ids = chunk_ids.reshape(-1)
                batch_size = batch_ids.shape[0]
                # print(batch_ids.shape[0], data_dict["pred_bboxes"].shape[0])
                # exit()
                assert batch_size == data_dict["pred_bboxes"].shape[0]

                batch_ids = batch_ids.long().detach().cpu().numpy()
                chunk_ids = chunk_ids.long().detach().cpu().numpy()
                for i in range(batch_size):
                    batch_idx = batch_ids[i]
                    chunk_idx = chunk_ids[i]

                    scene_id = dataset.chunked_data[batch_idx][chunk_idx]["scene_id"]
                    object_id = dataset.chunked_data[batch_idx][chunk_idx]["object_id"]
                    ann_id = dataset.chunked_data[batch_idx][chunk_idx]["ann_id"]

                    if scene_id not in predictions:
                        predictions[scene_id] = {}

                    if object_id not in predictions[scene_id]:
                        predictions[scene_id][object_id] = {}

                    if ann_id not in predictions[scene_id][object_id]:
                        predictions[scene_id][object_id][ann_id] = {}

                    predictions[scene_id][object_id][ann_id]["pred_bbox"] = data_dict["pred_bboxes"][i].detach().cpu().numpy()
                    predictions[scene_id][object_id][ann_id]["gt_bbox"] = data_dict["gt_bboxes"][i].detach().cpu().numpy()
                    predictions[scene_id][object_id][ann_id]["iou"] = data_dict["ref_iou"][i].detach().cpu().numpy()

            # save the last predictions
            with open(pred_path, "wb") as f:
                pickle.dump(predictions, f)

            # save to global
            ref_acc_all.append(ref_acc)
            ious_all.append(ious)
            masks_all.append(masks)
            others_all.append(others)
            lang_acc_all.append(lang_acc)

        # convert to numpy array
        ref_acc = np.array(ref_acc_all)
        ious = np.array(ious_all)
        masks = np.array(masks_all)
        others = np.array(others_all)
        lang_acc = np.array(lang_acc_all)

        # save the global scores
        with open(score_path, "wb") as f:
            scores = {
                "ref_acc": ref_acc_all,
                "ious": ious_all,
                "masks": masks_all,
                "others": others_all,
                "lang_acc": lang_acc_all
            }
            pickle.dump(scores, f)

    else:
        print("loading the scores...")
        with open(score_path, "rb") as f:
            scores = pickle.load(f)

            # unpack
            ref_acc = np.array(scores["ref_acc"])
            ious = np.array(scores["ious"])
            masks = np.array(scores["masks"])
            others = np.array(scores["others"])
            lang_acc = np.array(scores["lang_acc"])

    multiple_dict = {
        "unique": 0,
        "multiple": 1
    }
    others_dict = {
        "not_in_others": 0,
        "in_others": 1
    }

    # evaluation stats
    stats = {k: np.sum(masks[0] == v) for k, v in multiple_dict.items()}
    stats["overall"] = masks[0].shape[0]
    stats = {}
    for k, v in multiple_dict.items():
        stats[k] = {}
        for k_o, v_o in others_dict.items():
            stats[k][k_o] = np.sum(np.logical_and(masks[0] == v, others[0] == v_o))

        stats[k]["overall"] = np.sum(masks[0] == v)

    stats["overall"] = {}
    for k_o, v_o in others_dict.items():
        stats["overall"][k_o] = np.sum(others[0] == v_o)
    
    stats["overall"]["overall"] = masks[0].shape[0]

    # aggregate scores
    scores = {}
    for k, v in multiple_dict.items():
        for k_o in others_dict.keys():
            ref_accs, acc_025ious, acc_05ious = [], [], []
            for i in range(masks.shape[0]):
                running_ref_acc = np.mean(ref_acc[i][np.logical_and(masks[i] == multiple_dict[k], others[i] == others_dict[k_o])]) \
                    if np.sum(np.logical_and(masks[i] == multiple_dict[k], others[i] == others_dict[k_o])) > 0 else 0
                running_acc_025iou = ious[i][np.logical_and(np.logical_and(masks[i] == multiple_dict[k], others[i] == others_dict[k_o]), ious[i] >= 0.25)].shape[0] \
                    / ious[i][np.logical_and(masks[i] == multiple_dict[k], others[i] == others_dict[k_o])].shape[0] \
                    if np.sum(np.logical_and(masks[i] == multiple_dict[k], others[i] == others_dict[k_o])) > 0 else 0
                running_acc_05iou = ious[i][np.logical_and(np.logical_and(masks[i] == multiple_dict[k], others[i] == others_dict[k_o]), ious[i] >= 0.5)].shape[0] \
                    / ious[i][np.logical_and(masks[i] == multiple_dict[k], others[i] == others_dict[k_o])].shape[0] \
                    if np.sum(np.logical_and(masks[i] == multiple_dict[k], others[i] == others_dict[k_o])) > 0 else 0

                # store
                ref_accs.append(running_ref_acc)
                acc_025ious.append(running_acc_025iou)
                acc_05ious.append(running_acc_05iou)

            if k not in scores:
                scores[k] = {k_o: {} for k_o in others_dict.keys()}

            scores[k][k_o]["ref_acc"] = np.mean(ref_accs)
            scores[k][k_o]["acc@0.25iou"] = np.mean(acc_025ious)
            scores[k][k_o]["acc@0.5iou"] = np.mean(acc_05ious)

        ref_accs, acc_025ious, acc_05ious = [], [], []
        for i in range(masks.shape[0]):
            running_ref_acc = np.mean(ref_acc[i][masks[i] == multiple_dict[k]]) if np.sum(masks[i] == multiple_dict[k]) > 0 else 0
            running_acc_025iou = ious[i][np.logical_and(masks[i] == multiple_dict[k], ious[i] >= 0.25)].shape[0] \
                / ious[i][masks[i] == multiple_dict[k]].shape[0] if np.sum(masks[i] == multiple_dict[k]) > 0 else 0
            running_acc_05iou = ious[i][np.logical_and(masks[i] == multiple_dict[k], ious[i] >= 0.5)].shape[0] \
                / ious[i][masks[i] == multiple_dict[k]].shape[0] if np.sum(masks[i] == multiple_dict[k]) > 0 else 0

            # store
            ref_accs.append(running_ref_acc)
            acc_025ious.append(running_acc_025iou)
            acc_05ious.append(running_acc_05iou)

        scores[k]["overall"] = {}
        scores[k]["overall"]["ref_acc"] = np.mean(ref_accs)
        scores[k]["overall"]["acc@0.25iou"] = np.mean(acc_025ious)
        scores[k]["overall"]["acc@0.5iou"] = np.mean(acc_05ious)

    scores["overall"] = {}
    for k_o in others_dict.keys():
        ref_accs, acc_025ious, acc_05ious = [], [], []
        for i in range(masks.shape[0]):
            running_ref_acc = np.mean(ref_acc[i][others[i] == others_dict[k_o]]) if np.sum(others[i] == others_dict[k_o]) > 0 else 0
            running_acc_025iou = ious[i][np.logical_and(others[i] == others_dict[k_o], ious[i] >= 0.25)].shape[0] \
                / ious[i][others[i] == others_dict[k_o]].shape[0] if np.sum(others[i] == others_dict[k_o]) > 0 else 0
            running_acc_05iou = ious[i][np.logical_and(others[i] == others_dict[k_o], ious[i] >= 0.5)].shape[0] \
                / ious[i][others[i] == others_dict[k_o]].shape[0] if np.sum(others[i] == others_dict[k_o]) > 0 else 0

            # store
            ref_accs.append(running_ref_acc)
            acc_025ious.append(running_acc_025iou)
            acc_05ious.append(running_acc_05iou)

        # aggregate
        scores["overall"][k_o] = {}
        scores["overall"][k_o]["ref_acc"] = np.mean(ref_accs)
        scores["overall"][k_o]["acc@0.25iou"] = np.mean(acc_025ious)
        scores["overall"][k_o]["acc@0.5iou"] = np.mean(acc_05ious)
   
    ref_accs, acc_025ious, acc_05ious = [], [], []
    for i in range(masks.shape[0]):
        running_ref_acc = np.mean(ref_acc[i])
        running_acc_025iou = ious[i][ious[i] >= 0.25].shape[0] / ious[i].shape[0]
        running_acc_05iou = ious[i][ious[i] >= 0.5].shape[0] / ious[i].shape[0]

        # store
        ref_accs.append(running_ref_acc)
        acc_025ious.append(running_acc_025iou)
        acc_05ious.append(running_acc_05iou)

    # aggregate
    scores["overall"]["overall"] = {}
    scores["overall"]["overall"]["ref_acc"] = np.mean(ref_accs)
    scores["overall"]["overall"]["acc@0.25iou"] = np.mean(acc_025ious)
    scores["overall"]["overall"]["acc@0.5iou"] = np.mean(acc_05ious)

    # report
    print("\nstats:")
    for k_s in stats.keys():
        for k_o in stats[k_s].keys():
            print("{} | {}: {}".format(k_s, k_o, stats[k_s][k_o]))

    for k_s in scores.keys():
        print("\n{}:".format(k_s))
        for k_m in scores[k_s].keys():
            for metric in scores[k_s][k_m].keys():
                print("{} | {} | {}: {}".format(k_s, k_m, metric, scores[k_s][k_m][metric]))

    print("\nlanguage classification accuracy: {}".format(np.mean(lang_acc)))

def eval_captioning(cfg, dataset, dataloader, model):
    outputs = []
    with torch.no_grad():
        for data_dict in tqdm(dataloader):
            for key in data_dict.keys():
                if isinstance(data_dict[key][0], tuple): continue
                if isinstance(data_dict[key][0], dict): continue
                if isinstance(data_dict[key][0], list): continue

                data_dict[key] = data_dict[key].cuda()

            torch.cuda.empty_cache()

            ##### prepare input and forward
            data_dict = model.detector.feed(data_dict, 1)
            data_dict = model.speaker(data_dict, use_tf=False, is_eval=True, beam_opt=model.beam_opt)

            outs = eval_caption_step(
                cfg=cfg,
                data_dict=data_dict,
                dataset_chunked_data=dataset.chunked_data,
                dataset_vocabulary=dataset.vocabulary
            )
            outputs.append(outs)

    # aggregate captioning outputs
    candidates = {}
    for outs in outputs:
        for key, value in outs.items():
            if key not in candidates:
                candidates[key] = value

    # evaluate captions
    print("==> computing scores...")
    bleu, cider, rouge, meteor = eval_caption_epoch(
        candidates=candidates,
        cfg=cfg,
        device=model.device,
        phase="val",
        max_len=cfg.eval.max_des_len,
        min_iou=cfg.eval.min_iou_threshold
    )

    # report
    print("\n----------------------Evaluation-----------------------")
    print("[BLEU-1] Mean: {:.4f}, Max: {:.4f}, Min: {:.4f}".format(bleu[0][0], max(bleu[1][0]), min(bleu[1][0])))
    print("[BLEU-2] Mean: {:.4f}, Max: {:.4f}, Min: {:.4f}".format(bleu[0][1], max(bleu[1][1]), min(bleu[1][1])))
    print("[BLEU-3] Mean: {:.4f}, Max: {:.4f}, Min: {:.4f}".format(bleu[0][2], max(bleu[1][2]), min(bleu[1][2])))
    print("[BLEU-4] Mean: {:.4f}, Max: {:.4f}, Min: {:.4f}".format(bleu[0][3], max(bleu[1][3]), min(bleu[1][3])))
    print("[CIDEr] Mean: {:.4f}, Max: {:.4f}, Min: {:.4f}".format(cider[0], max(cider[1]), min(cider[1])))
    print("[ROUGE-L] Mean: {:.4f}, Max: {:.4f}, Min: {:.4f}".format(rouge[0], max(rouge[1]), min(rouge[1])))
    print("[METEOR] Mean: {:.4f}, Max: {:.4f}, Min: {:.4f}".format(meteor[0], max(meteor[1]), min(meteor[1])))
    print()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-f", "--folder", type=str, required=True, help="path to folder with model")
    # parser.add_argument("-c", "--config", type=str, default="conf/pointgroup_grounding.yaml", help="path to config file")
    parser.add_argument("-s", "--split", type=str, default="val", help="specify data split")
    parser.add_argument("-t", "--task", type=str, choices=["detection", "grounding", "captioning"], \
        help="specify task: detection | grounding | captioning", required=True)
    args = parser.parse_args()

    os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

    print("=> loading configurations...")
    cfg = load_conf(args)

    print("=> initializing data...")
    dataset, dataloader = init_data(cfg)

    print("=> initializing model...")
    model = init_model(cfg, dataset)
    
    print("=> start evaluating {}...".format(args.task))
    if args.task == "detection":
        eval_detection(cfg, dataloader["det"], model)
    elif args.task == "grounding":
        eval_grounding(cfg, dataset["val"], dataloader["val"], model)
    elif args.task == "captioning":
        eval_captioning(cfg, dataset["val"], dataloader["val"], model)
        

