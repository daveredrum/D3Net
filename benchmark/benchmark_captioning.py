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
from lib.captioning.eval_helper import eval_caption_step, eval_caption_epoch, decode_caption


def load_conf(args):
    base_cfg = OmegaConf.load("conf/path.yaml")
    cfg_path = os.path.join(base_cfg.OUTPUT_PATH, args.folder, "config.yaml")
    cfg = OmegaConf.load(cfg_path)
    cfg = OmegaConf.merge(base_cfg, cfg)
    
    root = os.path.join(cfg.general.output_root, args.folder)
    os.makedirs(root, exist_ok=True)

    # HACK manually setting those properties
    cfg.data.split = args.split
    cfg.general.task = "test"
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
    raw_test = json.load(open(cfg["{}_PATH".format(cfg.general.dataset.upper())].test_split))

    raw_train_scan_list = sorted(list(set([data["scene_id"] for data in raw_train])))
    raw_val_scan_list = sorted(list(set([data["scene_id"] for data in raw_val])))
    raw_test_scan_list = sorted(list(set([data["scene_id"] for data in raw_test])))

    mode = "speaker"

    new_test = []
    for scene_id in raw_test_scan_list:
        data = deepcopy(raw_test[0])
        data["scene_id"] = scene_id
        new_test.append(data)

    raw_test = new_test

    print("=> loading train split...")
    train_dataset = Dataset(cfg, cfg.general.dataset, mode, "train", raw_train, raw_train_scan_list, SCAN2CAD)

    print("=> loading val split...")
    val_dataset = Dataset(cfg, cfg.general.dataset, mode, "val", raw_val, raw_val_scan_list, SCAN2CAD)

    print("=> loading test split...")
    test_dataset = Dataset(cfg, cfg.general.dataset, mode, "test", raw_test, raw_test_scan_list, SCAN2CAD)

    print("=> loading complete")

    train_dataloader = DataLoader(train_dataset, batch_size=cfg.data.batch_size, \
        shuffle=True, pin_memory=True, num_workers=cfg.data.num_workers, collate_fn=collate_fn)
    val_dataloader = DataLoader(val_dataset, batch_size=cfg.data.batch_size, \
        shuffle=False, pin_memory=True, num_workers=cfg.data.num_workers, collate_fn=collate_fn)
    test_dataloader = DataLoader(test_dataset, batch_size=cfg.data.batch_size, \
        shuffle=False, pin_memory=True, num_workers=cfg.data.num_workers, collate_fn=collate_fn)

    dataset = {
        "train": train_dataset,
        "val": val_dataset,
        "test": test_dataset
    }

    dataloader = {
        "train": train_dataloader,
        "val": val_dataloader,
        "test": test_dataloader
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

def predict(cfg, dataset, dataloader, model):
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

    results = {}
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

            # for object detection
            pred_box_corners = data_dict["proposal_bbox_batched"] # B, num_proposal, 8, 3
            pred_sem_cls = data_dict['proposal_sem_cls_batched'] - 2 # B, num_proposal
            pred_sem_cls[pred_sem_cls < 0] = 17
            pred_sem_prob = torch.zeros(pred_sem_cls.shape[0], pred_sem_cls.shape[1], 18).type_as(pred_sem_cls)
            pred_sem_prob = torch.scatter(pred_sem_prob, 2, pred_sem_cls.unsqueeze(2).long(), 1) # B, num_proposal, 18
            pred_obj_prob = data_dict['proposal_scores_batched'] # B, num_proposal
            pred_obj_prob = torch.stack([torch.zeros_like(pred_obj_prob), pred_obj_prob], dim=2) # B, num_proposal, 2

            # nms mask
            _ = parse_predictions(data_dict, POST_DICT)
            nms_masks = torch.FloatTensor(data_dict["pred_mask"]).type_as(pred_box_corners).long()

            dataset_ids = data_dict["id"]
            captions = data_dict["lang_cap"]

            for batch_id in range(pred_box_corners.shape[0]):
                dataset_idx = dataset_ids[batch_id].item()
                scene_id = dataset.chunked_data[dataset_idx][0]["scene_id"]
                scene_outputs = []
                for prop_id in range(pred_box_corners.shape[1]):
                    if nms_masks[batch_id, prop_id] == 1:
                        caption = decode_caption(captions[batch_id][prop_id], dataset.vocabulary["idx2word"])

                        box = pred_box_corners[batch_id, prop_id].cpu().detach().numpy().tolist()
                        sem_prob = pred_sem_prob[batch_id, prop_id].cpu().detach().numpy().tolist()
                        obj_prob = pred_obj_prob[batch_id, prop_id].cpu().detach().numpy().tolist()

                        scene_outputs.append(
                            {
                                "caption": caption,
                                "box": box,
                                "sem_prob": sem_prob,
                                "obj_prob": obj_prob
                            }
                        )

                results[scene_id] = scene_outputs


    # dump
    save_path = os.path.join(cfg.general.root, "benchmark_{}.nms.json".format(cfg.data.split))
    with open(save_path, "w") as f:
        json.dump(results, f, indent=4)

    print("done!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-f", "--folder", type=str, required=True, help="path to folder with model")
    parser.add_argument("-s", "--split", type=str, default="test", help="specify data split")
    args = parser.parse_args()

    os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

    print("=> loading configurations...")
    cfg = load_conf(args)

    print("=> initializing data...")
    dataset, dataloader = init_data(cfg)

    print("=> initializing model...")
    model = init_model(cfg, dataset)
    
    print("=> start predicting...")
    predict(cfg, dataset[args.split], dataloader[args.split], model)
        

