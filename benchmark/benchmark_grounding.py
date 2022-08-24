import warnings
warnings.filterwarnings("ignore")


import os
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
    cfg.data.split = "test"
    cfg.general.task = "test"
    cfg.general.root = root
    cfg.data.num_des_per_scene = 1
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

    mode = "speaker" if cfg.general.task == "captioning" else "listener"

    print("=> loading train split...")
    train_dataset = Dataset(cfg, cfg.general.dataset, mode, "train", raw_train, raw_train_scan_list, SCAN2CAD)

    print("=> loading val split...")
    val_dataset = Dataset(cfg, cfg.general.dataset, mode, "val", raw_val, raw_val_scan_list, SCAN2CAD)

    print("=> loading test split...")
    test_dataset = Dataset(cfg, cfg.general.dataset, mode, "test", raw_test, raw_test_scan_list, None)

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
    # model.load_from_checkpoint(checkpoint_path)

    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint["state_dict"], strict=False)

    model.cuda()
    model.eval()

    return model

def predict(args, cfg, dataset, dataloader, model):
    predictions = []
    for data_dict in tqdm(dataloader):
        for key in data_dict:
            data_dict[key] = data_dict[key].cuda()

        torch.cuda.empty_cache()

        # feed
        data_dict = model.detector.feed(data_dict)
        data_dict = model.listener(data_dict)

        # predicted bbox
        pred_bbox_corners = data_dict["proposal_bbox_batched"] # (B, num_proposal, 8, 3)

        # store predictions
        batch_ids = data_dict["id"]
        chunk_ids = data_dict["chunk_ids"]
        cluster_preds = data_dict["cluster_ref"] # (B*C, num_proposal)
        unique_multiple = data_dict["unique_multiple"]
        object_cat = data_dict["object_cat"]
        _, chunk_size = chunk_ids.shape
        _, num_proposals = cluster_preds.shape
        cluster_preds = cluster_preds.argmax(-1) # (B,)
        batch_ids = batch_ids.unsqueeze(1).repeat(1, chunk_size).reshape(-1)
        chunk_ids = chunk_ids.reshape(-1)
        unique_multiple = unique_multiple.reshape(-1)
        object_cat = object_cat.reshape(-1)

        batch_size = batch_ids.shape[0]
        pred_bbox_corners = pred_bbox_corners.unsqueeze(1).repeat(1, chunk_size, 1, 1, 1).reshape(batch_size, num_proposals, 8, 3)
        assert batch_size == pred_bbox_corners.shape[0]

        batch_ids = batch_ids.long().detach().cpu().numpy()
        chunk_ids = chunk_ids.long().detach().cpu().numpy()
        cluster_preds = cluster_preds.detach().cpu().numpy()
        pred_bbox_corners = pred_bbox_corners.detach().cpu().numpy()
        unique_multiple = unique_multiple.detach().cpu().numpy()
        object_cat = object_cat.detach().cpu().numpy()

        cached = []
        for i in range(batch_size):
            batch_idx = batch_ids[i]
            chunk_idx = chunk_ids[i]

            scene_id = dataset.chunked_data[batch_idx][chunk_idx]["scene_id"]
            object_id = dataset.chunked_data[batch_idx][chunk_idx]["object_id"]
            ann_id = dataset.chunked_data[batch_idx][chunk_idx]["ann_id"]

            pred_ref_idx = cluster_preds[i]
            pred_bbox = pred_bbox_corners[i, pred_ref_idx]

            # construct the multiple mask
            multiple = unique_multiple[i]

            # construct the others mask
            others = 1 if object_cat[i] == 17 else 0

            query = "{}-{}-{}".format(scene_id, object_id, ann_id)
            if query not in cached:
                pred = {
                    "scene_id": scene_id,
                    "object_id": object_id,
                    "ann_id": ann_id,
                    "bbox": pred_bbox.tolist(),
                    "unique_multiple": int(multiple),
                    "others": int(others)
                }
                predictions.append(pred)

                cached.append(query)

    output_path = os.path.join(cfg.OUTPUT_PATH, args.folder, "pred.json")
    with open(output_path, "w") as f:
        json.dump(predictions, f, indent=4)

    print("done!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-f", "--folder", type=str, required=True, help="path to folder with model")
    args = parser.parse_args()

    os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

    print("=> loading configurations...")
    cfg = load_conf(args)

    print("=> initializing data...")
    dataset, dataloader = init_data(cfg)

    print("=> initializing model...")
    model = init_model(cfg, dataset)
    
    print("=> start predicting...")
    predict(args, cfg, dataset["test"], dataloader["test"], model)
            