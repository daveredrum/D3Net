from email.policy import strict
import warnings
warnings.filterwarnings("ignore")

import os
import sys
import json
import torch
import random
import argparse

from copy import deepcopy
from omegaconf import OmegaConf
from importlib import import_module

from torch.utils.data.dataloader import DataLoader

sys.path.append(".")

def load_conf(args):
    base_cfg = OmegaConf.load("conf/path.yaml")
    cfg = OmegaConf.load(args.config)
    cfg = OmegaConf.merge(base_cfg, cfg)
    
    checkpoint_path = args.path

    # HACK manually setting those properties
    cfg.general.checkpoint_path = checkpoint_path

    return cfg

def init_detector_data(cfg):
    DATA_MODULE = import_module(cfg.data.module)
    Dataset = getattr(DATA_MODULE, cfg.data.dataset)
    collate_fn = getattr(DATA_MODULE, "sparse_collate_fn")

    SCAN2CAD = json.load(open(cfg.SCAN2CAD))

    train_scan_list = [s.rstrip() for s in open(cfg.SCANNETV2_PATH.train_list).readlines()]
    val_scan_list = [s.rstrip() for s in open(cfg.SCANNETV2_PATH.val_list).readlines()]

    data_train = []
    for scan_id in train_scan_list:
        extry = {
            "scene_id": scan_id,
            "object_id": "SYNTHETIC",
            "object_name": "SYNTHETIC",
            "ann_id": "SYNTHETIC",
            "description": "SYNTHETIC",
            "token": ["SYNTHETIC"],
        }
        data_train.append(extry)

    data_val = []
    for scan_id in val_scan_list:
        extry = {
            "scene_id": scan_id,
            "object_id": "SYNTHETIC",
            "object_name": "SYNTHETIC",
            "ann_id": "SYNTHETIC",
            "description": "SYNTHETIC",
            "token": ["SYNTHETIC"],
        }
        data_val.append(extry)


    print("=> loading train split for detector...")
    dataset_train = Dataset(cfg, cfg.general.dataset, "speaker", "train", data_train, train_scan_list, SCAN2CAD)

    print("=> loading val split for detector...")
    dataset_val = Dataset(cfg, cfg.general.dataset, "speaker", "val", data_val, val_scan_list, SCAN2CAD)

    print("=> loading complete")

    datasets = {
        "train": dataset_train,
        "val": dataset_val
    }

    dataloaders = {
        "train": DataLoader(dataset_train, batch_size=cfg.data.batch_size, \
                shuffle=True, pin_memory=True, num_workers=cfg.data.num_workers, collate_fn=collate_fn),
        "val": DataLoader(dataset_val, batch_size=cfg.data.batch_size, \
                shuffle=False, pin_memory=True, num_workers=cfg.data.num_workers, collate_fn=collate_fn),
    }

    return datasets, dataloaders

def init_speaker_data(cfg):
    DATA_MODULE = import_module(cfg.data.module)
    Dataset = getattr(DATA_MODULE, cfg.data.dataset)
    collate_fn = getattr(DATA_MODULE, "sparse_collate_fn")

    SCAN2CAD = json.load(open(cfg.SCAN2CAD))

    raw_train = json.load(open(cfg["{}_PATH".format(cfg.general.dataset.upper())].train_split))
    raw_val = json.load(open(cfg["{}_PATH".format(cfg.general.dataset.upper())].val_split))

    raw_train_scan_list = sorted(list(set([data["scene_id"] for data in raw_train])))
    raw_val_scan_list = sorted(list(set([data["scene_id"] for data in raw_val])))

    data_train = deepcopy(raw_train)
    data_val = deepcopy(raw_val)
    scan_list_train = deepcopy(raw_train_scan_list)
    scan_list_val = deepcopy(raw_val_scan_list)

    # extra data for speaker
    if cfg.data.extra_ratio > 0:
        num_extra = int(len(raw_train) * cfg.data.extra_ratio)
        all_scans_in_scannet = [s.rstrip() for s in open(cfg.SCANNETV2_PATH.train_list).readlines()]
        extra_scans = [s for s in all_scans_in_scannet if s not in scan_list_train]

        extra_data = []
        for i in range(num_extra): # fill with synthetic data entries
            extra_scan_id = extra_scans[i] if i < len(extra_scans) else random.choice(extra_scans)

            extry = {
                "scene_id": extra_scan_id,
                "object_id": "SYNTHETIC",
                "object_name": "SYNTHETIC",
                "ann_id": "SYNTHETIC",
                "description": "SYNTHETIC",
                "token": ["SYNTHETIC"],
            }
            extra_data.append(extry)

        # add to current train data
        data_train += extra_data
        scan_list_train += sorted(list(set([data["scene_id"] for data in extra_data])))

        # shuffle
        random.shuffle(data_train)
    
    print("=> loading train split for speaker...")
    dataset_train = Dataset(cfg, cfg.general.dataset, "speaker", "train", data_train, scan_list_train, SCAN2CAD)

    print("=> loading val split for speaker...")
    dataset_val = Dataset(cfg, cfg.general.dataset, "speaker", "val", data_val, scan_list_val, SCAN2CAD)

    print("=> loading complete")

    datasets = {
        "train": dataset_train,
        "val": dataset_val
    }

    dataloaders = {
        "train": DataLoader(dataset_train, batch_size=cfg.data.batch_size, \
                shuffle=True, pin_memory=True, num_workers=cfg.data.num_workers, collate_fn=collate_fn),
        "val": DataLoader(dataset_val, batch_size=cfg.data.batch_size, \
                shuffle=False, pin_memory=True, num_workers=cfg.data.num_workers, collate_fn=collate_fn),
    }

    return datasets, dataloaders

def init_listener_data(cfg):
    DATA_MODULE = import_module(cfg.data.module)
    Dataset = getattr(DATA_MODULE, cfg.data.dataset)
    collate_fn = getattr(DATA_MODULE, "sparse_collate_fn")

    SCAN2CAD = json.load(open(cfg.SCAN2CAD))

    raw_train = json.load(open(cfg["{}_PATH".format(cfg.general.dataset.upper())].train_split))
    raw_val = json.load(open(cfg["{}_PATH".format(cfg.general.dataset.upper())].val_split))

    raw_train_scan_list = sorted(list(set([data["scene_id"] for data in raw_train])))
    raw_val_scan_list = sorted(list(set([data["scene_id"] for data in raw_val])))

    data_train = deepcopy(raw_train)
    data_val = deepcopy(raw_val)
    scan_list_train = deepcopy(raw_train_scan_list)
    scan_list_val = deepcopy(raw_val_scan_list)
    
    print("=> loading train split for listener...")
    dataset_train = Dataset(cfg, cfg.general.dataset, "listener", "train", data_train, scan_list_train, SCAN2CAD)

    print("=> loading val split for listener...")
    dataset_val = Dataset(cfg, cfg.general.dataset, "listener", "val", data_val, scan_list_val, SCAN2CAD)

    print("=> loading complete")

    datasets = {
        "train": dataset_train,
        "val": dataset_val
    }

    dataloaders = {
        "train": DataLoader(dataset_train, batch_size=cfg.data.batch_size, \
                shuffle=True, pin_memory=True, num_workers=cfg.data.num_workers, collate_fn=collate_fn),
        "val": DataLoader(dataset_val, batch_size=cfg.data.batch_size, \
                shuffle=False, pin_memory=True, num_workers=cfg.data.num_workers, collate_fn=collate_fn),
    }

    return datasets, dataloaders

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

    print("=> loading train split for captioning...")
    cap_train_dataset = Dataset(cfg, cfg.general.dataset, "speaker", "train", raw_train, raw_train_scan_list, SCAN2CAD)

    print("=> loading val split for captioning...")
    cap_val_dataset = Dataset(cfg, cfg.general.dataset, "speaker", "val", raw_val, raw_val_scan_list, SCAN2CAD)

    print("=> loading complete")

    train_dataloader = DataLoader(cap_train_dataset, batch_size=cfg.data.batch_size, \
        shuffle=True, pin_memory=True, num_workers=cfg.data.num_workers, collate_fn=collate_fn)
    val_dataloader = DataLoader(cap_val_dataset, batch_size=cfg.data.batch_size, \
        shuffle=False, pin_memory=True, num_workers=cfg.data.num_workers, collate_fn=collate_fn)

    dataset = {
        "train": cap_train_dataset,
        "val": cap_val_dataset,
    }

    dataloader = {
        "train": train_dataloader,
        "val": val_dataloader,
    }

    return dataset, dataloader

def init_model(cfg, dataset=None):
    PipelineNet = getattr(import_module("model.pipeline"), "PipelineNet")
    model = PipelineNet(cfg, dataset if not cfg.model.no_captioning else None)

    print("=> current mode: {}...".format(model.mode))

    checkpoint_path = os.path.join(cfg.general.checkpoint_path)
    checkpoint = torch.load(checkpoint_path)

    model.load_state_dict(checkpoint["state_dict"], strict=False)

    return model



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-p", "--path", type=str, required=True, help="path to the model")
    parser.add_argument("-c", "--config", type=str, default="conf/pointgroup_grounding.yaml", help="path to config file")
    parser.add_argument("-m", "--model", type=str, required=True, help="which model to save", 
        choices=["detector", "speaker", "listener"])
    parser.add_argument("-n", "--model_name", type=str, required=True, help="name of the checkpoint")
    args = parser.parse_args()

    os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

    print("=> loading configurations...")
    cfg = load_conf(args)

    print("=> initializing data...")
    datasets, dataloaders = init_data(cfg)

    print("=> initializing model...")
    model = init_model(cfg, datasets)

    print("=> saving modules...")
    if args.model == "speaker":
        torch.save(model.speaker.state_dict(), os.path.join(cfg.PRETRAINED_PATH, "{}.pth".format(args.model_name)))
    elif args.model == "listener":
        torch.save(model.listener.state_dict(), os.path.join(cfg.PRETRAINED_PATH, "{}.pth".format(args.model_name)))
    else:
        torch.save(model.detector.state_dict(), os.path.join(cfg.PRETRAINED_PATH, "{}.pth".format(args.model_name)))
    
    print("done!")
