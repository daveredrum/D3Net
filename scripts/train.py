import warnings

from torch.utils.data import dataloader
warnings.filterwarnings("ignore")

import os
import sys
import json
import torch
import random
import argparse

import pytorch_lightning as pl

from copy import deepcopy
from omegaconf import OmegaConf
from importlib import import_module

from pytorch_lightning.plugins import DDPPlugin

from torch.utils.data.dataloader import DataLoader

sys.path.append(".")

def load_conf(args):
    base_cfg = OmegaConf.load("conf/path.yaml")
    cfg = OmegaConf.load(args.config)
    cfg = OmegaConf.merge(base_cfg, cfg)
    
    root = os.path.join(cfg.general.output_root, cfg.general.experiment.upper())
    os.makedirs(root, exist_ok=True)

    cfg.general.task = "train"
    cfg.general.root = root

    cfg_backup_path = os.path.join(cfg.general.root, "config.yaml")
    OmegaConf.save(cfg, cfg_backup_path)

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

    print("=> loaded {} samples for training and {} samples for valiation...".format(len(data_train), len(data_val)))

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

    new_val = []
    for scene_id in raw_val_scan_list:
        data = deepcopy(raw_val[0])
        data["scene_id"] = scene_id
        new_val.append(data)

    raw_val = new_val

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

    print("=> loaded {} samples for training and {} samples for validation...".format(len(data_train), len(data_val)))

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

    print("=> use {} samples for training".format(len(dataset_train)))

    print("=> loading val split for listener...")
    dataset_val = Dataset(cfg, cfg.general.dataset, "listener", "val", data_val, scan_list_val, SCAN2CAD)

    print("=> use {} samples for training".format(len(dataset_val)))

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
    datasets, dataloaders = {}, {}

    assert not cfg.model.no_detection or not cfg.model.no_captioning or not cfg.model.no_grounding, \
        "invalid mode, detection: {}, captioning: {}, grounding: {}".format(
            not cfg.model.no_detection, not cfg.model.no_captioning, not cfg.model.no_grounding
        )

    # only load the detection data in pure detector mode
    if not cfg.model.no_detection and (cfg.model.no_captioning and cfg.model.no_grounding):
        dataset, dataloader = init_detector_data(cfg)
        datasets["det"] = dataset
        dataloaders["det"] = dataloader        

    # only load the detection data in pure detector mode
    if not cfg.model.no_captioning:
        dataset, dataloader = init_speaker_data(cfg)
        datasets["spk"] = dataset
        dataloaders["spk"] = dataloader

    if not cfg.model.no_grounding:
        dataset, dataloader = init_listener_data(cfg)
        datasets["lis"] = dataset
        dataloaders["lis"] = dataloader

    return datasets, dataloaders

def init_logger(cfg):
    logger = pl.loggers.TensorBoardLogger(cfg.general.root, name="logs")

    return logger

def init_monitor(cfg):
    monitor = pl.callbacks.ModelCheckpoint(
        monitor="{}".format(cfg.general.monitor),
        mode="{}".format(cfg.general.monitor_mode),
        # save_weights_only=True,
        dirpath=cfg.general.root,
        filename="model",
        save_last=True
    )

    return monitor

def init_trainer(cfg):
    trainer = pl.Trainer(
        gpus=-1, # use all available GPUs 
        strategy="ddp_find_unused_parameters_false",
        accelerator="gpu", # use multiple GPUs on the same machine
        max_epochs=cfg.train.epochs, 
        num_sanity_val_steps=cfg.train.num_sanity_val_steps, # validate on all val data before training 
        log_every_n_steps=cfg.train.log_every_n_steps,
        check_val_every_n_epoch=cfg.train.check_val_every_n_epoch,
        callbacks=[monitor], # comment when debug
        logger=logger,
        profiler="simple",
        # resume_from_checkpoint=checkpoint,
        # plugins=DDPPlugin(find_unused_parameters=False)
    )

    return trainer

def init_model(cfg, dataset):
    PipelineNet = getattr(import_module("model.pipeline"), "PipelineNet")
    model = PipelineNet(cfg, dataset["spk"] if not cfg.model.no_captioning else None)

    print("=> current mode: {}...".format(model.mode))

    if cfg.model.pretrained_detector and not cfg.model.no_detection and not cfg.model.use_checkpoint:
        device_name = "cuda:{}".format(os.environ.get("LOCAL_RANK", 0))

        print("=> loading pretrained detector to {} ...".format(device_name))
        detector_path = os.path.join(cfg.PRETRAINED_PATH, cfg.model.pretrained_detector)
        detector_weights = torch.load(detector_path, map_location=device_name)
        model.detector.load_state_dict(detector_weights)

    if cfg.model.pretrained_speaker and not cfg.model.no_captioning and not cfg.model.use_checkpoint:
        device_name = "cuda:{}".format(os.environ.get("LOCAL_RANK", 0))

        print("=> loading pretrained speaker to {} ...".format(device_name))
        speaker_path = os.path.join(cfg.PRETRAINED_PATH, cfg.model.pretrained_speaker)
        speaker_weights = torch.load(speaker_path, map_location=device_name)
        model.speaker.load_state_dict(speaker_weights)

    if cfg.model.pretrained_listener and not cfg.model.no_grounding and not cfg.model.use_checkpoint:
        device_name = "cuda:{}".format(os.environ.get("LOCAL_RANK", 0))

        print("=> loading pretrained listener to {} ...".format(device_name))
        listener_path = os.path.join(cfg.PRETRAINED_PATH, cfg.model.pretrained_listener)
        listener_weights = torch.load(listener_path, map_location=device_name)
        model.listener.load_state_dict(listener_weights)

    if cfg.model.freeze_detector:
        print("=> freezing detector...")
        for param in model.detector.parameters():
            param.requires_grad = False

    if cfg.model.freeze_speaker:
        print("=> freezing speaker...")
        for param in model.speaker.parameters():
            param.requires_grad = False

    if cfg.model.freeze_listener:
        print("=> freezing listener...")
        for param in model.listener.parameters():
            param.requires_grad = False

    return model

def start_training(trainer, model, dataloaders):
    print("=> train with MODE: {}".format(model.mode))

    if cfg.model.use_checkpoint:
        print("=> configuring trainer with checkpoint from {} ...".format(cfg.model.use_checkpoint))
        checkpoint = os.path.join(cfg.general.output_root, cfg.model.use_checkpoint, "last.ckpt")
    else:
        checkpoint = None

    if model.mode == 0:
        trainer.fit(
            model=model, 
            train_dataloaders=dataloaders["det"]["train"], 
            val_dataloaders=dataloaders["det"]["val"],
            ckpt_path=checkpoint
        )
    elif model.mode == 1 or model.mode == 4:
        trainer.fit(
            model=model, 
            train_dataloaders=dataloaders["spk"]["train"], 
            val_dataloaders=dataloaders["spk"]["val"],
            ckpt_path=checkpoint
        )
    elif model.mode == 2 or model.mode == 5:
        trainer.fit(
            model=model, 
            train_dataloaders=dataloaders["lis"]["train"], 
            val_dataloaders=dataloaders["lis"]["val"],
            ckpt_path=checkpoint
        )
    elif model.mode == 3 or model.mode == 6:
        trainer.fit(
            model=model, 
            train_dataloaders=[dataloaders["spk"]["train"], dataloaders["lis"]["train"]], 
            val_dataloaders=[dataloaders["spk"]["val"], dataloaders["lis"]["val"]],
            ckpt_path=checkpoint
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--config", type=str, default="conf/pointgroup_speaker-listener.yaml", help="path to config file")
    args = parser.parse_args()

    os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

    print("=> loading configurations...")
    cfg = load_conf(args)

    print("=> initializing data...")
    datasets, dataloaders = init_data(cfg)

    print("=> initializing model...")
    model = init_model(cfg, datasets)

    print("=> initializing logger...")
    logger = init_logger(cfg)
    
    print("=> initializing monitor...")
    monitor = init_monitor(cfg)

    print("=> initializing trainer...")
    trainer = init_trainer(cfg)

    print("=> start training...")
    start_training(trainer, model, dataloaders)
