import os
import sys
import time
import json
import h5py 
import torch
import pickle
import random

import numpy as np
import multiprocessing as mp

from copy import deepcopy
from itertools import chain
from collections import Counter
from tqdm import tqdm
from torch.utils.data import Dataset
from MinkowskiEngine.utils import batched_coordinates

sys.path.append("../")  # HACK add the lib folder
from data.scannet.model_util_scannet import ScannetDatasetConfig, rotate_aligned_boxes_along_axis
from lib.pointgroup_ops.functions import pointgroup_ops
from lib.utils.pc import crop
from lib.utils.transform import jitter, flip, rotz, elastic
from lib.utils.bbox import get_3d_box, get_3d_box_batch

MEAN_COLOR_RGB = np.array([109.8, 97.2, 83.8])

class PipelineDataset(Dataset):

    def __init__(self, cfg, name, mode, split, raw_data, scan_list, scan2cad_rotation=None, is_augment=False):
        self.cfg = cfg
        self.name = name
        self.mode = mode
        self.split = split
        self.raw_data = raw_data
        self.scan_list = scan_list
        self.use_gt = cfg.model.no_detection
        self.scan2cad_rotation = scan2cad_rotation
        self.is_augment = is_augment

        # unpack configurations from config file
        self.DC = ScannetDatasetConfig(self.cfg)
        self.root = cfg.SCANNETV2_PATH.split_data
        self.file_suffix = cfg.data.file_suffix

        self.full_scale = cfg.data.full_scale
        self.scale = cfg.data.scale
        self.max_num_point = cfg.data.max_num_point

        self.max_des_len = cfg.data.max_spk_len if self.mode == "speaker" else cfg.data.max_lis_len

        self.use_color = cfg.model.use_color
        self.use_multiview = cfg.model.use_multiview
        self.use_normal = cfg.model.use_normal
        self.requires_bbox = cfg.data.requires_bbox
    
        self.multiview_data = {}
        self.gt_feature_data = {}
        
        self._load()

    def __len__(self):
        return len(self.chunked_data)

    def __getitem__(self, idx):
        scene_id = self.chunked_data[idx][0]["scene_id"]

        chunk_id_list = np.zeros((self.chunk_size))
        object_id_list = np.zeros((self.chunk_size))
        ann_id_list = np.zeros((self.chunk_size))
        lang_feat_list = np.zeros((self.chunk_size, self.max_des_len + 2, 300))
        lang_len_list = np.zeros((self.chunk_size))
        lang_id_list = np.zeros((self.chunk_size, self.max_des_len + 2))
        annotated_list = np.zeros((self.chunk_size))
        unique_multiple_list = np.zeros((self.chunk_size))
        object_cat_list = np.zeros((self.chunk_size))

        actual_chunk_size = len(self.chunked_data[idx])

        # prepare scene data
        scene = self.scenes[scene_id]
        points, feats = self._get_coord_and_feat_from_mesh(scene["aligned_mesh"], scene_id)
        
        # print("point data loaded.")
        # store to dict
        data = {}
        data["id"] = np.array(idx).astype(np.int64)

        if self.split != "test" and self.cfg.general.task != "test":
            for i in range(self.chunk_size):
                if i < actual_chunk_size:
                    chunk_id = i
                    object_id = self.chunked_data[idx][i]["object_id"]
                    if object_id != "SYNTHETIC":
                        annotated = 1
                        object_id = int(object_id)
                        object_name = " ".join(self.chunked_data[idx][i]["object_name"].split("_"))
                        ann_id = self.chunked_data[idx][i]["ann_id"]
                        object_cat = self.raw2label[object_name] if object_name in self.raw2label else 17
                        
                        # get language features
                        lang_feat = deepcopy(self.lang[scene_id][str(object_id)][ann_id])
                        lang_len = len(self.chunked_data[idx][i]["token"]) + 2
                        lang_len = lang_len if lang_len <= self.max_des_len + 2 else self.max_des_len + 2

                        # NOTE 50% chance that 20% of the tokens are erased during training
                        if self.is_augment and random.random() < 0.5 and self.cfg.train.apply_word_erase:
                            lang_feat = self._tranform_des_with_erase(lang_feat, lang_len, p=0.2)

                        lang_ids = self.lang_ids[scene_id][str(object_id)][ann_id]
                        unique_multiple_flag = self.unique_multiple_lookup[scene_id][str(object_id)][ann_id]
                    else:
                        annotated = 0
                        object_id = -1
                        object_name = ""
                        ann_id = -1
                        object_cat = 17 # will be changed in the model

                        # synthesize language features
                        lang_feat = np.zeros((self.max_des_len + 2, 300))
                        lang_len = 0

                        lang_ids = np.zeros(self.max_des_len + 2)
                        unique_multiple_flag = 0

                # store
                # HACK the last sample will be repeated if chunk size 
                # is smaller than num_des_per_scene
                chunk_id_list[i] = chunk_id
                object_id_list[i] = object_id
                ann_id_list[i] = ann_id
                lang_feat_list[i] = lang_feat
                lang_len_list[i] = lang_len
                lang_id_list[i] = lang_ids
                annotated_list[i] = annotated
                unique_multiple_list[i] = unique_multiple_flag
                object_cat_list[i] = object_cat

            if not self.use_gt:
                instance_ids = scene["instance_ids"]
                sem_labels = scene["sem_labels"]  # {0,1,...,19}, -1 as ignored (unannotated) class
                
                # augment and scale
                points_augment = self._augment(points)[0] if self.is_augment else points.copy()
                points = points_augment * self.scale
                
                # elastic
                # if self.is_augment and self.cfg.general.task == "train":
                if self.is_augment and (not self.cfg.model.no_detection and self.cfg.model.no_captioning and self.cfg.model.no_grounding):
                    points = elastic(points, 6 * self.scale // 50, 40 * self.scale / 50)
                    points = elastic(points, 20 * self.scale // 50, 160 * self.scale / 50)
                
                # offset
                points -= points.min(0)

                # if self.is_augment and self.cfg.general.task == "train":
                if self.is_augment and (not self.cfg.model.no_detection and self.cfg.model.no_captioning and self.cfg.model.no_grounding):
                    ### crop
                    points, valid_idxs = crop(points, self.max_num_point, self.full_scale[1])
                    
                    points = points[valid_idxs]
                    points_augment = points_augment[valid_idxs]
                    feats = feats[valid_idxs]
                    sem_labels = sem_labels[valid_idxs]
                    instance_ids = self._croppedInstanceIds(instance_ids, valid_idxs)

                (num_instance, instance_info, instance_num_point, 
                instance_bboxes, instance_bboxes_semcls, instance_bbox_ids, 
                angle_classes, angle_residuals, 
                size_classes, size_residuals, bbox_label) = self._getInstanceInfo(points_augment, instance_ids, sem_labels)

                if self.cfg.data.requires_gt_mask:
                    gt_proposals_idx, gt_proposals_offset, _, _ = self._generate_gt_clusters(points, instance_ids)
                    
                    data["gt_proposals_idx"] = gt_proposals_idx
                    data["gt_proposals_offset"] = gt_proposals_offset

                # for instance segmentation
                data["locs"] = points_augment.astype(np.float32)  # (N, 3)
                data["locs_scaled"] = points.astype(np.float32)  # (N, 3)
                data["feats"] = feats.astype(np.float32)  # (N, 3)
                data["sem_labels"] = sem_labels.astype(np.int32)  # (N,)
                data["instance_ids"] = instance_ids.astype(np.int32)  # (N,) 0~total_nInst, -1
                data["num_instance"] = np.array(num_instance).astype(np.int32)  # int
                data["instance_info"] = instance_info.astype(np.float32)  # (N, 12)
                data["instance_num_point"] = np.array(instance_num_point).astype(np.int32)  # (num_instance,)

            else: # pre-computed GT bounding boxes and features
                # gt_object_ids = np.zeros((self.cfg.data.max_num_instance,))
                # gt_features = np.zeros((self.cfg.data.max_num_instance, self.cfg.model.m))
                # gt_corners = np.zeros((self.cfg.data.max_num_instance, 8, 3))
                # gt_centers = np.zeros((self.cfg.data.max_num_instance, 3))
                # gt_scores = np.zeros((self.cfg.data.max_num_instance,))
                # gt_masks = np.zeros((self.cfg.data.max_num_instance,))
                # gt_sems = np.zeros((self.cfg.data.max_num_instance,))

                # cur_object_ids, cur_features, cur_corners, cur_centers, cur_sems = self._get_feature(scene_id)
                # num_valid_objects = cur_object_ids.shape[0]
                # gt_object_ids[:num_valid_objects] = cur_object_ids
                # gt_features[:num_valid_objects] = cur_features
                # gt_corners[:num_valid_objects] = cur_corners
                # gt_centers[:num_valid_objects] = cur_centers
                # gt_scores[:num_valid_objects] = 1
                # gt_masks[:num_valid_objects] = 1
                # gt_sems[:num_valid_objects] = cur_sems

                # if self.is_augment and self.cfg.general.task == "train":
                #     ids = np.random.permutation(self.cfg.data.max_num_instance)
                #     gt_object_ids = gt_object_ids[ids]
                #     gt_features = gt_features[ids]
                #     gt_corners = gt_corners[ids]
                #     gt_centers = gt_centers[ids]
                #     gt_scores = gt_scores[ids]
                #     gt_masks = gt_masks[ids]
                #     gt_sems = gt_sems[ids]

                # # assignments
                # _, assignments = self._nn_distance(gt_centers, gt_centers.astype(np.float32)[:,0:3])

                # # current target
                # bbox_idx = np.zeros((self.chunk_size))
                # for j in range(len(object_id_list)):
                #     for i in range(len(gt_object_ids)):
                #         if gt_masks[i] == 1 and gt_object_ids[i] == object_id_list[j]:
                #             bbox_idx[j] = i

                # instance_bboxes = self._conver_corners_to_cwdh(gt_corners)
                # instance_bboxes_semcls = gt_sems
                # instance_bbox_ids = gt_object_ids
                # angle_classes = np.zeros((self.cfg.data.max_num_instance,))
                # angle_residuals = np.zeros((self.cfg.data.max_num_instance,))
                # size_classes = np.zeros((self.cfg.data.max_num_instance,))
                # size_residuals = np.zeros((self.cfg.data.max_num_instance, 3))
                # bbox_label = gt_masks

                # # store
                # data["proposal_object_ids"] = gt_object_ids.astype(np.int64)
                # data["proposal_feats_batched"] = gt_features.astype(np.float32)
                # data["proposal_bbox_batched"] = gt_corners.astype(np.float32)
                # data["proposal_center_batched"] = gt_centers.astype(np.float32)
                # data["proposal_sem_cls_batched"] = gt_sems.astype(np.int64)
                # data["proposal_scores_batched"] = gt_scores.astype(np.int64)
                # data["proposal_batch_mask"] = gt_masks.astype(np.int64)
                # data["object_assignment"] = assignments.astype(np.int64)
                # data["bbox_idx"] = bbox_idx.astype(np.int64)

                raise NotImplementedError("GTs are not ready yet.")

             # object rotations
            scene_object_rotations = np.zeros((self.cfg.data.max_num_instance, 3, 3))
            scene_object_rotation_masks = np.zeros((self.cfg.data.max_num_instance,)) # NOTE this is not object mask!!!
            # if scene is not in scan2cad annotations, skip
            # if the instance is not in scan2cad annotations, skip
            if self.scan2cad_rotation and scene_id in self.scan2cad_rotation:
                for i, instance_id in enumerate(instance_bbox_ids.astype(int)):
                    if bbox_label[i] == 1:
                        try:
                            rotation = np.array(self.scan2cad_rotation[scene_id][str(instance_id)])

                            scene_object_rotations[i] = rotation
                            scene_object_rotation_masks[i] = 1
                        except KeyError:
                            pass

            # construct the reference target label for each bbox
            ref_box_label_list = np.zeros((self.chunk_size, self.cfg.data.max_num_instance))
            ref_box_corner_label_list = np.zeros((self.chunk_size, 8, 3)) # NOTE the grounding GT should be decoded
            for j in range(self.chunk_size):
                ref_box_label = np.zeros(self.cfg.data.max_num_instance)
                for i, gt_id in enumerate(instance_bbox_ids):
                    if bbox_label[i] == 1 and gt_id == object_id_list[j]:
                        ref_box_label[i] = 1
                        ref_box_corner_label = get_3d_box(instance_bboxes[i, 0:3], instance_bboxes[i, 3:6]).astype(np.float32)

                        # store
                        ref_box_label_list[j] = ref_box_label
                        ref_box_corner_label_list[j] = ref_box_corner_label


            # basic info
            data["istrain"] = np.array(1) if self.split == "train" else np.array(0)
            data["annotated"] = np.array(annotated_list).astype(np.int64)
            data["chunk_ids"] = np.array(chunk_id_list).astype(np.int64)

            # for bbox regression - DEPRECATED
            data["center_label"] = instance_bboxes.astype(np.float32)[:,0:3] # (num_instance, 3) for GT box center XYZ
            data["sem_cls_label"] = instance_bboxes_semcls.astype(np.int64) # (num_instance,) semantic class index
            data["heading_class_label"] = angle_classes.astype(np.int64) # (num_instance,) with int values in 0,...,NUM_HEADING_BIN-1
            data["heading_residual_label"] = angle_residuals.astype(np.float32) # (num_instance,)
            data["size_class_label"] = size_classes.astype(np.int64) # (num_instance,) with int values in 0,...,NUM_SIZE_CLUSTER
            data["size_residual_label"] = size_residuals.astype(np.float32) # (num_instance, 3)
            
            # bbox GT labels
            data["gt_bbox_object_id"] = instance_bbox_ids.astype(np.int64)
            data["gt_bbox_label"] = bbox_label.astype(np.int64)
            data["gt_bbox"] = get_3d_box_batch(instance_bboxes[:, 0:3], instance_bboxes[:, 3:6], angle_classes).astype(np.float32) # (num_instance, 8, 3)
        
            # language data
            data["lang_feat"] = np.array(lang_feat_list).astype(np.float32) # language feature vectors
            data["lang_len"] = np.array(lang_len_list).astype(np.int64) # length of each description
            data["lang_ids"] = np.array(lang_id_list).astype(np.int64)

            # object language labels
            data["object_id"] = np.array(object_id_list).astype(np.int64)
            data["ann_id"] = np.array(ann_id_list).astype(np.int64)
            data["object_cat"] = np.array(object_cat_list).astype(np.int64)
            data["unique_multiple"] = np.array(unique_multiple_list).astype(np.int64)

            # rotation
            data["scene_object_ids"] = instance_bbox_ids.astype(np.int64) # (MAX_NUM_OBJ,) object ids of all objects
            data["scene_object_rotations"] = scene_object_rotations.astype(np.float32) # (MAX_NUM_OBJ, 3, 3)
            data["scene_object_rotation_masks"] = scene_object_rotation_masks.astype(np.int64) # (MAX_NUM_OBJ)

            # grounding GT
            data["ref_box_label"] = np.array(ref_box_label_list).astype(np.int64) # 0/1 reference labels for each object bbox
            data["ref_box_corner_label"] = np.array(ref_box_corner_label_list).astype(np.float32)

        else:
            for i in range(self.chunk_size):
                if i < actual_chunk_size:
                    chunk_id = i
                    object_id = self.chunked_data[idx][i]["object_id"]
                    annotated = 1
                    object_id = int(object_id)
                    object_name = " ".join(self.chunked_data[idx][i]["object_name"].split("_"))
                    ann_id = self.chunked_data[idx][i]["ann_id"]
                    object_cat = self.raw2label[object_name] if object_name in self.raw2label else 17
                    
                    # get language features
                    lang_feat = deepcopy(self.lang[scene_id][str(object_id)][ann_id])
                    lang_len = len(self.chunked_data[idx][i]["token"]) + 2
                    lang_len = lang_len if lang_len <= self.max_des_len + 2 else self.max_des_len + 2

                    lang_ids = self.lang_ids[scene_id][str(object_id)][ann_id]
                    unique_multiple_flag = self.unique_multiple_lookup[scene_id][str(object_id)][ann_id]

                # store
                # HACK the last sample will be repeated if chunk size 
                # is smaller than num_des_per_scene
                chunk_id_list[i] = chunk_id
                object_id_list[i] = object_id
                ann_id_list[i] = ann_id
                lang_feat_list[i] = lang_feat
                lang_len_list[i] = lang_len
                lang_id_list[i] = lang_ids
                annotated_list[i] = annotated
                unique_multiple_list[i] = unique_multiple_flag
                object_cat_list[i] = object_cat

            # augment
            points_augment = points.copy()
            
            # scale
            points = points_augment * self.scale
            # points *= self.scale
            
            # offset
            points -= points.min(0)

            # basic info
            data["istrain"] = np.array(1) if self.split == "train" else np.array(0)
            data["annotated"] = np.array(annotated_list).astype(np.int64)
            data["chunk_ids"] = np.array(chunk_id_list).astype(np.int64)

            # language data
            data["lang_feat"] = np.array(lang_feat_list).astype(np.float32) # language feature vectors
            data["lang_len"] = np.array(lang_len_list).astype(np.int64) # length of each description
            data["lang_ids"] = np.array(lang_id_list).astype(np.int64)

            # object language labels
            data["object_id"] = np.array(object_id_list).astype(np.int64)
            data["ann_id"] = np.array(ann_id_list).astype(np.int64)
            data["object_cat"] = np.array(object_cat_list).astype(np.int64)
            data["unique_multiple"] = np.array(unique_multiple_list).astype(np.int64)

            data["locs"] = points_augment.astype(np.float32)  # (N, 3)
            data["locs_scaled"] = points.astype(np.float32)  # (N, 3)
            data["feats"] = feats.astype(np.float32)  # (N, 3)

        return data

    def _load(self):
        # loading preprocessed scene data
        if not self.use_gt:
            self.scenes = {scene_id: torch.load(os.path.join(self.root, self.split, scene_id+self.file_suffix))
                for scene_id in tqdm(self.scan_list)}

        # load language features
        self.glove = np.load(self.cfg["{}_PATH".format(self.name.upper())].glove_numpy)
        self.vocabulary = self._build_vocabulary(self.max_des_len)
        self.lang, self.lang_ids = self._tranform_des(self.max_des_len)
        self.organized = self._organize_data()

        # chunk data
        self.chunk_size = self.cfg.data.num_des_per_scene
        self.chunked_data = self._get_chunked_data(self.raw_data, self.chunk_size)

        # prepare class mapping
        lines = [line.rstrip() for line in open(self.cfg.SCANNETV2_PATH.combine_file)]
        lines = lines[1:]
        raw2nyuid = {}
        for i in range(len(lines)):
            elements = lines[i].split("\t")
            raw_name = elements[1]
            nyu40_name = int(elements[4])
            raw2nyuid[raw_name] = nyu40_name

        # store
        self.raw2nyuid = raw2nyuid
        self.raw2label = self._get_raw2label()
        if self.use_gt: self.objectid2label = self._get_objectid2label(self.scan_list, self.raw2label)
        self.unique_multiple_lookup = self._get_unique_multiple_lookup()

    def _get_objectid2label(self, scene_list, objectname2label):
        objectid2label = {}
        for scan_id in scene_list:
            entry = {}
            with open(os.path.join(self.cfg.SCANNETV2_PATH.raw_scans, scan_id, scan_id + ".aggregation.json")) as f:
                aggr = json.load(f)
                for data in aggr["segGroups"]:
                    object_id = int(data["objectId"])
                    object_name = data["label"]

                    object_label = int(objectname2label[object_name])

                    entry[object_id] = object_label

            objectid2label[scan_id] = entry

        return objectid2label

    def _build_vocabulary(self, max_len):
        vocab_path = self.cfg["{}_PATH".format(self.name.upper())].vocabulary
        if os.path.exists(vocab_path):
            vocabulary = json.load(open(vocab_path))
        else:
            if self.split == "train":
                train_data = [d for d in self.raw_data if d["object_id"] != "SYNTHETIC"]
                all_words = chain(*[data["token"][:max_len] for data in train_data])
                word_counter = Counter(all_words)
                word_counter = sorted([(k, v) for k, v in word_counter.items() if k in self.glove], key=lambda x: x[1], reverse=True)
                word_list = [k for k, _ in word_counter]

                # build vocabulary
                word2idx, idx2word = {}, {}
                spw = ["pad_", "unk", "sos", "eos"] # NOTE distinguish padding token "pad_" and the actual word "pad"
                for i, w in enumerate(word_list):
                    shifted_i = i + len(spw)
                    word2idx[w] = shifted_i
                    idx2word[shifted_i] = w

                # add special words into vocabulary
                for i, w in enumerate(spw):
                    word2idx[w] = i
                    idx2word[i] = w

                vocab = {
                    "word2idx": word2idx,
                    "idx2word": idx2word
                }
                json.dump(vocab, open(vocab_path, "w"), indent=4)

                vocabulary = vocab

        emb_mat_path = self.cfg["{}_PATH".format(self.name.upper())].glove_numpy
        if not os.path.exists(emb_mat_path):
            emb_pickle_path = self.cfg["{}_PATH".format(self.name.upper())].glove_pickle
            all_glove = pickle.load(open(emb_pickle_path, "rb"))

            glove_trimmed = np.zeros((len(self.vocabulary["word2idx"]), 300))
            for word, idx in self.vocabulary["word2idx"].items():
                emb = all_glove[word]
                glove_trimmed[int(idx)] = emb

            np.save(emb_mat_path, glove_trimmed)

        return vocabulary

    def _tranform_des(self, max_len):
        lang = {}
        label = {}
        for data in self.raw_data:
            scene_id = data["scene_id"]
            object_id = data["object_id"]
            ann_id = data["ann_id"]

            if object_id != "SYNTHETIC":

                if scene_id not in lang:
                    lang[scene_id] = {}
                    label[scene_id] = {}

                if object_id not in lang[scene_id]:
                    lang[scene_id][object_id] = {}
                    label[scene_id][object_id] = {}

                if ann_id not in lang[scene_id][object_id]:
                    lang[scene_id][object_id][ann_id] = {}
                    label[scene_id][object_id][ann_id] = {}

                # trim long descriptions
                tokens = data["token"][:max_len]

                # tokenize the description
                tokens = ["sos"] + tokens + ["eos"]
                embeddings = np.zeros((max_len + 2, 300))
                labels = np.zeros((max_len + 2)) # start and end

                # embeddings = np.zeros((max_len, 300))
                # labels = np.zeros((max_len)) # start and end

                # load
                for token_id in range(len(tokens)):
                    token = tokens[token_id] 
                    if token not in self.vocabulary["word2idx"]: token = "unk"

                    glove_id = int(self.vocabulary["word2idx"][token])                    
                    embeddings[token_id] = self.glove[glove_id]

                    if token_id < max_len + 2:
                        labels[token_id] = self.vocabulary["word2idx"][token]
                
                # store
                lang[scene_id][object_id][ann_id] = embeddings
                label[scene_id][object_id][ann_id] = labels

        return lang, label

    def _tranform_des_with_erase(self, lang_feat, lang_len, p=0.2):
        num_erase = int((lang_len - 2) * p)
        erase_ids = np.arange(1, lang_len - 2, 1).tolist()
        erase_ids = np.random.choice(erase_ids, num_erase, replace=False) # randomly pick indices of erased tokens
        
        unk_idx = int(self.vocabulary["word2idx"]["unk"])
        unk = self.glove[unk_idx] # 300
        unk_exp = unk.reshape((1, -1)).repeat(erase_ids.shape[0], axis=0)

        lang_feat[erase_ids] = unk_exp
        
        return lang_feat

    def _organize_data(self):
        organized = {}

        for data in self.raw_data:
            scene_id = data["scene_id"]
            object_id = data["object_id"]

            if object_id != "SYNTHETIC":

                if scene_id not in organized: organized[scene_id] = {}
                if object_id not in organized[scene_id]: organized[scene_id][object_id] = []

                organized[scene_id][object_id].append(data)

        return organized

    def _get_chunked_data(self, raw_data, chunk_size):
        # scene data lookup dict: <scene_id> -> [scene_data_1, scene_data_2, ...]
        scene_data_dict = {}
        for data in raw_data:
            scene_id = data["scene_id"]

            if scene_id not in scene_data_dict: scene_data_dict[scene_id] = []

            scene_data_dict[scene_id].append(data)

        # chunk data
        new_data = []
        for scene_id, scene_data_list in scene_data_dict.items():
            for cur_chunk in self._chunks(scene_data_list, chunk_size):
                new_data.append(cur_chunk)

        return new_data

    def _chunks(self, lst, n):
        """Yield successive n-sized chunks from lst."""
        for i in range(0, len(lst), n):
            yield lst[i:i + n]

    def _get_raw2label(self):
        # mapping
        scannet_labels = self.DC.type2class.keys()
        scannet2label = {label: i for i, label in enumerate(scannet_labels)}

        lines = [line.rstrip() for line in open(self.cfg.SCANNETV2_PATH.combine_file)]
        lines = lines[1:]
        raw2label = {}
        for i in range(len(lines)):
            label_classes_set = set(scannet_labels)
            elements = lines[i].split("\t")
            raw_name = elements[1]
            nyu40_name = elements[7]
            if nyu40_name not in label_classes_set:
                raw2label[raw_name] = scannet2label["others"]
            else:
                raw2label[raw_name] = scannet2label[nyu40_name]

        return raw2label

    def _get_unique_multiple_lookup(self):
        all_sem_labels = {}
        cache = {}
        for data in self.raw_data:
            scene_id = data["scene_id"]
            object_id = data["object_id"]
            object_name = " ".join(data["object_name"].split("_"))
            ann_id = data["ann_id"]

            if scene_id not in all_sem_labels:
                all_sem_labels[scene_id] = []

            if scene_id not in cache:
                cache[scene_id] = {}

            if object_id not in cache[scene_id]:
                cache[scene_id][object_id] = {}
                try:
                    all_sem_labels[scene_id].append(self.raw2label[object_name])
                except KeyError:
                    all_sem_labels[scene_id].append(17)

        # convert to numpy array
        all_sem_labels = {scene_id: np.array(all_sem_labels[scene_id]) for scene_id in all_sem_labels.keys()}

        unique_multiple_lookup = {}
        for data in self.raw_data:
            scene_id = data["scene_id"]
            object_id = data["object_id"]
            object_name = " ".join(data["object_name"].split("_"))
            ann_id = data["ann_id"]

            try:
                sem_label = self.raw2label[object_name]
            except KeyError:
                sem_label = 17

            unique_multiple = 0 if (all_sem_labels[scene_id] == sem_label).sum() == 1 else 1

            # store
            if scene_id not in unique_multiple_lookup:
                unique_multiple_lookup[scene_id] = {}

            if object_id not in unique_multiple_lookup[scene_id]:
                unique_multiple_lookup[scene_id][object_id] = {}

            if ann_id not in unique_multiple_lookup[scene_id][object_id]:
                unique_multiple_lookup[scene_id][object_id][ann_id] = None

            unique_multiple_lookup[scene_id][object_id][ann_id] = unique_multiple

        return unique_multiple_lookup

    def _augment(self, xyz, instance_bboxes=None, return_mat=False):
        m = np.eye(3)
        if self.cfg.data.transform.jitter:
            m *= jitter()
        if self.cfg.data.transform.flip:
            flip_m = flip(0, random=True)
            m *= flip_m
            if instance_bboxes is not None and flip_m[0, 0] == -1:
                instance_bboxes[:, 0] = -1 * instance_bboxes[:, 0]    
        if self.cfg.data.transform.rot:
            t = np.random.rand() * 2 * np.pi
            rot_m = rotz(t)
            m = np.matmul(m, rot_m)  # rotation around z
            if instance_bboxes is not None:
                instance_bboxes[:, :6] = rotate_aligned_boxes_along_axis(instance_bboxes, rot_m, "z")
        if return_mat:
            return np.matmul(xyz, m), instance_bboxes, m
        else:
            return np.matmul(xyz, m), instance_bboxes

    def _croppedInstanceIds(self, instance_ids, valid_idxs):
        """
        Postprocess instance_ids after cropping
        """
        instance_ids = instance_ids[valid_idxs]
        j = 0
        while (j < instance_ids.max()):
            if (len(np.where(instance_ids == j)[0]) == 0):
                instance_ids[instance_ids == instance_ids.max()] = j
            j += 1
        return instance_ids

    def _getInstanceInfo(self, xyz, instance_ids, sem_labels=None):
        """
        :param xyz: (n, 3)
        :param instance_ids: (n), int, (0~nInst-1, -1)
        :return: num_instance, dict
        """
        instance_num_point = []  # (nInst), int
        unique_instance_ids = np.unique(instance_ids)
        num_instance = len(unique_instance_ids) - 1 if -1 in unique_instance_ids else len(unique_instance_ids)
        instance_info = np.zeros(
            (xyz.shape[0], 12), dtype=np.float32
        )  # (n, 12), float, (meanx, meany, meanz, cx, cy, cz, minx, miny, minz, maxx, maxy, maxz)
        
        if self.requires_bbox:
            assert sem_labels is not None, "sem_labels are not provided"
            max_num_instance = self.cfg.data.max_num_instance # NOTE this should always be the same number
            instance_bboxes = np.zeros((max_num_instance, 6))
            instance_bboxes_semcls = np.zeros((max_num_instance))
            instance_bbox_ids = np.zeros((max_num_instance))
            angle_classes = np.zeros((max_num_instance,))
            angle_residuals = np.zeros((max_num_instance,))
            size_classes = np.zeros((max_num_instance,))
            size_residuals = np.zeros((max_num_instance, 3))
            bbox_label = np.zeros((max_num_instance))

        for k, i_ in enumerate(unique_instance_ids, -1):
            if i_ < 0: continue
            
            inst_i_idx = np.where(instance_ids == i_)

            ### instance_info
            xyz_i = xyz[inst_i_idx]
            min_xyz_i = xyz_i.min(0)
            max_xyz_i = xyz_i.max(0)
            mean_xyz_i = xyz_i.mean(0)
            c_xyz_i = (max_xyz_i + min_xyz_i) / 2
            instance_info_i = instance_info[inst_i_idx]
            instance_info_i[:, 0:3] = mean_xyz_i
            instance_info_i[:, 3:6] = c_xyz_i
            instance_info_i[:, 6:9] = min_xyz_i
            instance_info_i[:, 9:12] = max_xyz_i
            instance_info[inst_i_idx] = instance_info_i

            ### instance_num_point
            instance_num_point.append(inst_i_idx[0].size)
            
            if self.requires_bbox:
                if k >= 128: continue
                instance_bboxes[k, :3] = c_xyz_i
                instance_bboxes[k, 3:] = max_xyz_i - min_xyz_i
                sem_cls = sem_labels[inst_i_idx][0]
                sem_cls = sem_cls - 2 if sem_cls >=  2 else 17
                instance_bboxes_semcls[k] = sem_cls
                instance_bbox_ids[k] = i_
                size_classes[k] = sem_cls
                size_residuals[k, :] = instance_bboxes[k, 3:] - self.DC.mean_size_arr[int(sem_cls),:]
                bbox_label[k] = 1
                
        if self.requires_bbox:
            return num_instance, instance_info, instance_num_point, instance_bboxes, instance_bboxes_semcls, instance_bbox_ids, angle_classes, angle_residuals, size_classes, size_residuals, bbox_label
        else:
            return num_instance, instance_info, instance_num_point

    def _get_coord_and_feat_from_mesh(self, mesh_data, scene_id):
        data = mesh_data[:, :3]

        if self.use_color:
            colors = mesh_data[:, 3:6]
            data = np.concatenate([data, colors], 1)

        if self.use_normal:
            assert mesh_data.shape[1] == 9 # make sure xyz, rgb and normals are included
            normals = mesh_data[:, 6:9]
            data = np.concatenate([data, normals], 1)
        
        if self.use_multiview:
            # load multiview database
            pid = mp.current_process().pid
            if pid not in self.multiview_data:
                self.multiview_data[pid] = h5py.File(self.cfg.SCANNETV2_PATH.multiview_features, "r", libver="latest")

            try:
                multiview = self.multiview_data[pid][scene_id][()]
            except KeyError:
                multiview = np.zeros((data.shape[0], 128)) # placeholder

            data = np.concatenate([data, multiview], 1)

        coords = data[:, :3]
        feats = data[:, 3:]

        return coords, feats

    def _generate_gt_clusters(self, points, instance_ids):
        gt_proposals_idx = []
        gt_proposals_offset = [0]
        unique_instance_ids = np.unique(instance_ids)
        num_instance = len(unique_instance_ids) - 1 if -1 in unique_instance_ids else len(unique_instance_ids)
        instance_bboxes = np.zeros((num_instance, 6))
        
        object_ids = []
        for cid, i_ in enumerate(unique_instance_ids, -1):
            if i_ < 0: continue
            
            object_ids.append(i_)
            inst_i_idx = np.where(instance_ids == i_)[0]
            inst_i_points = points[inst_i_idx]
            xmin = np.min(inst_i_points[:, 0])
            ymin = np.min(inst_i_points[:, 1])
            zmin = np.min(inst_i_points[:, 2])
            xmax = np.max(inst_i_points[:, 0])
            ymax = np.max(inst_i_points[:, 1])
            zmax = np.max(inst_i_points[:, 2])
            bbox = np.array([(xmin+xmax)/2, (ymin+ymax)/2, (zmin+zmax)/2, xmax-xmin, ymax-ymin, zmax-zmin]) 
            instance_bboxes[cid, :] = bbox
            
            proposals_idx_i = np.vstack((np.ones(len(inst_i_idx)) * cid, inst_i_idx)).transpose().astype(np.int32)
            gt_proposals_idx.append(proposals_idx_i)
            gt_proposals_offset.append(len(inst_i_idx) + gt_proposals_offset[-1])
            
        gt_proposals_idx = np.concatenate(gt_proposals_idx, axis=0)
        gt_proposals_offset = np.array(gt_proposals_offset).astype(np.int32)
        
        return gt_proposals_idx, gt_proposals_offset, object_ids, instance_bboxes

    def _conver_corners_to_cwdh(self, corners):
        coord_min = np.min(corners, axis=1) # num_bboxes, 3
        coord_max = np.max(corners, axis=1) # num_bboxes, 3
        coord_mean = (coord_max + coord_min) / 2

        bbox_cwhd = np.concatenate([coord_mean, coord_max - coord_min], axis=1)

        return bbox_cwhd

    def _get_bbox_centers(self, corners):
        coord_min = np.min(corners, axis=1) # num_bboxes, 3
        coord_max = np.max(corners, axis=1) # num_bboxes, 3

        return (coord_min + coord_max) / 2

    def _get_feature(self, scene_id, num_epochs=200):
        pid = mp.current_process().pid
        if pid not in self.gt_feature_data:
            database_path = os.path.join(self.cfg.SCANNETV2_PATH.gt_features, "{}.hdf5".format(self.split))
            self.gt_feature_data[pid] = h5py.File(database_path, "r", libver="latest")

        # pick out the features for the train split for a random epoch
        # the epoch pointer is always 0 in the eval mode for train split
        # this doesn"t apply to val split
        if self.split == "train":
            epoch_id = random.choice(range(num_epochs))
        else:
            epoch_id = 0

        # load object bounding box information
        gt_object_ids = self.gt_feature_data[pid]["{}|{}_gt_ids".format(epoch_id, scene_id)]

        # # load object bounding box information
        gt_corners = self.gt_feature_data[pid]["{}|{}_gt_corners".format(epoch_id, scene_id)]
        gt_centers = self._get_bbox_centers(gt_corners)

        # load object features
        gt_features = self.gt_feature_data[pid]["{}|{}_features".format(epoch_id, scene_id)]

        gt_sems = [self.objectid2label[scene_id][o_id] for o_id in gt_object_ids]

        return np.array(gt_object_ids), np.array(gt_features), np.array(gt_corners), np.array(gt_centers), np.array(gt_sems)

    def _nn_distance(self, pc1, pc2):
        N = pc1.shape[0]
        M = pc2.shape[0]
        pc1_expand_tile = pc1[:, np.newaxis]
        pc2_expand_tile = pc2[np.newaxis, :]
        pc_diff = pc1_expand_tile - pc2_expand_tile

        pc_dist = np.sum(pc_diff**2, axis=-1) # (N,M)
        idx1 = np.argmin(pc_dist, axis=1) # (N)
        idx2 = np.argmin(pc_dist, axis=0) # (M)

        return idx1, idx2

def scannet_collate_fn(batch):
    batch_size = batch.__len__()
    data = {}
    for key in batch[0].keys():
        if key in ['locs', 'locs_scaled', 'feats', 'sem_labels', 'instance_ids', 'num_instance', 'instance_info', 'instance_num_point', 'gt_proposals_idx', 'gt_proposals_offset']:
            continue
        if isinstance(batch[0][key], tuple):
            coords, feats = list(zip(*[sample[key] for sample in batch]))
            coords_b = batched_coordinates(coords)
            feats_b = torch.from_numpy(np.concatenate(feats, 0)).float()
            data[key] = (coords_b, feats_b)
        elif isinstance(batch[0][key], np.ndarray):
            data[key] = torch.stack(
                [torch.from_numpy(sample[key]) for sample in batch],
                axis=0)
        elif isinstance(batch[0][key], torch.Tensor):
            data[key] = torch.stack([sample[key] for sample in batch],
                                        axis=0)
        elif isinstance(batch[0][key], dict):
            data[key] = sparse_collate_fn(
                [sample[key] for sample in batch])
        else:
            data[key] = [sample[key] for sample in batch]
    return data

def sparse_collate_fn(batch):
    data = scannet_collate_fn(batch)
    # print("raw data loaded.")

    if "locs" in batch[0]:
        locs = []
        locs_scaled = []
        feats = []
        sem_labels = []
        instance_ids = []
        instance_info = []  # (N, 12)
        instance_num_point = []  # (total_nInst), int
        batch_offsets = [0]
        instance_offsets = [0]
        total_num_inst = 0
        total_points = 0

        gt_proposals_idx = []
        gt_proposals_offset = []
    
        for i, b in enumerate(batch):
            locs.append(torch.from_numpy(b["locs"]))
            locs_scaled.append(
                torch.cat([
                    torch.LongTensor(b["locs_scaled"].shape[0], 1).fill_(i),
                    torch.from_numpy(b["locs_scaled"]).long()
                ], 1))
            
            feats.append(torch.from_numpy(b["feats"]))
            batch_offsets.append(batch_offsets[-1] + b["locs_scaled"].shape[0])

            if "gt_proposals_idx" in b:
                gt_proposals_idx_i = b["gt_proposals_idx"]
                gt_proposals_idx_i[:, 0] += total_num_inst
                gt_proposals_idx_i[:, 1] += total_points
                gt_proposals_idx.append(torch.from_numpy(b["gt_proposals_idx"]))
                if gt_proposals_offset != []:
                    gt_proposals_offset_i = b["gt_proposals_offset"]
                    gt_proposals_offset_i += gt_proposals_offset[-1][-1].item()
                    gt_proposals_offset.append(torch.from_numpy(gt_proposals_offset_i[1:]))
                else:
                    gt_proposals_offset.append(torch.from_numpy(b["gt_proposals_offset"]))
            
            if "instance_ids" in b:
                instance_ids_i = b["instance_ids"]
                instance_ids_i[np.where(instance_ids_i != -1)] += total_num_inst
                total_num_inst += b["num_instance"].item()
                total_points += len(instance_ids_i)
                instance_ids.append(torch.from_numpy(instance_ids_i))
                
                sem_labels.append(torch.from_numpy(b["sem_labels"]))

                instance_info.append(torch.from_numpy(b["instance_info"]))
                instance_num_point.append(torch.from_numpy(b["instance_num_point"]))
                instance_offsets.append(instance_offsets[-1] + b["num_instance"].item())

        data["locs"] = torch.cat(locs, 0).to(torch.float32)  # float (N, 3)
        data["locs_scaled"] = torch.cat(locs_scaled, 0)  # long (N, 1 + 3), the batch item idx is put in locs[:, 0]
        data["feats"] = torch.cat(feats, 0)  #.to(torch.float32)            # float (N, C)
        data["batch_offsets"] = torch.tensor(batch_offsets, dtype=torch.int)  # int (B+1)
        
        if len(instance_ids) > 0:
            data["sem_labels"] = torch.cat(sem_labels, 0).long()  # long (N,)
            data["instance_ids"] = torch.cat(instance_ids, 0).long()  # long, (N,)
            data["instance_info"] = torch.cat(instance_info, 0).to(torch.float32)  # float (total_nInst, 12)
            data["instance_num_point"] = torch.cat(instance_num_point, 0).int()  # (total_nInst)
            data["instance_offsets"] = torch.tensor(instance_offsets, dtype=torch.int)  # int (B+1)

        if len(gt_proposals_idx) > 0:
            data["gt_proposals_idx"] = torch.cat(gt_proposals_idx, 0).to(torch.int32)
            data["gt_proposals_offset"] = torch.cat(gt_proposals_offset, 0).to(torch.int32)

        # print("raw data collated.")

        ### voxelize
        data["voxel_locs"], data["p2v_map"], data["v2p_map"] = pointgroup_ops.voxelization_idx(data["locs_scaled"], len(batch), 4) # mode=4
        # print("raw data voxelized.")

    return data