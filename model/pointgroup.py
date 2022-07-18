import os
import re
import torch
import functools
import random

import numpy as np
import torch.nn as nn
import MinkowskiEngine as ME
import pytorch_lightning as pl

from tqdm import tqdm
from glob import glob
from importlib import import_module

from data.scannet.model_util_scannet import ScannetDatasetConfig
from lib.pointgroup_ops.functions import pointgroup_ops
from lib.utils.eval import get_nms_instances
from lib.utils.bbox import get_3d_box_batch, get_aabb3d_iou_batch, get_3d_box
from lib.utils.nn_distance import nn_distance

from model.common import ResidualBlock, VGGBlock, UBlock


class PointGroup(pl.LightningModule):
    def __init__(self, cfg):
        super().__init__()

        self.cfg = cfg

        self.save_hyperparameters()

        self.DC = ScannetDatasetConfig(cfg)

        self.task = cfg.general.task
        self.total_epoch = cfg.train.epochs

        in_channel = cfg.model.use_color * 3 + cfg.model.use_normal * 3 + cfg.model.use_coords * 3 + cfg.model.use_multiview * 128
        m = cfg.model.m
        D = 3
        classes = cfg.data.classes
        blocks = cfg.model.blocks
        cluster_blocks = cfg.model.cluster_blocks
        block_reps = cfg.model.block_reps
        block_residual = cfg.model.block_residual
        
        self.requires_gt_mask = cfg.data.requires_gt_mask

        self.cluster_radius = cfg.cluster.cluster_radius
        self.cluster_meanActive = cfg.cluster.cluster_meanActive
        self.cluster_shift_meanActive = cfg.cluster.cluster_shift_meanActive
        self.cluster_npoint_thre = cfg.cluster.cluster_npoint_thre
        self.freeze_backbone = cfg.cluster.freeze_backbone

        self.score_scale = cfg.train.score_scale
        self.score_fullscale = cfg.train.score_fullscale
        self.mode = cfg.train.score_mode

        self.prepare_epochs = cfg.cluster.prepare_epochs

        if block_residual:
            block = ResidualBlock
        else:
            block = VGGBlock
        sp_norm = functools.partial(ME.MinkowskiBatchNorm, eps=1e-4, momentum=0.1)
        norm = functools.partial(nn.BatchNorm1d, eps=1e-4, momentum=0.1)

        #### backbone
        self.backbone = nn.Sequential(
            ME.MinkowskiConvolution(in_channel, m, kernel_size=3, bias=False, dimension=D),
            UBlock([m*c for c in blocks], sp_norm, block_reps, block),
            sp_norm(m),
            ME.MinkowskiReLU(inplace=True)
        )

        #### semantic segmentation
        self.sem_seg = nn.Linear(m, classes) # bias(default): True

        #### offset
        self.offset_net = nn.Sequential(
            nn.Linear(m, m),
            norm(m),
            nn.ReLU(inplace=True),
            nn.Linear(m, 3)
        )

        #### score
        self.score_net = nn.Sequential(
            UBlock([m*c for c in cluster_blocks], sp_norm, 2, block),
            sp_norm(m),
            ME.MinkowskiReLU(inplace=True)
        )
        
        if cfg.model.pred_bbox:
            num_class = cfg.model.num_bbox_class
            num_heading_bin = cfg.model.num_heading_bin
            num_size_cluster = cfg.model.num_size_cluster
            self.bbox_regressor = nn.Sequential(
                nn.Linear(m, m, bias=False),
                nn.BatchNorm1d(m),
                nn.ReLU(inplace=True),
                nn.Linear(m, m, bias=False),
                nn.BatchNorm1d(m),
                nn.ReLU(inplace=True),
                nn.Linear(m, 3+num_heading_bin*2+num_size_cluster*4+num_class)
            )
        
        self.score_linear = nn.Linear(m, 1)
        
    
    @staticmethod
    def get_batch_offsets(batch_idxs, batch_size):
        """
        :param batch_idxs: (N), int
        :param batch_size: int
        :return: batch_offsets: (batch_size + 1)
        """
        batch_offsets = torch.zeros(batch_size + 1).int().cuda()
        for i in range(batch_size):
            batch_offsets[i + 1] = batch_offsets[i] + (batch_idxs == i).sum()
        assert batch_offsets[-1] == batch_idxs.shape[0]
        return batch_offsets


    def clusters_voxelization(self, clusters_idx, clusters_offset, feats, coords, fullscale, scale, mode):
        """
        :param clusters_idx: (SumNPoint, 2), int, dim 0 for cluster_id, dim 1 for corresponding point idxs in N, cpu
        :param clusters_offset: (nCluster + 1), int, cpu
        :param feats: (N, C), float, cuda
        :param coords: (N, 3), float, cuda
        :return:
        """
        c_idxs = clusters_idx[:, 1].cuda()
        clusters_feats = feats[c_idxs.long()]
        clusters_coords = coords[c_idxs.long()]

        clusters_coords_mean = pointgroup_ops.sec_mean(clusters_coords, clusters_offset.cuda())  # (nCluster, 3), float
        clusters_coords_mean_all = torch.index_select(clusters_coords_mean, 0, clusters_idx[:, 0].cuda().long())  # (sumNPoint, 3), float
        clusters_coords -= clusters_coords_mean_all

        clusters_coords_min = pointgroup_ops.sec_min(clusters_coords, clusters_offset.cuda())  # (nCluster, 3), float
        clusters_coords_max = pointgroup_ops.sec_max(clusters_coords, clusters_offset.cuda())  # (nCluster, 3), float

        #### extract bbox info for each cluster
        clusters_size = clusters_coords_max - clusters_coords_min # (nCluster, 3), float
        clusters_center = (clusters_coords_max + clusters_coords_min) / 2 + clusters_coords_mean # (nCluster, 3), float
        ####

        #### make sure the the range of scaled clusters are at most fullscale
        clusters_scale = 1 / ((clusters_coords_max - clusters_coords_min) / fullscale).max(1)[0] - 0.01  # (nCluster), float
        clusters_scale = torch.clamp(clusters_scale, min=None, max=scale)

        min_xyz = clusters_coords_min * clusters_scale.unsqueeze(-1)  # (nCluster, 3), float
        max_xyz = clusters_coords_max * clusters_scale.unsqueeze(-1)

        clusters_scale = torch.index_select(clusters_scale, 0, clusters_idx[:, 0].cuda().long())

        clusters_coords = clusters_coords * clusters_scale.unsqueeze(-1)

        range = max_xyz - min_xyz
        offset = - min_xyz + torch.clamp(fullscale - range - 0.001, min=0) * torch.rand(3).cuda() + torch.clamp(fullscale - range + 0.001, max=0) * torch.rand(3).cuda()
        offset = torch.index_select(offset, 0, clusters_idx[:, 0].cuda().long())
        clusters_coords += offset
        assert clusters_coords.shape.numel() == ((clusters_coords >= 0) * (clusters_coords < fullscale)).sum()

        clusters_coords = clusters_coords.long()
        clusters_coords = torch.cat([clusters_idx[:, 0].view(-1, 1).long(), clusters_coords.cpu()], 1)  # (sumNPoint, 1 + 3)

        clusters_voxel_coords, clusters_p2v_map, clusters_v2p_map = pointgroup_ops.voxelization_idx(clusters_coords, int(clusters_idx[-1, 0]) + 1, mode)
        # clusters_voxel_coords: M * (1 + 3) long
        # clusters_p2v_map: sumNPoint int, in M
        # clusters_v2p_map: M * (maxActive + 1) int, in N

        clusters_voxel_feats = pointgroup_ops.voxelization(clusters_feats, clusters_v2p_map.cuda(), mode)  # (M, C), float, cuda

        clusters_voxel_feats = ME.SparseTensor(features=clusters_voxel_feats, coordinates=clusters_voxel_coords.int().cuda())

        return clusters_voxel_feats, clusters_p2v_map, (clusters_center, clusters_size)


    def decode_bbox_prediction(self, encoded_bbox, data_dict):
        """
        decode the predicted parameters for the bounding boxes
        """
        num_heading_bin = self.cfg.model.num_heading_bin
        num_size_cluster = self.cfg.model.num_size_cluster
        # encoded_bbox = encoded_bbox.transpose(2,1).contiguous() # (batch_size, 1024, ..)
        num_proposal = encoded_bbox.shape[0]

        # objectness_scores = encoded_bbox[:,:,0:2]

        base_xyz = data_dict["proposal_info"][0] # (num_proposal, 3)
        center = base_xyz + encoded_bbox[:, :3] # (num_proposal, 3)

        heading_scores = encoded_bbox[:, 3:3+num_heading_bin] # (num_proposal, 1)
        heading_residuals_normalized = encoded_bbox[:, 3+num_heading_bin:3+num_heading_bin*2] # (num_proposal, 1)
        
        size_scores = encoded_bbox[:, 3+num_heading_bin*2:3+num_heading_bin*2+num_size_cluster]
        size_residuals_normalized = encoded_bbox[:, 3+num_heading_bin*2+num_size_cluster:3+num_heading_bin*2+num_size_cluster*4].view([num_proposal, num_size_cluster, 3]) # (num_proposal, num_size_cluster, 3)
        
        sem_cls_scores = encoded_bbox[:, 3+num_heading_bin*2+num_size_cluster*4:] # num_proposalx18

        # store
        # data_dict["objectness_scores"] = objectness_scores
        data_dict["center"] = center
        data_dict["heading_scores"] = heading_scores # Bxnum_proposalxnum_heading_bin
        data_dict["heading_residuals_normalized"] = heading_residuals_normalized # Bxnum_proposalxnum_heading_bin (should be -1 to 1)
        data_dict["heading_residuals"] = heading_residuals_normalized * (np.pi/num_heading_bin) # (num_proposal, num_heading_bin)
        data_dict["size_scores"] = size_scores
        data_dict["size_residuals_normalized"] = size_residuals_normalized
        data_dict["size_residuals"] = size_residuals_normalized * torch.from_numpy(self.DC.mean_size_arr.astype(np.float32)).cuda().unsqueeze(0)
        data_dict["sem_cls_scores"] = sem_cls_scores

        return data_dict
    
    def get_object_assignments(self, data_dict):
        _, ind1, _, _ = nn_distance(data_dict["proposal_center_batched"], data_dict["center_label"], l1=True)
        
        data_dict["object_assignment"] = ind1

        return data_dict
    
    def convert_stack_to_batch(self, data_dict):
        batch_size = len(data_dict["batch_offsets"]) - 1
        max_num_proposal = self.cfg.model.max_num_proposal
        data_dict["proposal_feats_batched"] = torch.zeros(batch_size, max_num_proposal, self.cfg.model.m).type_as(data_dict["proposal_feats"])
        data_dict["proposal_bbox_batched"] = torch.zeros(batch_size, max_num_proposal, 8, 3).type_as(data_dict["proposal_feats"])
        data_dict["proposal_center_batched"] = torch.zeros(batch_size, max_num_proposal, 3).type_as(data_dict["proposal_feats"])
        data_dict["proposal_sem_cls_batched"] = torch.zeros(batch_size, max_num_proposal).type_as(data_dict["proposal_feats"])
        data_dict["proposal_scores_batched"] = torch.zeros(batch_size, max_num_proposal).type_as(data_dict["proposal_feats"])
        data_dict["proposal_batch_mask"] = torch.zeros(batch_size, max_num_proposal).type_as(data_dict["proposal_feats"])

        proposal_bbox = data_dict["proposal_crop_bbox"].detach().cpu().numpy()
        proposal_bbox = get_3d_box_batch(proposal_bbox[:, :3], proposal_bbox[:, 3:6], proposal_bbox[:, 6]) # (nProposals, 8, 3)
        proposal_bbox_tensor = torch.tensor(proposal_bbox).type_as(data_dict["proposal_feats"])

        for b in range(batch_size):
            proposal_batch_idx = torch.nonzero(data_dict["proposals_batchId"] == b).squeeze(-1)
            pred_num = len(proposal_batch_idx)
            pred_num = pred_num if pred_num < max_num_proposal else max_num_proposal
            
            # NOTE proposals should be truncated if more than max_num_proposal proposals are predicted
            data_dict["proposal_feats_batched"][b, :pred_num, :] = data_dict["proposal_feats"][proposal_batch_idx][:pred_num]
            data_dict["proposal_bbox_batched"][b, :pred_num, :, :] = proposal_bbox_tensor[proposal_batch_idx][:pred_num]
            data_dict["proposal_center_batched"][b, :pred_num, :] = data_dict["proposal_crop_bbox"][proposal_batch_idx, :3][:pred_num]
            data_dict["proposal_sem_cls_batched"][b, :pred_num] = data_dict["proposal_crop_bbox"][proposal_batch_idx, 7][:pred_num]
            data_dict["proposal_scores_batched"][b, :pred_num] = data_dict["proposal_objectness_scores"][proposal_batch_idx][:pred_num]
            data_dict["proposal_batch_mask"][b, :pred_num] = 1

            # NOTE all proposal data in batch should be shuffled to prevent overfitting
            rearrange_ids = torch.randperm(max_num_proposal)
            data_dict["proposal_feats_batched"][b] = data_dict["proposal_feats_batched"][b][rearrange_ids]
            data_dict["proposal_bbox_batched"][b] = data_dict["proposal_bbox_batched"][b][rearrange_ids]
            data_dict["proposal_center_batched"][b] = data_dict["proposal_center_batched"][b][rearrange_ids]
            data_dict["proposal_sem_cls_batched"][b] = data_dict["proposal_sem_cls_batched"][b][rearrange_ids]
            data_dict["proposal_scores_batched"][b] = data_dict["proposal_scores_batched"][b][rearrange_ids]
            data_dict["proposal_batch_mask"][b] = data_dict["proposal_batch_mask"][b][rearrange_ids]

        # compute GT object assignments
        if self.cfg.general.task != "test": 
            data_dict = self.get_object_assignments(data_dict)

        return data_dict


    def forward(self, data_dict):
        batch_size = len(data_dict["batch_offsets"]) - 1
        x = ME.SparseTensor(features=data_dict["voxel_feats"], coordinates=data_dict["voxel_locs"].int())

        #### backbone
        out = self.backbone(x)
        pt_feats = out.features[data_dict["p2v_map"].long()] # (N, m)

        #### semantic segmentation
        semantic_scores = self.sem_seg(pt_feats)   # (N, nClass), float
        semantic_preds = semantic_scores.max(1)[1]    # (N), long, {0, 1, ..., classes}
        data_dict["semantic_scores"] = semantic_scores

        #### offsets
        pt_offsets = self.offset_net(pt_feats) # (N, 3), float32
        data_dict["pt_offsets"] = pt_offsets

        if data_dict["epoch"] > self.prepare_epochs or self.freeze_backbone:
            #### get prooposal clusters
            batch_idxs = data_dict["locs_scaled"][:, 0].int()
            
            if not self.requires_gt_mask:
                object_idxs = torch.nonzero(semantic_preds > 0, as_tuple=False).view(-1)
                batch_idxs_ = batch_idxs[object_idxs]
                batch_offsets_ = self.get_batch_offsets(batch_idxs_, batch_size)
                coords_ = data_dict["locs"][object_idxs]
                pt_offsets_ = pt_offsets[object_idxs]

                semantic_preds_cpu = semantic_preds[object_idxs].int().cpu()

                idx_shift, start_len_shift = pointgroup_ops.ballquery_batch_p(coords_ + pt_offsets_, batch_idxs_, batch_offsets_, self.cluster_radius, self.cluster_shift_meanActive)
                proposals_idx_shift, proposals_offset_shift = pointgroup_ops.bfs_cluster(semantic_preds_cpu, idx_shift.cpu(), start_len_shift.cpu(), self.cluster_npoint_thre)
                proposals_idx_shift[:, 1] = object_idxs[proposals_idx_shift[:, 1].long()].int()
                proposals_batchId_shift_all = batch_idxs[proposals_idx_shift[:, 1].long()].int()
                # proposals_idx_shift: (sumNPoint, 2), int, dim 0 for cluster_id, dim 1 for corresponding point idxs in N
                # proposals_offset_shift: (nProposal + 1), int
                # proposals_batchId_shift_all: (sumNPoint,) batch id

                idx, start_len = pointgroup_ops.ballquery_batch_p(coords_, batch_idxs_, batch_offsets_, self.cluster_radius, self.cluster_meanActive)
                proposals_idx, proposals_offset = pointgroup_ops.bfs_cluster(semantic_preds_cpu, idx.cpu(), start_len.cpu(), self.cluster_npoint_thre)
                proposals_idx[:, 1] = object_idxs[proposals_idx[:, 1].long()].int()
                proposals_batchId_all = batch_idxs[proposals_idx[:, 1].long()].int()
                # proposals_idx: (sumNPoint, 2), int, dim 0 for cluster_id, dim 1 for corresponding point idxs in N
                # proposals_offset: (nProposal + 1), int
                # proposals_batchId_all: (sumNPoint,) batch id

                proposals_idx_shift[:, 0] += (proposals_offset.size(0) - 1)
                proposals_offset_shift += proposals_offset[-1]
                proposals_idx = torch.cat((proposals_idx, proposals_idx_shift), dim=0)
                proposals_offset = torch.cat((proposals_offset, proposals_offset_shift[1:]))
                proposals_batchId_all = torch.cat((proposals_batchId_all, proposals_batchId_shift_all[1:])) # (sumNPoint,)
                # proposals_idx = proposals_idx_shift
                # proposals_offset = proposals_offset_shift
                # proposals_batchId_all = proposals_batchId_shift_all
            else:
                proposals_idx = data_dict["gt_proposals_idx"].cpu()
                proposals_offset = data_dict["gt_proposals_offset"].cpu()
                proposals_batchId_all = batch_idxs[proposals_idx[:, 1].long()].int() # (sumNPoint,)

            #### proposals voxelization again
            proposals_voxel_feats, proposals_p2v_map, (proposals_center, proposals_size) = self.clusters_voxelization(proposals_idx, proposals_offset, pt_feats, data_dict["locs"], self.score_fullscale, self.score_scale, self.mode)
            # proposals_voxel_feats: (M, C) M: voxels
            # proposals_p2v_map: point2voxel map (sumNPoint,)
            # proposals_center / proposals_size: (nProposals, 3)

            #### score
            score_feats = self.score_net(proposals_voxel_feats)
            pt_score_feats = score_feats.features[proposals_p2v_map.long()] # (sumNPoint, C)
            proposals_score_feats = pointgroup_ops.roipool(pt_score_feats, proposals_offset.cuda())  # (nProposal, C)
            # proposals_score_feats = self.proposal_mlp(proposals_score_feats) # (nProposal, 128)
            scores = self.score_linear(proposals_score_feats)  # (nProposal, 1)
            data_dict["proposal_scores"] = (scores, proposals_idx, proposals_offset)

            ############ extract batch related features and bbox #############
            num_proposals = proposals_offset.shape[0] - 1
            
            proposals_npoint = torch.zeros(num_proposals).cuda()
            for i in range(num_proposals):
                proposals_npoint[i] = (proposals_idx[:, 0] == i).sum()
            thres_mask = torch.logical_and(torch.sigmoid(scores.view(-1)) > self.cfg.test.TEST_SCORE_THRESH, proposals_npoint > self.cfg.test.TEST_NPOINT_THRESH) # (nProposal,)
            data_dict["proposals_npoint"] = proposals_npoint
            data_dict["proposal_thres_mask"] = thres_mask
            
            proposals_batchId = proposals_batchId_all[proposals_offset[:-1].long()] # (nProposal,)
            proposals_batchId = proposals_batchId[thres_mask]
            data_dict["proposals_batchId"] = proposals_batchId # (nProposal,)
            data_dict["proposal_feats"] = proposals_score_feats[thres_mask]
            data_dict["proposal_objectness_scores"] = torch.sigmoid(scores.view(-1))[thres_mask]
            
            if self.cfg.model.crop_bbox:
                proposal_crop_bbox = torch.zeros(num_proposals, 9).cuda() # (nProposals, center+size+heading+label)
                proposal_crop_bbox[:, :3] = proposals_center
                proposal_crop_bbox[:, 3:6] = proposals_size
                proposal_crop_bbox[:, 7] = semantic_preds[proposals_idx[proposals_offset[:-1].long(), 1].long()]
                proposal_crop_bbox[:, 8] = torch.sigmoid(scores.view(-1))
                proposal_crop_bbox = proposal_crop_bbox[thres_mask]
                data_dict["proposal_crop_bbox"] = proposal_crop_bbox

            if self.cfg.model.pred_bbox:
                encoded_pred_bbox = self.bbox_regressor(proposals_score_feats) # (nProposal, 3+num_heading_bin*2+num_size_cluster*4+num_class)
                encoded_pred_bbox = encoded_pred_bbox[thres_mask]
                data_dict["proposal_info"] = (proposals_center[thres_mask], proposals_size[thres_mask])
                data_dict = self.decode_bbox_prediction(encoded_pred_bbox)
            
        return data_dict
    
    def configure_optimizers(self):
        print("=> configure optimizer...")

        optim_class_name = self.cfg.train.optim.classname
        optim = getattr(torch.optim, optim_class_name)
        if optim_class_name == "Adam":
            optimizer = optim(filter(lambda p: p.requires_grad, self.parameters()), lr=self.cfg.train.optim.lr)
        elif optim_class_name == "SGD":
            optimizer = optim(filter(lambda p: p.requires_grad, self.parameters()), lr=self.cfg.train.optim.lr, momentum=self.cfg.train.optim.momentum, weight_decay=self.cfg.train.optim.weight_decay)
        else:
            raise NotImplemented

        return [optimizer]
        
        
    def loss(self, data_dict, epoch):

        def get_segmented_scores(scores, fg_thresh=1.0, bg_thresh=0.0):
            """
            :param scores: (N), float, 0~1
            :return: segmented_scores: (N), float 0~1, >fg_thresh: 1, <bg_thresh: 0, mid: linear
            """
            fg_mask = scores > fg_thresh
            bg_mask = scores < bg_thresh
            interval_mask = (fg_mask == 0) & (bg_mask == 0)

            segmented_scores = (fg_mask > 0).float()
            k = 1 / (fg_thresh - bg_thresh)
            b = bg_thresh / (bg_thresh - fg_thresh)
            segmented_scores[interval_mask] = scores[interval_mask] * k + b

            return segmented_scores

        """semantic loss"""
        semantic_scores, semantic_labels = data_dict["semantic_scores"]
        # semantic_scores: (N, nClass), float32, cuda
        # semantic_labels: (N), long, cuda

        semantic_criterion = nn.CrossEntropyLoss(ignore_index=self.cfg.data.ignore_label)
        semantic_loss = semantic_criterion(semantic_scores, semantic_labels)
        data_dict["semantic_loss"] = (semantic_loss, semantic_scores.shape[0])

        """offset loss"""
        pt_offsets, coords, instance_info, instance_ids = data_dict["pt_offsets"]
        # pt_offsets: (N, 3), float, cuda
        # coords: (N, 3), float32
        # instance_info: (N, 12), float32 tensor (meanxyz, center, minxyz, maxxyz)
        # instance_ids: (N), long

        gt_offsets = instance_info[:, 0:3] - coords   # (N, 3)
        pt_diff = pt_offsets - gt_offsets   # (N, 3)
        pt_dist = torch.sum(torch.abs(pt_diff), dim=-1)   # (N)
        valid = (instance_ids != self.cfg.data.ignore_label).float()
        offset_norm_loss = torch.sum(pt_dist * valid) / (torch.sum(valid) + 1e-6)

        gt_offsets_norm = torch.norm(gt_offsets, p=2, dim=1)   # (N), float
        gt_offsets_ = gt_offsets / (gt_offsets_norm.unsqueeze(-1) + 1e-8)
        pt_offsets_norm = torch.norm(pt_offsets, p=2, dim=1)
        pt_offsets_ = pt_offsets / (pt_offsets_norm.unsqueeze(-1) + 1e-8)
        direction_diff = - (gt_offsets_ * pt_offsets_).sum(-1)   # (N)
        offset_dir_loss = torch.sum(direction_diff * valid) / (torch.sum(valid) + 1e-6)

        data_dict["offset_norm_loss"] = (offset_norm_loss, valid.sum())
        data_dict["offset_dir_loss"] = (offset_dir_loss, valid.sum())

        if (epoch > self.cfg.cluster.prepare_epochs):
            """score loss"""
            scores, proposals_idx, proposals_offset, instance_pointnum = data_dict["proposal_scores"]
            # scores: (nProposal, 1), float32
            # proposals_idx: (sumNPoint, 2), int, cpu, dim 0 for cluster_id, dim 1 for corresponding point idxs in N
            # proposals_offset: (nProposal + 1), int, cpu
            # instance_pointnum: (total_nInst), int

            ious = pointgroup_ops.get_iou(proposals_idx[:, 1].cuda(), proposals_offset.cuda(), instance_ids, instance_pointnum) # (nProposal, nInstance), float
            gt_ious, gt_instance_idxs = ious.max(1)  # (nProposal) float, long
            gt_scores = get_segmented_scores(gt_ious, self.cfg.train.fg_thresh, self.cfg.train.bg_thresh)

            # score_criterion = nn.BCELoss(reduction="none")
            # score_loss = score_criterion(torch.sigmoid(scores.view(-1)), gt_scores)
            score_criterion = nn.BCEWithLogitsLoss(reduction="none")
            score_loss = score_criterion(scores.view(-1), gt_scores)
            score_loss = score_loss.mean()

            data_dict["score_loss"] = (score_loss, gt_ious.shape[0])

        """total loss"""
        loss = self.cfg.train.loss_weight[0] * semantic_loss + self.cfg.train.loss_weight[1] * offset_norm_loss + self.cfg.train.loss_weight[2] * offset_dir_loss
        if(epoch > self.cfg.cluster.prepare_epochs):
            loss += (self.cfg.train.loss_weight[3] * score_loss)
        data_dict["total_loss"] = (loss, semantic_labels.shape[0])

        return data_dict
        
        
    def feed(self, data_dict, epoch=0):
        data_dict["epoch"] = epoch

        if self.cfg.model.use_coords:
            data_dict["feats"] = torch.cat((data_dict["feats"], data_dict["locs"]), 1)

        data_dict["voxel_feats"] = pointgroup_ops.voxelization(data_dict["feats"], data_dict["v2p_map"], self.cfg.data.mode)  # (M, C), float, cuda

        data_dict = self.forward(data_dict)

        if data_dict["epoch"] > self.prepare_epochs or self.freeze_backbone:
            data_dict = self.convert_stack_to_batch(data_dict)
        
        return data_dict

    def parse_feed_ret(self, data_dict, epoch=0):
        semantic_scores = data_dict["semantic_scores"] # (N, nClass) float32, cuda
        pt_offsets = data_dict["pt_offsets"]           # (N, 3), float32, cuda
        
        preds = {}        
        preds["semantic"] = semantic_scores
        preds["pt_offsets"] = pt_offsets
        if self.mode != "test":
            data_dict["semantic_scores"] = (semantic_scores, data_dict["sem_labels"])
            data_dict["pt_offsets"] = (pt_offsets, data_dict["locs"], data_dict["instance_info"], data_dict["instance_ids"])

        if epoch > self.cfg.cluster.prepare_epochs:
            scores, proposals_idx, proposals_offset = data_dict["proposal_scores"]
            preds["score"] = scores
            preds["proposals"] = (proposals_idx, proposals_offset)
            preds["proposal_crop_bboxes"] = data_dict["proposal_crop_bbox"]
            
            if self.mode != "test":
                data_dict["proposal_scores"] = (scores, proposals_idx, proposals_offset, data_dict["instance_num_point"])
                # scores: (nProposal, 1) float, cuda
                # proposals_idx: (sumNPoint, 2), int, cpu, dim 0 for cluster_id, dim 1 for corresponding point idxs in N
                # proposals_offset: (nProposal + 1), int, cpu
                data_dict["proposal_thres_mask"] = data_dict["proposal_thres_mask"]
                data_dict["proposals_batchId"] = data_dict["proposals_batchId"]
                if self.cfg.model.crop_bbox:
                    data_dict["proposal_crop_bboxes"] = data_dict["proposal_crop_bbox"]
                if self.cfg.model.pred_bbox:
                    data_dict["proposal_pred_bboxes"] = (data_dict["center"], data_dict["heading_scores"], data_dict["heading_residuals_normalized"], data_dict["heading_residuals"], data_dict["size_scores"], data_dict["size_residuals_normalized"], data_dict["size_residuals"], data_dict["sem_cls_scores"])
        
        return preds, data_dict

        
    def training_step(self, data_dict, idx):
        torch.cuda.empty_cache()

        ##### prepare input and forward
        data_dict = self.feed(data_dict, self.current_epoch)
        _, data_dict = self.parse_feed_ret(data_dict, self.current_epoch)
        data_dict = self.loss(data_dict, self.current_epoch)
        loss = data_dict["total_loss"][0]

        in_prog_bar = ["total_loss"]
        for key, value in data_dict.items():
            if "loss" in key:
                self.log("train/{}".format(key), value[0], prog_bar=key in in_prog_bar, on_step=True, on_epoch=True, sync_dist=True)

        return loss

    def validation_step(self, data_dict, idx):
        torch.cuda.empty_cache()

        ##### prepare input and forward
        data_dict = self.feed(data_dict, self.current_epoch)
        _, data_dict = self.parse_feed_ret(data_dict)
        data_dict = self.loss(data_dict, self.current_epoch)

        in_prog_bar = ["total_loss"]
        for key, value in data_dict.items():
            if "loss" in key:
                self.log("val/{}".format(key), value[0], prog_bar=key in in_prog_bar, on_step=False, on_epoch=True, sync_dist=True)

    ######### NOTE DANGER ZONE!!!
    def test(self, split="val"):
        from data.scannet.model_util_scannet import NYU20_CLASS_IDX
        NYU20_CLASS_IDX = NYU20_CLASS_IDX[1:] # for scannet temporarily
        self.mode = "test"
        self.current_epoch = self.start_epoch
        self.eval()
        
        dataloader = self.dataloader[split]
        print(">>>>>>>>>>>>>>>> Start Inference >>>>>>>>>>>>>>>>")

        with torch.no_grad():
            for i, batch in enumerate(tqdm(dataloader)):
                # if batch["scene_id"] < 20800: continue
                N = batch["feats"].shape[0]
                scene_name = self.dataset[split].scene_names[i]
                
                batch = self.feed(batch, self.current_epoch)
                preds, _ = self.parse_feed_ret(batch)
                
                ##### get predictions (#1 semantic_pred, pt_offsets; #2 scores, proposals_pred)
                semantic_scores = preds["semantic"]  # (N, nClass) float32, cuda, 0: unannotated
                semantic_pred_labels = semantic_scores.max(1)[1]  # (N) long, cuda
                semantic_class_idx = torch.tensor(NYU20_CLASS_IDX, dtype=torch.int).cuda() # (nClass)
                semantic_pred_class_idx = semantic_class_idx[semantic_pred_labels].cpu().numpy()
                
                if self.current_epoch > self.cfg.cluster.prepare_epochs:
                    proposals_score = torch.sigmoid(preds["score"].view(-1)) # (nProposal,) float, cuda
                    proposals_idx, proposals_offset = preds["proposals"]
                    # proposals_idx: (sumNPoint, 2), int, cpu, dim 0 for cluster_id, dim 1 for corresponding point idxs in N
                    # proposals_offset: (nProposal + 1), int, cpu

                    num_proposals = proposals_offset.shape[0] - 1
                    N = semantic_scores.shape[0]
                    
                    proposals_mask = torch.zeros((num_proposals, N), dtype=torch.int).cuda() # (nProposal, N), int, cuda
                    proposals_mask[proposals_idx[:, 0].long(), proposals_idx[:, 1].long()] = 1
                    
                    ##### score threshold & min_npoint mask
                    proposals_npoint = proposals_mask.sum(1)
                    proposals_thres_mask = torch.logical_and(proposals_score > self.cfg.test.TEST_SCORE_THRESH, proposals_npoint > self.cfg.test.TEST_NPOINT_THRESH)
                    
                    proposals_score = proposals_score[proposals_thres_mask]
                    proposals_mask = proposals_mask[proposals_thres_mask]
                    
                    ##### non_max_suppression
                    if proposals_score.shape[0] == 0:
                        pick_idxs = np.empty(0)
                    else:
                        proposals_mask_f = proposals_mask.float()  # (nProposal, N), float, cuda
                        intersection = torch.mm(proposals_mask_f, proposals_mask_f.t())  # (nProposal, nProposal), float, cuda
                        proposals_npoint = proposals_mask_f.sum(1)  # (nProposal), float, cuda
                        proposals_np_repeat_h = proposals_npoint.unsqueeze(-1).repeat(1, proposals_npoint.shape[0])
                        proposals_np_repeat_v = proposals_npoint.unsqueeze(0).repeat(proposals_npoint.shape[0], 1)
                        cross_ious = intersection / (proposals_np_repeat_h + proposals_np_repeat_v - intersection) # (nProposal, nProposal), float, cuda
                        pick_idxs = get_nms_instances(cross_ious.cpu().numpy(), proposals_score.cpu().numpy(), self.cfg.test.TEST_NMS_THRESH)  # int, (nCluster,)

                    clusters_mask = proposals_mask[pick_idxs].cpu().numpy() # int, (nCluster, N)
                    clusters_score = proposals_score[pick_idxs].cpu().numpy() # float, (nCluster,)
                    nclusters = clusters_mask.shape[0]
                
                pred_path = os.path.join(self.cfg.general.root, "split_pred", split)
                
                sem_pred_path = os.path.join(pred_path, "semantic")
                os.makedirs(sem_pred_path, exist_ok=True)
                sem_pred_file_path = os.path.join(sem_pred_path, f"{scene_name}.txt")
                np.savetxt(sem_pred_file_path, semantic_pred_class_idx, fmt="%d")
                
                if self.current_epoch > self.cfg.cluster.prepare_epochs:
                    inst_pred_path = os.path.join(pred_path, "instance")
                    inst_pred_masks_path = os.path.join(inst_pred_path, "predicted_masks")
                    os.makedirs(inst_pred_path, exist_ok=True)
                    os.makedirs(inst_pred_masks_path, exist_ok=True)
                    cluster_ids = np.ones(shape=(N)) * -1 # id starts from 0
                    with open(os.path.join(inst_pred_path, f"{scene_name}.txt"), "w") as f:
                        for c_id in range(nclusters):
                            cluster_i = clusters_mask[c_id]  # (N)
                            cluster_ids[cluster_i == 1] = c_id
                            assert np.unique(semantic_pred_class_idx[cluster_i == 1]).size == 1
                            cluster_i_class_idx = semantic_pred_class_idx[cluster_i == 1][0]
                            score = clusters_score[c_id]
                            f.write(f"predicted_masks/{scene_name}_{c_id:03d}.txt {cluster_i_class_idx} {score:.4f}\n")
                            np.savetxt(os.path.join(inst_pred_masks_path, f"{scene_name}_{c_id:03d}.txt"), cluster_i, fmt="%d")
                    np.savetxt(os.path.join(inst_pred_path, f"{scene_name}.cluster_ids.txt"), cluster_ids, fmt="%d")
