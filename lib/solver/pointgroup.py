import os, time
from tqdm import tqdm
import numpy as np

import torch
import torch.nn as nn
import MinkowskiEngine as ME

from data.scannet.model_util_scannet import ScannetDatasetConfig
from lib.solver.base_solver import BaseSolver
from lib.pointgroup_ops.functions import pointgroup_ops
from lib.utils.log import Meters
from lib.utils.solver import step_learning_rate
from lib.utils.eval import get_nms_instances
from lib.utils.bbox import get_3d_box_batch, get_aabb3d_iou_batch, get_3d_box


class PointGroupSolver(BaseSolver):

    def __init__(self, cfg):
        super(PointGroupSolver, self).__init__(cfg)
        self.DC = ScannetDatasetConfig(cfg)
        self._init_model()
        self._init_optim()
        self._init_criterion()
        if cfg.general.task != 'test':
            self._resume_from_checkpoint()
        else:
            self._load_pretrained_model()
        if cfg.data.requires_bbox:
            cfg.log.meter_names += cfg.log.bbox_names
            cfg.log.tb_names += cfg.log.bbox_names
        self.logger.store_backup_config()
        
        
    def _load_pretrained_module(self):
        self.logger.info(f'=> loading pretrained {self.cfg.model.pretrained_module}...')
        model_dict = self.model.state_dict()
        ckp = torch.load(self.cfg.model.pretrained_module_path)
        pretrained_module_dict = {k: v for k, v in ckp.items() if k.startswith(tuple(self.cfg.model.pretrained_module))}
        model_dict.update(pretrained_module_dict)
        self.model.load_state_dict(model_dict)
        
        if 'epoch' in ckp:
            self.start_epoch = ckp['epoch']
        else:
            self.start_epoch = 129
        

    def _init_criterion(self):
        self.semantic_criterion = nn.CrossEntropyLoss(ignore_index=self.cfg.data.ignore_label).cuda()
        self.score_criterion = nn.BCELoss(reduction='none').cuda()


    def _loss(self, loss_input, data_dict, epoch):

        def get_segmented_scores(scores, fg_thresh=1.0, bg_thresh=0.0):
            '''
            :param scores: (N), float, 0~1
            :return: segmented_scores: (N), float 0~1, >fg_thresh: 1, <bg_thresh: 0, mid: linear
            '''
            fg_mask = scores > fg_thresh
            bg_mask = scores < bg_thresh
            interval_mask = (fg_mask == 0) & (bg_mask == 0)

            segmented_scores = (fg_mask > 0).float()
            k = 1 / (fg_thresh - bg_thresh)
            b = bg_thresh / (bg_thresh - fg_thresh)
            segmented_scores[interval_mask] = scores[interval_mask] * k + b

            return segmented_scores

        loss_dict = {}

        '''semantic loss'''
        semantic_scores, semantic_labels = loss_input['semantic_scores']
        # semantic_scores: (N, nClass), float32, cuda
        # semantic_labels: (N), long, cuda

        semantic_loss = self.semantic_criterion(semantic_scores, semantic_labels)
        loss_dict['semantic_loss'] = (semantic_loss, semantic_scores.shape[0])

        '''offset loss'''
        pt_offsets, coords, instance_info, instance_ids = loss_input['pt_offsets']
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

        loss_dict['offset_norm_loss'] = (offset_norm_loss, valid.sum())
        loss_dict['offset_dir_loss'] = (offset_dir_loss, valid.sum())

        if (epoch > self.cfg.cluster.prepare_epochs):
            '''score loss'''
            scores, proposals_idx, proposals_offset, instance_pointnum = loss_input['proposal_scores']
            # scores: (nProposal, 1), float32
            # proposals_idx: (sumNPoint, 2), int, cpu, dim 0 for cluster_id, dim 1 for corresponding point idxs in N
            # proposals_offset: (nProposal + 1), int, cpu
            # instance_pointnum: (total_nInst), int

            ious = pointgroup_ops.get_iou(proposals_idx[:, 1].cuda(), proposals_offset.cuda(), instance_ids, instance_pointnum) # (nProposal, nInstance), float
            gt_ious, gt_instance_idxs = ious.max(1)  # (nProposal) float, long
            # thres_mask = loss_input['proposal_thres_mask']
            # loss_dict['mask_ious'] = (gt_ious[thres_mask].mean(), thres_mask.sum())
            ######## gt_ious HAS PROBLEMS !!!!!!
            gt_scores = get_segmented_scores(gt_ious, self.cfg.train.fg_thresh, self.cfg.train.bg_thresh)
            score_loss = self.score_criterion(torch.sigmoid(scores.view(-1)), gt_scores)
            score_loss = score_loss.mean()

            loss_dict['score_loss'] = (score_loss, gt_ious.shape[0])
            
            '''bbox loss'''
            if self.cfg.model.pred_bbox:
                from lib.utils.loss import compute_box_and_sem_cls_loss
                bbox_loss = compute_box_and_sem_cls_loss(loss_input, data_dict, loss_dict, self.DC.mean_size_arr)

        '''total loss'''
        loss = self.cfg.train.loss_weight[0] * semantic_loss + self.cfg.train.loss_weight[1] * offset_norm_loss + self.cfg.train.loss_weight[2] * offset_dir_loss
        if(epoch > self.cfg.cluster.prepare_epochs):
            loss += (self.cfg.train.loss_weight[3] * score_loss)
            if self.cfg.model.pred_bbox:
                loss += (self.cfg.train.loss_weight[4] * bbox_loss)
        loss_dict['total_loss'] = (loss, semantic_labels.shape[0])

        return loss_dict
    
    
    def get_bbox_iou(self, loss_input, data_dict, eval_dict):
        batch_size = len(data_dict["batch_offsets"]) - 1
        proposal_thres_mask = loss_input['proposal_thres_mask']
        instance_offsets = data_dict['instance_offsets'].detach().cpu().numpy()
        proposals_batchId = loss_input['proposals_batchId']
        
        # parse gt_bbox
        center_label = data_dict['center_label'].detach().cpu().numpy()
        heading_class_label = data_dict['heading_class_label'].detach().cpu().numpy()
        heading_residual_label = data_dict['heading_residual_label'].detach().cpu().numpy()
        size_class_label = data_dict['size_class_label'].detach().cpu().numpy()
        size_residual_label = data_dict['size_residual_label'].detach().cpu().numpy()
        num_instance = center_label.shape[0]
        gt_bboxes = np.zeros((num_instance, 8, 3))
        for j in range(num_instance):
            heading_angle = self.DC.class2angle(heading_class_label[j], heading_residual_label[j])
            box_size = self.DC.class2size(int(size_class_label[j]), size_residual_label[j])
            gt_bboxes[j] = get_3d_box(center_label[j,:], box_size, heading_angle)
        data_dict['gt_bboxes'] = gt_bboxes
        
        if self.cfg.model.crop_bbox:
            # parse crop_bbox
            proposal_crop_bboxes = loss_input['proposal_crop_bboxes']
            proposal_crop_bboxes = proposal_crop_bboxes.detach().cpu().numpy() # (nProposals, center+size+heading+label)
            proposal_crop_bboxes = get_3d_box_batch(proposal_crop_bboxes[:, 0:3], proposal_crop_bboxes[:, 3:6], proposal_crop_bboxes[:, 6]) # (nProposals, 8, 3)
            num_proposal = proposal_crop_bboxes.shape[0] 
            
            crop_bbox_ious = np.zeros(num_proposal)
            for b in range(batch_size):
                proposal_batch_idx = torch.nonzero(proposals_batchId == b)
                pred_num = len(proposal_batch_idx) #pred_batch_end - pred_batch_start # N
                gt_batch_start, gt_batch_end = instance_offsets[b], instance_offsets[b+1]
                gt_num = gt_batch_end - gt_batch_start # M
                for i in range(pred_num):
                    crop_bbox_iou = get_aabb3d_iou_batch(np.tile(proposal_crop_bboxes[proposal_batch_idx[i]], (gt_num, 1, 1)), gt_bboxes[gt_batch_start:gt_batch_end])
                    crop_bbox_ious[proposal_batch_idx[i]] = np.max(crop_bbox_iou)
            eval_dict['crop_bbox_iou'] = (crop_bbox_ious.mean(), num_proposal)
            data_dict['proposal_crop_bboxes'] = proposal_crop_bboxes
            # import pdb; pdb.set_trace()
            
        if self.cfg.model.pred_bbox:
            # parse pred_bbox
            pred_center, heading_scores, heading_residuals_normalized, heading_residuals, size_scores, size_residuals_normalized, size_residuals, sem_cls_scores = loss_input['proposal_pred_bboxes']
            pred_center = pred_center.detach().cpu().numpy()
            pred_heading_class = torch.argmax(heading_scores, -1).detach().cpu().numpy() # num_proposal
            pred_heading_residual = torch.gather(heading_scores, 1, pred_heading_class.unsqueeze(-1)).squeeze(1).detach().cpu().numpy() # num_proposal
            pred_size_class = torch.argmax(size_scores, -1).detach().cpu().numpy() # num_proposal
            pred_size_residual = torch.gather(size_residuals, 1, pred_size_class.unsqueeze(-1).unsqueeze(-1).repeat(1,1,3)).squeeze(1).detach().cpu().numpy() # num_proposal,3

            num_proposal = pred_center.shape[0] 
            proposal_pred_bboxes = np.zeros((num_proposal, 8, 3))
            for j in range(num_proposal):
                heading_angle = self.DC.class2angle(pred_heading_class[j], pred_heading_residual[j])
                box_size = self.DC.class2size(int(pred_size_class[j]), pred_size_residual[j])
                proposal_pred_bboxes[j] = get_3d_box(pred_center[j,:], box_size, heading_angle)
            
            pred_bbox_ious = np.zeros(num_proposal)
            for b in range(batch_size):
                proposal_batch_idx = torch.nonzero(proposals_batchId == b)
                pred_num = len(proposal_batch_idx) #pred_batch_end - pred_batch_start # N
                gt_batch_start, gt_batch_end = instance_offsets[b], instance_offsets[b+1]
                gt_num = gt_batch_end - gt_batch_start # M
                for i in range(pred_num):
                    pred_bbox_iou = get_aabb3d_iou_batch(np.tile(proposal_pred_bboxes[proposal_batch_idx[i]], (gt_num, 1, 1)), gt_bboxes[gt_batch_start:gt_batch_end])
                    pred_bbox_ious[proposal_batch_idx[i]] = np.max(pred_bbox_iou)
            eval_dict['pred_bbox_iou'] = (pred_bbox_ious.mean(), num_proposal)
            data_dict['proposal_pred_bboxes'] = proposal_pred_bboxes
            
        if self.cfg.model.crop_bbox and self.cfg.model.pred_bbox:
            assert proposal_crop_bboxes.shape == proposal_pred_bboxes.shape
            ious = get_aabb3d_iou_batch(proposal_crop_bboxes, proposal_pred_bboxes)
            eval_dict['pred_crop_bbox_iou'] = (ious.mean(), num_proposal)


    def _feed(self, data, epoch=0):
        for key in data:
            data[key] = data[key].cuda()
        data['epoch'] = epoch

        if self.cfg.model.use_coords:
            data['feats'] = torch.cat((data['feats'], data['locs']), 1)

        data['voxel_feats'] = pointgroup_ops.voxelization(data['feats'], data['v2p_map'], self.cfg.data.mode)  # (M, C), float, cuda

        ret = self.model(data)
        
        return ret
    
    
    def _parse_feed_ret(self, data, ret):
        semantic_scores = ret['semantic_scores'] # (N, nClass) float32, cuda
        pt_offsets = ret['pt_offsets']           # (N, 3), float32, cuda
        
        preds = {}
        loss_input = {}
        
        preds['semantic'] = semantic_scores
        preds['pt_offsets'] = pt_offsets
        if self.mode != 'test':
            loss_input['semantic_scores'] = (semantic_scores, data["sem_labels"])
            loss_input['pt_offsets'] = (pt_offsets, data['locs'], data["instance_info"], data["instance_ids"])
        
        if self.curr_epoch > self.cfg.cluster.prepare_epochs:
            scores, proposals_idx, proposals_offset = ret['proposal_scores']
            preds['score'] = scores
            preds['proposals'] = (proposals_idx, proposals_offset)
            preds['proposal_crop_bboxes'] = ret['proposal_crop_bbox']
            
            if self.mode != 'test':
                loss_input['proposal_scores'] = (scores, proposals_idx, proposals_offset, data["instance_num_point"])
                # scores: (nProposal, 1) float, cuda
                # proposals_idx: (sumNPoint, 2), int, cpu, dim 0 for cluster_id, dim 1 for corresponding point idxs in N
                # proposals_offset: (nProposal + 1), int, cpu
                loss_input['proposal_thres_mask'] = ret['proposal_thres_mask']
                loss_input['proposals_batchId'] = ret['proposals_batchId']
                if self.cfg.model.crop_bbox:
                    loss_input['proposal_crop_bboxes'] = ret['proposal_crop_bbox']
                if self.cfg.model.pred_bbox:
                    loss_input['proposal_pred_bboxes'] = (ret['center'], ret['heading_scores'], ret['heading_residuals_normalized'], ret['heading_residuals'], ret['size_scores'], ret['size_residuals_normalized'], ret['size_residuals'], ret['sem_cls_scores'])
        
        return preds, loss_input
    
    
    def _log_report(self, meters, meter_dict, iter, iter_start_time=0):
        with torch.no_grad():
            meter_dict['iter_time'] = time.time() - iter_start_time
            
            for k, v in meter_dict.items():
                if type(v) == tuple:
                    meters.update(k, v[0], v[1])
                else:
                    meters.update(k, v)
                    
            if self.mode == 'train' and (iter+1) % self.cfg.log.verbose == 0:
                curr_iter = (self.curr_epoch - 1) * len(self.loader) + iter + 1
                remain_iter = self.max_iter - curr_iter
                remain_time_sec = time.gmtime(remain_iter * meters.get_avg('iter_time'))
                remain_time = time.strftime("%H:%M:%S", remain_time_sec)
                
                self.logger.debug(
                    f"epoch: {self.curr_epoch}/{self.total_epoch} iter: {iter+1}/{len(self.loader)} bbox_loss: {meters.get_val('bbox_loss'):.4f}({meters.get_avg('bbox_loss'):.4f}) loss: {meters.get_val('total_loss'):.4f}({meters.get_avg('total_loss'):.4f}) avg_iter_time: {meters.get_avg('iter_time'):.4f} remain_time: {remain_time} crop_bbox_iou: {meters.get_avg('crop_bbox_iou'):.4f}")
            if (iter == len(self.loader) - 1): print()


    def train(self, epoch):
        self.mode = 'train'
        self.loader = self.dataloader[self.mode]
        self.max_iter = self.total_epoch * len(self.loader)
        self.model.train()

        self.curr_epoch = epoch
        meters = Meters(*self.cfg.log.meter_names)
        start_epoch_time = time.time()

        for i, batch in enumerate(self.loader):
            torch.cuda.empty_cache()
            iter_start_time = time.time()

            ##### adjust learning rate
            # step_learning_rate(self.optimizer, self.cfg.train.optim.lr, epoch - 2, self.cfg.train.step_epoch, self.cfg.train.multiplier)

            ##### prepare input and forward
            ret = self._feed(batch, epoch)
            _, loss_input = self._parse_feed_ret(batch, ret)
            meter_dict = self._loss(loss_input, batch, epoch)
            self.get_bbox_iou(loss_input, batch, meter_dict)

            ##### backward
            self.optimizer.zero_grad()
            meter_dict['total_loss'][0].backward()
            self.optimizer.step()

            self._log_report(meters, meter_dict, i, iter_start_time)

        self.logger.info(f"===> summary of epoch: {self.curr_epoch}/{self.total_epoch}\ntrain loss: {meters.get_avg('total_loss'):.4f}, time: {time.time() - start_epoch_time:.2f}s\n")

        if self.check_save_condition(epoch):
            self.save_checkpoint(epoch)

        for k in self.cfg.log.tb_names:
            self.logger.tb_add_scalar(k + f'_{self.mode}', meters.get_avg(k), epoch)


    def eval(self, epoch):
        self.mode = 'val'
        self.model.eval()
        self.loader = self.dataloader[self.mode]
        self.max_iter = len(self.loader)
        self.logger.info('>>>>>>>>>>>>>>>> Start Evaluation >>>>>>>>>>>>>>>>')
        meters = Meters(*self.cfg.log.meter_names)

        with torch.no_grad():
            start_epoch_time = time.time()
            for i, batch in enumerate(self.loader):
                torch.cuda.empty_cache()
                iter_start_time = time.time()

                ##### prepare input and forward
                ret = self._feed(batch, self.curr_epoch)
                _, loss_input = self._parse_feed_ret(batch, ret)
                meter_dict = self._loss(loss_input, batch, self.curr_epoch)

                ##### meter_dict
                self._log_report(meters, meter_dict, i, iter_start_time)
                ##### print
                self.logger.debug(f"\riter: {i+1}/{len(self.loader)} loss: {meters.get_val('total_loss'):.4f}({meters.get_avg('total_loss'):.4f}) crop_bbox_iou: {meters.get_avg('crop_bbox_iou'):.4f}")

            self.logger.info(f"epoch: {self.curr_epoch}/{self.total_epoch}, val loss: {meters.get_avg('total_loss'):.4f}, time: {time.time() - start_epoch_time:.2f}s\n")

            for k in self.cfg.log.tb_names:
                self.logger.tb_add_scalar(k + f'_{self.mode}', meters.get_avg(k), epoch)
                
    
    def test(self, split):
        from data.scannet.model_util_scannet import NYU20_CLASS_IDX
        NYU20_CLASS_IDX = NYU20_CLASS_IDX[1:] # for scannet temporarily
        self.mode = 'test'
        self.curr_epoch = self.start_epoch
        self.model.eval()
        self.loader = self.dataloader[split]
        self.logger.info('>>>>>>>>>>>>>>>> Start Inference >>>>>>>>>>>>>>>>')

        with torch.no_grad():
            for i, batch in enumerate(tqdm(self.loader)):
                # if batch['scene_id'] < 20800: continue
                N = batch['feats'].shape[0]
                scene_name = self.dataset[split].scene_names[i]
                mesh = self.dataset[split].scenes[i]['aligned_mesh']
                
                ret = self._feed(batch, self.curr_epoch)
                preds, _ = self._parse_feed_ret(batch, ret)
                
                ##### get predictions (#1 semantic_pred, pt_offsets; #2 scores, proposals_pred)
                semantic_scores = preds['semantic']  # (N, nClass) float32, cuda, 0: unannotated
                semantic_pred_labels = semantic_scores.max(1)[1]  # (N) long, cuda
                semantic_class_idx = torch.tensor(NYU20_CLASS_IDX, dtype=torch.int).cuda() # (nClass)
                semantic_pred_class_idx = semantic_class_idx[semantic_pred_labels].cpu().numpy()
                
                if self.curr_epoch > self.cfg.cluster.prepare_epochs:
                    proposals_score = torch.sigmoid(preds['score'].view(-1)) # (nProposal,) float, cuda
                    proposals_idx, proposals_offset = preds['proposals']
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
                
                pred_path = os.path.join(self.logger.log_path, 'split_pred', split)
                
                sem_pred_path = os.path.join(pred_path, 'semantic')
                os.makedirs(sem_pred_path, exist_ok=True)
                sem_pred_file_path = os.path.join(sem_pred_path, f'{scene_name}.txt')
                np.savetxt(sem_pred_file_path, semantic_pred_class_idx, fmt='%d')
                
                if self.curr_epoch > self.cfg.cluster.prepare_epochs:
                    inst_pred_path = os.path.join(pred_path, 'instance')
                    inst_pred_masks_path = os.path.join(inst_pred_path, 'predicted_masks')
                    os.makedirs(inst_pred_path, exist_ok=True)
                    os.makedirs(inst_pred_masks_path, exist_ok=True)
                    cluster_ids = np.ones(shape=(N)) * -1 # id starts from 0
                    with open(os.path.join(inst_pred_path, f'{scene_name}.txt'), 'w') as f:
                        for c_id in range(nclusters):
                            cluster_i = clusters_mask[c_id]  # (N)
                            cluster_ids[cluster_i == 1] = c_id
                            assert np.unique(semantic_pred_class_idx[cluster_i == 1]).size == 1
                            cluster_i_class_idx = semantic_pred_class_idx[cluster_i == 1][0]
                            score = clusters_score[c_id]
                            f.write(f'predicted_masks/{scene_name}_{c_id:03d}.txt {cluster_i_class_idx} {score:.4f}\n')
                            np.savetxt(os.path.join(inst_pred_masks_path, f'{scene_name}_{c_id:03d}.txt'), cluster_i, fmt='%d')
                    np.savetxt(os.path.join(inst_pred_path, f'{scene_name}.cluster_ids.txt'), cluster_ids, fmt='%d')
                    
                    if self.cfg.test.requires_visualize:
                        from visualize.scannet.generate_ply import visualize_pred_instance
                        os.makedirs(os.path.join(inst_pred_path, 'visualize'), exist_ok=True)
                        inst_ply_path = os.path.join(inst_pred_path, 'visualize', f'{scene_name}.ply')
                        visualize_pred_instance(inst_ply_path, mesh, cluster_ids, semantic_pred_class_idx)
                        if self.cfg.model.crop_bbox:
                            from visualize.scannet.generate_ply import visualize_crop_bboxes
                            from lib.utils.bbox import get_aabb3d_iou_batch, get_3d_box_batch
                            crop_bbox_ply_path = os.path.join(inst_pred_path, 'visualize', f'{scene_name}.crop_bbox.ply')
                            visualize_crop_bboxes(crop_bbox_ply_path, mesh, preds['proposal_crop_bboxes'].detach().cpu().numpy())
                            # gt_bbox_ply_path = os.path.join(inst_pred_path, 'visualize', f'{scene_name}.gt_bbox.ply')
                            # visualize_crop_bboxes(gt_bbox_ply_path, mesh, batch['gt_bbox'].detach().cpu().numpy())
                        import pdb; pdb.set_trace()
                    
                    
    def eval_detection(self):
        self.mode = 'val'
        self.model.eval()
        self.loader = self.dataloader[self.mode]
        self.max_iter = len(self.loader)
        self.logger.info('>>>>>>>>>>>>>>>> Start Evaluation >>>>>>>>>>>>>>>>')
        self.curr_epoch = self.start_epoch

        from lib.utils.eval import APCalculator, parse_predictions, parse_groundtruths
        # config
        POST_DICT = {
            "remove_empty_box": False, 
            "use_3d_nms": True, 
            "nms_iou": 0.25,
            "use_old_type_nms": False, 
            "cls_nms": True, 
            "per_class_proposal": True,
            "conf_thresh": 0.09,
            "dataset_config": self.DC
        }
        AP_IOU_THRESHOLDS = [0.25, 0.5]
        AP_CALCULATOR_LIST = [APCalculator(iou_thresh, self.DC.class2type) for iou_thresh in AP_IOU_THRESHOLDS]

        sem_acc = []
        
        with torch.no_grad():
            for i, batch in tqdm(enumerate(self.loader)):
                torch.cuda.empty_cache()

                ##### prepare input and forward
                ret = self._feed(batch, self.curr_epoch)
                _, loss_input = self._parse_feed_ret(batch, ret)
                meter_dict = self._loss(loss_input, batch, self.curr_epoch)
                self.get_bbox_iou(loss_input, batch, meter_dict)
                
                batch_pred_map_cls = parse_predictions(ret, batch, POST_DICT) 
                batch_gt_map_cls = parse_groundtruths(batch, POST_DICT) 
                for ap_calculator in AP_CALCULATOR_LIST:
                    ap_calculator.step(batch_pred_map_cls, batch_gt_map_cls)
                
        for i, ap_calculator in enumerate(AP_CALCULATOR_LIST):
            print()
            print("-"*10, "iou_thresh: %f"%(AP_IOU_THRESHOLDS[i]), "-"*10)
            metrics_dict = ap_calculator.compute_metrics()
            for key in metrics_dict:
                print("eval %s: %f"%(key, metrics_dict[key]))
