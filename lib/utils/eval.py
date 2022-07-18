import os
import numpy as np
import torch

from lib.utils.bbox import get_3d_box
from lib.utils.eval_det import eval_det_cls, eval_det_multiprocessing, get_iou_obb
from lib.utils.nms import nms_2d_faster, nms_3d_faster, nms_3d_faster_samecls


#####################################################
############### Instance Segmentation ###############
#####################################################

def write_gt_sem_inst_ids(sem_labels, instance_ids, file_path):
    """ Generate instance txt files for evaluation. Each line represents a number xx00y combining semantic label (x) and instance id (y) for each point.

    Args:
        sem_labels (np.array): {0,1,...,20} (N,) 0:unannotated
        instance_ids (np.array): {0,1,...,instance_num} (N,) 0:unannotated
    """
    from data.nyuv2.model_utils import NYU20_CLASS_IDX
    # 0 for unannotated, xx00y: x for semantic_label, y for inst_id (1~instance_num)
    sem_class_idx = np.array(NYU20_CLASS_IDX, dtype=np.int)[sem_labels]
    sem_inst_encoding = sem_class_idx * 1000 + instance_ids # np.zeros(instance_ids.shape, dtype=np.int32)  
    # instance_num = int(instance_ids.max())
    # for inst_id in range(1, instance_num+1):
    #     instance_mask = np.where(instance_ids == inst_id)[0]
    #     sem_seg = sem_labels[instance_mask]
    #     unique_sem_ids, sem_id_counts = np.unique(sem_seg, return_counts=True)
    #     sem_id = unique_sem_ids[np.argmax(sem_id_counts)] # choose the most frequent semantic id
        
    #     semantic_label = NYU20_CLASS_IDX[sem_id]
    #     sem_inst_encoding[instance_mask] = semantic_label * 1000 + inst_id
    np.savetxt(file_path, sem_inst_encoding, fmt='%d')
    

def read_sem_ids(file_path, mode):
    with open(file_path, 'r') as f:
        if mode == 'gt':
            sem_class_idx = [int(encoded_id.rstrip()) // 1000 for encoded_id in f.readlines()]
        else:
            sem_class_idx = [int(encoded_id.rstrip()) for encoded_id in f.readlines()]
        sem_class_idx = np.array(sem_class_idx, dtype=np.int)
    return sem_class_idx


def read_inst_ids(file_path, mode):
    with open(file_path, 'r') as f:
        instance_ids = [int(encoded_id.rstrip()) for encoded_id in f.readlines()]
        instance_ids = np.array(instance_ids, dtype=np.int)
        if mode == 'gt':
            sem_class_idx, instance_ids = instance_ids // 1000, instance_ids % 1000
        else:
            sem_class_idx = None
    return sem_class_idx, instance_ids


def parse_inst_pred_file(file_path, alignment=None):
    lines = open(file_path).read().splitlines()
    instance_info = {}
    for line in lines:
        info = {}
        if alignment:
            mask_rel_path, class_idx, aligned_token_idx, confidence = line.split(' ')
            info['aligned_token_idx'] = int(aligned_token_idx)
        else:
            mask_rel_path, class_idx, confidence = line.split(' ')
        mask_file = os.path.join(os.path.dirname(file_path), mask_rel_path)
        info["class_idx"] = int(class_idx)
        info["confidence"] = float(confidence)
        instance_info[mask_file] = info
    return instance_info


def get_nms_instances(cross_ious, scores, threshold):
    """ non max suppression for 3D instance proposals based on cross ious and scores

    Args:
        ious (np.array): cross ious, (n, n)
        scores (np.array): scores for each proposal, (n,)
        threshold (float): iou threshold

    Returns:
        np.array: idx of picked instance proposals
    """
    # ixs = scores.argsort()[::-1]
    ixs = np.argsort(-scores) # descending order
    pick = []
    while len(ixs) > 0:
        i = ixs[0]
        pick.append(i)
        ious = cross_ious[i, ixs[1:]]
        remove_ixs = np.where(ious > threshold)[0] + 1
        ixs = np.delete(ixs, remove_ixs)
        ixs = np.delete(ixs, 0)
        
    return np.array(pick, dtype=np.int32)


class Instance(object):
    target_id = 0
    class_idx = 0
    vert_count = 0
    med_dist = -1
    dist_conf = 0.0

    def __init__(self, instance_ids, class_idx, target_id, aligned_token_idx=[]):
        self.target_id = int(target_id)
        self.class_idx = int(class_idx)
        self.aligned_token_idx = aligned_token_idx
        self.vert_count = int(self.get_num_verts(instance_ids, target_id))

    def get_num_verts(self, instance_ids, target_id):
        return (instance_ids == target_id).sum()

    def to_json(self):
        return json.dumps(self, default=lambda o: o.__dict__, sort_keys=True, indent=4)

    def to_dict(self):
        dict = {}
        dict["target_id"] = self.target_id
        dict["class_idx"] = self.class_idx
        dict["aligned_token_idx"] = self.aligned_token_idx
        dict["vert_count"] = self.vert_count
        dict["med_dist"] = self.med_dist
        dict["dist_conf"] = self.dist_conf
        return dict

    def from_json(self, data):
        self.target_id = int(data["target_id"])
        self.class_idx = int(data["class_idx"])
        self.aligned_token_idx = dict["aligned_token_idx"]
        self.vert_count = int(data["vert_count"])
        if ("med_dist" in data):
            self.med_dist = float(data["med_dist"])
            self.dist_conf = float(data["dist_conf"])

    def __str__(self):
        return f"({str(self.target_id)})"


def get_instances(sem_class_idx, instance_ids, gt_class_idx, gt_class_names, id2name):
    instances = {}
    for class_name in gt_class_names:
        instances[class_name] = []
    unique_inst_ids = np.unique(instance_ids)
    for inst_id in unique_inst_ids:
        if inst_id == 0:
            continue
        class_idx = np.argmax(np.bincount(sem_class_idx[instance_ids == inst_id]))
        inst = Instance(instance_ids, class_idx, inst_id)
        if inst.class_idx in gt_class_idx:
            instances[id2name[inst.class_idx]].append(inst.to_dict())
    return instances


#####################################################
###############    Object Detection   ###############
#####################################################

def softmax(x):
    ''' Numpy function for softmax'''
    shape = x.shape
    probs = np.exp(x - np.max(x, axis=len(shape)-1, keepdims=True))
    probs /= np.sum(probs, axis=len(shape)-1, keepdims=True)
    return probs


def parse_predictions(end_points, config_dict):
    """ Parse predictions to OBB parameters and suppress overlapping boxes
    
    Args:
        end_points: dict
            {point_clouds, center, heading_scores, heading_residuals,
            size_scores, size_residuals, sem_cls_scores}
        config_dict: dict
            {dataset_config, remove_empty_box, use_3d_nms, nms_iou,
            use_old_type_nms, conf_thresh, per_class_proposal}
    Returns:
        batch_pred_map_cls: a list of len == batch size (BS)
            [pred_list_i], i = 0, 1, ..., BS-1
            where pred_list_i = [(pred_sem_cls, box_params, box_score)_j]
            where j = 0, ..., num of valid detections - 1 from sample input i
    """
    DC = config_dict['dataset_config']
    bsize = len(end_points["batch_offsets"]) - 1
        
    bbox_corners = end_points["proposal_crop_bboxes"] # (nProposal, 8, 3)
    num_proposal = end_points["proposal_crop_bboxes"].shape[0]
    pred_sem_cls = end_points['proposal_crop_bbox'].cpu().numpy()[:, 7] - 2
    pred_sem_cls[pred_sem_cls < 0] = 17

    K = num_proposal
    nonempty_box_mask = np.ones((K,))

    if config_dict['remove_empty_box']:
        # -------------------------------------
        # Remove predicted boxes without any point within them..
        batch_pc = end_points['point_clouds'].cpu().numpy()[:,:,0:3] # B,N,3
        for i in range(bsize):
            pc = batch_pc[i,:,:] # (N,3)
            for j in range(K):
                box3d = bbox_corners[i,j,:,:] # (8,3)
                # box3d = flip_axis_to_depth(box3d)
                pc_in_box,inds = extract_pc_in_box3d(pc, box3d)
                if len(pc_in_box) < 5:
                    nonempty_box_mask[i,j] = 0
        # -------------------------------------

    thres_mask = end_points['proposal_thres_mask']
    proposals_batchId = end_points['proposals_batchId']
    batch_pred_map_cls = [] # a list (len: batch_size) of list (len: num of predictions per sample) of tuples of pred_cls, pred_box and conf (0-1)
    obj_prob = torch.sigmoid(end_points['proposal_scores'][0].view(-1))[thres_mask].detach().cpu().numpy()
    
    for b in range(bsize):
        proposal_batch_idx = torch.nonzero(proposals_batchId == b).view(-1).detach().cpu().numpy()
        num_proposal_batch = len(proposal_batch_idx)
        bbox_corners_batch = bbox_corners[proposal_batch_idx]
        obj_prob_batch = obj_prob[proposal_batch_idx]
        pred_sem_cls_batch = pred_sem_cls[proposal_batch_idx]
        pred_mask = np.zeros((num_proposal_batch,))
        if not config_dict['use_3d_nms']:
            # ---------- NMS input: pred_with_prob in (B,K,7) -----------
            boxes_2d_with_prob = np.zeros((num_proposal_batch, 5))
            for j in range(num_proposal_batch):
                boxes_2d_with_prob[j,0] = np.min(bbox_corners_batch[j,:,0])
                boxes_2d_with_prob[j,2] = np.max(bbox_corners_batch[j,:,0])
                boxes_2d_with_prob[j,1] = np.min(bbox_corners_batch[j,:,2])
                boxes_2d_with_prob[j,3] = np.max(bbox_corners_batch[j,:,2])
                boxes_2d_with_prob[j,4] = obj_prob_batch[j]
            nonempty_box_inds = np.where(nonempty_box_mask[proposal_batch_idx]==1)[0]
            pick = nms_2d_faster(boxes_2d_with_prob[nonempty_box_inds,:], config_dict['nms_iou'], config_dict['use_old_type_nms'])
            assert(len(pick)>0)
            pred_mask[nonempty_box_inds[pick]] = 1
            # ---------- NMS output: pred_mask in (B,K) -----------
        elif config_dict['use_3d_nms'] and (not config_dict['cls_nms']):
            # ---------- NMS input: pred_with_prob in (B,K,7) -----------
            boxes_3d_with_prob = np.zeros((num_proposal_batch, 7))
            for j in range(num_proposal_batch):
                boxes_3d_with_prob[j,0] = np.min(bbox_corners_batch[j,:,0])
                boxes_3d_with_prob[j,1] = np.min(bbox_corners_batch[j,:,1])
                boxes_3d_with_prob[j,2] = np.min(bbox_corners_batch[j,:,2])
                boxes_3d_with_prob[j,3] = np.max(bbox_corners_batch[j,:,0])
                boxes_3d_with_prob[j,4] = np.max(bbox_corners_batch[j,:,1])
                boxes_3d_with_prob[j,5] = np.max(bbox_corners_batch[j,:,2])
                boxes_3d_with_prob[j,6] = obj_prob_batch[j]
            nonempty_box_inds = np.where(nonempty_box_mask[proposal_batch_idx]==1)[0]
            pick = nms_3d_faster(boxes_3d_with_prob[nonempty_box_inds,:], config_dict['nms_iou'], config_dict['use_old_type_nms'])
            assert(len(pick)>0)
            pred_mask[nonempty_box_inds[pick]] = 1
            # ---------- NMS output: pred_mask in (B,K) -----------
        elif config_dict['use_3d_nms'] and config_dict['cls_nms']:
            # ---------- NMS input: pred_with_prob in (B,K,8) -----------
            boxes_3d_with_prob = np.zeros((num_proposal_batch, 8))
            for j in range(num_proposal_batch):
                boxes_3d_with_prob[j,0] = np.min(bbox_corners_batch[j,:,0])
                boxes_3d_with_prob[j,1] = np.min(bbox_corners_batch[j,:,1])
                boxes_3d_with_prob[j,2] = np.min(bbox_corners_batch[j,:,2])
                boxes_3d_with_prob[j,3] = np.max(bbox_corners_batch[j,:,0])
                boxes_3d_with_prob[j,4] = np.max(bbox_corners_batch[j,:,1])
                boxes_3d_with_prob[j,5] = np.max(bbox_corners_batch[j,:,2])
                boxes_3d_with_prob[j,6] = obj_prob_batch[j]
                boxes_3d_with_prob[j,7] = pred_sem_cls_batch[j] # only suppress if the two boxes are of the same class!!
            nonempty_box_inds = np.where(nonempty_box_mask[proposal_batch_idx]==1)[0]
            pick = nms_3d_faster_samecls(boxes_3d_with_prob[nonempty_box_inds,:], config_dict['nms_iou'], config_dict['use_old_type_nms'])
            assert(len(pick)>0)
            pred_mask[nonempty_box_inds[pick]] = 1
            # ---------- NMS output: pred_mask in (B,K) -----------
        if config_dict['per_class_proposal']:
            cur_list = []
            for ii in range(DC.num_class):
                cur_list += [(ii, bbox_corners_batch[j], obj_prob_batch[j]) \
                    for j in range(num_proposal_batch) if pred_mask[j]==1 and pred_sem_cls_batch[j]==ii and obj_prob_batch[j]>config_dict['conf_thresh']]
            batch_pred_map_cls.append(cur_list)
        else:
            batch_pred_map_cls.append([(pred_sem_cls_batch[j], bbox_corners_batch[j], obj_prob_batch[j]) \
                for j in range(num_proposal_batch) if pred_mask[j]==1 and obj_prob_batch[j]>config_dict['conf_thresh']])
    
    end_points['batch_pred_map_cls'] = batch_pred_map_cls

    return batch_pred_map_cls


def parse_groundtruths(end_points, config_dict):
    """ Parse groundtruth labels to OBB parameters.
    
    Args:
        end_points: dict
            {center_label, heading_class_label, heading_residual_label,
            size_class_label, size_residual_label, sem_cls_label,
            box_label_mask}
        config_dict: dict
            {dataset_config}
    Returns:
        batch_gt_map_cls: a list  of len == batch_size (BS)
            [gt_list_i], i = 0, 1, ..., BS-1
            where gt_list_i = [(gt_sem_cls, gt_box_params)_j]
            where j = 0, ..., num of objects - 1 at sample input i
    """
    DC = config_dict['dataset_config']
    center_label = end_points['center_label']
    heading_class_label = end_points['heading_class_label'].detach().cpu().numpy()
    heading_residual_label = end_points['heading_residual_label'].detach().cpu().numpy()
    size_class_label = end_points['size_class_label'].detach().cpu().numpy()
    size_residual_label = end_points['size_residual_label'].detach().cpu().numpy()
    sem_cls_label = end_points['sem_cls_label']
    
    num_proposal = center_label.shape[0] 
    bsize = len(end_points["batch_offsets"]) - 1
    proposal_offsets = end_points['instance_offsets'].detach().cpu().numpy()
    batch_gt_map_cls = []

    gt_corners_3d_upright_camera = np.zeros((num_proposal, 8, 3))
    gt_center_upright_camera = center_label[:,0:3].detach().cpu().numpy()
    
    for b in range(bsize):
        start, end = proposal_offsets[b], proposal_offsets[b+1]
        num_proposal_batch = end - start
        gt_center_upright_camera_batch = gt_center_upright_camera[start:end,:]
        for j in range(num_proposal_batch):
            heading_angle = DC.class2angle(heading_class_label[start+j], heading_residual_label[start+j])
            box_size = DC.class2size(int(size_class_label[start+j]), size_residual_label[start+j])
            corners_3d_upright_camera = get_3d_box(gt_center_upright_camera_batch[j,:], box_size, heading_angle)
            gt_corners_3d_upright_camera[start+j] = corners_3d_upright_camera

        batch_gt_map_cls.append([(sem_cls_label[start+j].item(), gt_corners_3d_upright_camera[start+j]) for j in range(num_proposal_batch)])
        
    end_points['batch_gt_map_cls'] = batch_gt_map_cls

    return batch_gt_map_cls


class APCalculator(object):
    ''' Calculating Average Precision '''
    def __init__(self, ap_iou_thresh=0.25, class2type_map=None):
        """
        Args:
            ap_iou_thresh: float between 0 and 1.0
                IoU threshold to judge whether a prediction is positive.
            class2type_map: [optional] dict {class_int:class_name}
        """
        self.ap_iou_thresh = ap_iou_thresh
        self.class2type_map = class2type_map
        self.reset()
        
    def step(self, batch_pred_map_cls, batch_gt_map_cls):
        """ Accumulate one batch of prediction and groundtruth.
        
        Args:
            batch_pred_map_cls: a list of lists [[(pred_cls, pred_box_params, score),...],...]
            batch_gt_map_cls: a list of lists [[(gt_cls, gt_box_params),...],...]
                should have the same length with batch_pred_map_cls (batch_size)
        """
        
        bsize = len(batch_pred_map_cls)
        assert(bsize == len(batch_gt_map_cls))
        for i in range(bsize):
            self.gt_map_cls[self.scan_cnt] = batch_gt_map_cls[i] 
            self.pred_map_cls[self.scan_cnt] = batch_pred_map_cls[i] 
            self.scan_cnt += 1
    
    def compute_metrics(self):
        """ Use accumulated predictions and groundtruths to compute Average Precision.
        """
        rec, prec, ap = eval_det_multiprocessing(self.pred_map_cls, self.gt_map_cls, ovthresh=self.ap_iou_thresh, get_iou_func=get_iou_obb)
        ret_dict = {} 
        for key in sorted(ap.keys()):
            clsname = self.class2type_map[key] if self.class2type_map else str(key)
            ret_dict['%s Average Precision'%(clsname)] = ap[key]
        ret_dict['mAP'] = np.mean(list(ap.values()))
        rec_list = []
        for key in sorted(ap.keys()):
            clsname = self.class2type_map[key] if self.class2type_map else str(key)
            try:
                ret_dict['%s Recall'%(clsname)] = rec[key][-1]
                rec_list.append(rec[key][-1])
            except:
                ret_dict['%s Recall'%(clsname)] = 0
                rec_list.append(0)
        ret_dict['AR'] = np.mean(rec_list)
        return ret_dict

    def reset(self):
        self.gt_map_cls = {} # {scan_id: [(classname, bbox)]}
        self.pred_map_cls = {} # {scan_id: [(classname, bbox, score)]}
        self.scan_cnt = 0