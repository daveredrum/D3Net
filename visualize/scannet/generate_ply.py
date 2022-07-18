import os, sys
import argparse
from omegaconf import OmegaConf
from plyfile import PlyData

import numpy as np
import torch
import matplotlib.pyplot as plt

sys.path.append(os.getcwd())
from data.scannet.model_util_scannet import SCANNET_COLOR_MAP
from lib.utils.pc import write_ply_rgb, write_ply_colorful, write_ply_rgb_face
from lib.utils.bbox import write_cylinder_bbox


cfg = OmegaConf.load('conf/path.yaml')
MEAN_COLOR_RGB = np.array([109.8, 97.2, 83.8])


def generate_rgb_ply(args):
    split = args.split
    data_dir = cfg.SCANNETV2_PATH.split_data
    output_dir = f'/project/3dlg-hcvc/dense-scanrefer/scannet/rgb/{split}' # TODO
    os.makedirs(output_dir, exist_ok=True)
    scene_ids_file = os.path.join(cfg.SCANNETV2_PATH.meta_data, f'scannetv2_{split}.txt')
    scene_ids = [scene_id.rstrip() for scene_id in open(scene_ids_file)]
    
    for scene_id in scene_ids:
        print(scene_id)
        rgb_file = os.path.join(data_dir, split, f'{scene_id}.pth')
        rgb_ply = os.path.join(output_dir, f'{scene_id}.ply')
        scannet_data = torch.load(rgb_file)
        points = scannet_data['aligned_mesh'][:, :3].astype(np.float)
        colors = scannet_data['aligned_mesh'][:, 3:6].astype(np.uint8)
        write_ply_rgb(points, colors, rgb_ply)


def generate_gt_sem_ply(args):
    from data.scannet.model_util_scannet import NYU20_CLASS_IDX
    split = args.split
    scan_dir = cfg.SCANNETV2_PATH.split_scans
    output_dir = f'/local-scratch/qiruiw/dataset/scannet/split_nyu20_labels/{split}' # TODO
    os.makedirs(output_dir, exist_ok=True)
    scene_ids_file = os.path.join(cfg.SCANNETV2_PATH.meta_data, f'scannetv2_{split}.txt')
    scene_ids = [scene_id.rstrip() for scene_id in open(scene_ids_file)]
    
    valid_class_idx = np.array(NYU20_CLASS_IDX[1:])
    
    for scene_id in scene_ids:
        print(scene_id)
        nyu40_sem_ply = os.path.join(scan_dir, split, scene_id, f'{scene_id}_vh_clean_2.labels.ply')
        nyu20_sem_ply = os.path.join(output_dir, f'{scene_id}.ply')
        
        with open(nyu40_sem_ply, 'rb') as f:
            plydata = PlyData.read(f)
            num_verts = plydata['vertex'].count
            points = np.zeros(shape=[num_verts, 3], dtype=np.float)
            points[:,0] = plydata['vertex'].data['x']
            points[:,1] = plydata['vertex'].data['y']
            points[:,2] = plydata['vertex'].data['z']
            sem_labels = np.array(plydata['vertex']['label'])
            invalid_label_idx = np.logical_not(np.in1d(sem_labels, valid_class_idx))
            sem_labels[invalid_label_idx] = 0
            
        write_ply_colorful(points, sem_labels, nyu20_sem_ply, colormap=SCANNET_COLOR_MAP)


def generate_pred_sem_ply(args):
    split = args.split
    scan_dir = cfg.SCANNETV2_PATH.split_scans
    pred_dir = f'/local-scratch/qiruiw/research/pointgroup-minkowski/log/scannet/pointgroup/test/2021-02-10_01-53-53/split_pred/{split}/semantic' # TODO
    scene_ids_file = os.path.join(cfg.SCANNETV2_PATH.meta_data, f'scannetv2_{split}.txt')
    scene_ids = [scene_id.rstrip() for scene_id in open(scene_ids_file)]
    
    for scene_id in scene_ids:
        print(scene_id)
        gt_sem_ply = os.path.join(scan_dir, split, scene_id, f'{scene_id}_vh_clean_2.labels.ply')
        pred_sem_ply = os.path.join(pred_dir, f'{scene_id}.ply')
        pred_sem_labels = np.loadtxt(os.path.join(pred_dir, f'{scene_id}.txt'))
        
        with open(gt_sem_ply, 'rb') as f:
            plydata = PlyData.read(f)
            num_verts = plydata['vertex'].count
            assert num_verts == len(pred_sem_labels)
            points = np.zeros(shape=[num_verts, 3], dtype=np.float)
            points[:,0] = plydata['vertex'].data['x']
            points[:,1] = plydata['vertex'].data['y']
            points[:,2] = plydata['vertex'].data['z']
            
        write_ply_colorful(points, pred_sem_labels, pred_sem_ply, colormap=SCANNET_COLOR_MAP)
        
        
def generate_gt_inst_ply(args):
    split = args.split
    scan_dir = cfg.SCANNETV2_PATH.split_scans
    inst_data_dir = cfg.SCANNETV2_PATH.split_data
    output_dir = f'/project/3dlg-hcvc/dense-scanrefer/scannet/rgb_instance/gt/{split}' # TODO
    os.makedirs(output_dir, exist_ok=True)
    scene_ids_file = os.path.join(cfg.SCANNETV2_PATH.meta_data, f'scannetv2_{split}.txt')
    scene_ids = [scene_id.rstrip() for scene_id in open(scene_ids_file)]
    DONOTCARE_CLASS_IDS = np.array([1, 2, 22]) # exclude wall, floor and ceiling
    
    for scene_id in scene_ids:
        print(scene_id)
        # rgb_ply = os.path.join(scan_dir, split, scene_id, f'{scene_id}_vh_clean_2.ply')
        gt_sem_ply = os.path.join(scan_dir, split, scene_id, f'{scene_id}_vh_clean_2.labels.ply')
        gt_inst_file = os.path.join(inst_data_dir, split, f'{scene_id}.pth')
        rgb_inst_ply = os.path.join(output_dir, f'{scene_id}.ply')
        
        # rgb_data = PlyData.read(open(rgb_ply, 'rb'))
        scannet_data = torch.load(gt_inst_file)
        points = scannet_data['aligned_mesh'][:, :3].astype(np.float)
        colors = scannet_data['aligned_mesh'][:, 3:6].astype(np.uint8)
        num_verts = len(points)
        instance_ids = scannet_data['instance_ids']
        sem_data = PlyData.read(open(gt_sem_ply, 'rb'))
        sem_labels = np.array(sem_data['vertex']['label'])
        assert num_verts == len(sem_labels) and num_verts == len(instance_ids)
        
        invalid_vert_idx = np.in1d(sem_labels, DONOTCARE_CLASS_IDS)
        instance_ids[invalid_vert_idx] = -1 # -1: points are not assigned to any objects
        
        # points = np.zeros(shape=[num_verts, 3], dtype=np.float)
        # colors = np.zeros(shape=[num_verts, 3], dtype=np.uint)
        # points[:,0] = rgb_data['vertex'].data['x']
        # points[:,1] = rgb_data['vertex'].data['y']
        # points[:,2] = rgb_data['vertex'].data['z']
        # colors[:,0] = rgb_data['vertex'].data['red']
        # colors[:,1] = rgb_data['vertex'].data['green']
        # colors[:,2] = rgb_data['vertex'].data['blue']
        
        unique_inst_ids = np.unique(instance_ids)
        colormap = [plt.cm.rainbow(i/(len(unique_inst_ids)+1)) for i in range(len(unique_inst_ids))]
        
        for i, inst_id in enumerate(unique_inst_ids):
            if inst_id == -1: continue
            inst_vert_idx = instance_ids == inst_id
            colors[inst_vert_idx, :] = (np.array(colormap[i][:3]) * 255).astype(np.uint8)
        
        write_ply_rgb(points, colors, rgb_inst_ply)
        
        
def visualize_pred_instance(filename, mesh, instance_ids, sem_labels):
    points = mesh[:, :3].astype(np.float)
    colors = (mesh[:, 3:6]*256.0 + MEAN_COLOR_RGB).astype(np.uint8)
    sem_labels = sem_labels.astype(np.int)
    instance_ids = instance_ids.astype(np.int)
    num_verts = len(points)
    assert num_verts == len(sem_labels) and num_verts == len(instance_ids)
    
    unique_inst_ids = np.unique(instance_ids)
    colormap = [plt.cm.rainbow(i/(len(unique_inst_ids)+1)) for i in range(len(unique_inst_ids))]
    
    for i, inst_id in enumerate(unique_inst_ids):
        if inst_id == -1: continue
        inst_vert_idx = instance_ids == inst_id
        assert len(np.unique(sem_labels[inst_vert_idx])) == 1
        inst_pred_class = sem_labels[inst_vert_idx][0]
        if inst_pred_class not in [1, 2, 22]:
            colors[inst_vert_idx, :] = (np.array(colormap[i][:3]) * 255).astype(np.uint8)
    
    write_ply_rgb(points, colors, filename)
        
        
def generate_pred_inst_ply(args):
    split = args.split
    use_checkpoint = args.use_checkpoint
    # scan_dir = cfg.SCANNETV2_PATH.split_scans
    data_dir = cfg.SCANNETV2_PATH.split_data
    pred_dir = f'/local-scratch/qiruiw/research/dense-scanrefer/log/scannet/pointgroup/test/{use_checkpoint}/split_pred' # TODO
    output_dir = f'/project/3dlg-hcvc/dense-scanrefer/scannet/rgb_instance/pred_{use_checkpoint}/{split}' # TODO
    os.makedirs(output_dir, exist_ok=True)
    scene_ids_file = os.path.join(cfg.SCANNETV2_PATH.meta_data, f'scannetv2_{split}.txt')
    scene_ids = [scene_id.rstrip() for scene_id in open(scene_ids_file)]
    DONOTCARE_CLASS_IDS = np.array([1, 2, 22]) # exclude wall, floor and ceiling
    
    for scene_id in scene_ids:
        print(scene_id)
        # rgb_ply = os.path.join(scan_dir, split, scene_id, f'{scene_id}_vh_clean_2.ply')
        rgb_file = os.path.join(data_dir, split, f'{scene_id}.pth')
        pred_sem_file = os.path.join(pred_dir, split, f'semantic/{scene_id}.txt')
        pred_inst_file = os.path.join(pred_dir, split, f'instance/{scene_id}.cluster_ids.txt')
        rgb_inst_ply = os.path.join(output_dir, f'{scene_id}.ply')
        
        # rgb_data = PlyData.read(open(rgb_ply, 'rb'))
        scannet_data = torch.load(rgb_file)
        points = scannet_data['aligned_mesh'][:, :3].astype(np.float)
        colors = scannet_data['aligned_mesh'][:, 3:6].astype(np.uint8)
        num_verts = len(points)
        # num_verts = rgb_data['vertex'].count
        sem_labels = np.loadtxt(pred_sem_file, dtype=np.int)
        instance_ids = np.loadtxt(pred_inst_file, dtype=np.int)
        assert num_verts == len(sem_labels) and num_verts == len(instance_ids)
        
        # points = np.zeros(shape=[num_verts, 3], dtype=np.float)
        # colors = np.zeros(shape=[num_verts, 3], dtype=np.uint)
        # points[:,0] = rgb_data['vertex'].data['x']
        # points[:,1] = rgb_data['vertex'].data['y']
        # points[:,2] = rgb_data['vertex'].data['z']
        # colors[:,0] = rgb_data['vertex'].data['red']
        # colors[:,1] = rgb_data['vertex'].data['green']
        # colors[:,2] = rgb_data['vertex'].data['blue']
        
        unique_inst_ids = np.unique(instance_ids)
        colormap = [plt.cm.rainbow(i/(len(unique_inst_ids)+1)) for i in range(len(unique_inst_ids))]
        
        for i, inst_id in enumerate(unique_inst_ids):
            if inst_id == -1: continue
            inst_vert_idx = instance_ids == inst_id
            assert len(np.unique(sem_labels[inst_vert_idx])) == 1
            inst_pred_class = sem_labels[inst_vert_idx][0]
            if inst_pred_class not in [1, 2, 22]:
                colors[inst_vert_idx, :] = (np.array(colormap[i][:3]) * 255).astype(np.uint8)
        
        write_ply_rgb(points, colors, rgb_inst_ply)
        
        
def visualize_crop_bboxes(filename, mesh, crop_bboxes):
    num_instances = len(crop_bboxes)
    bbox_verts_all = []
    bbox_colors_all = []
    bbox_indices_all = []
    for i in range(num_instances):
        crop_bbox = crop_bboxes[i][:6]
        crop_bbox_verts, crop_bbox_colors, crop_bbox_indices = write_cylinder_bbox(crop_bbox, mode=1)
        crop_bbox_indices = [ind + len(bbox_verts_all) for ind in crop_bbox_indices]
        bbox_verts_all.extend(crop_bbox_verts)
        bbox_colors_all.extend(crop_bbox_colors)
        bbox_indices_all.extend(crop_bbox_indices)
        
    write_ply_rgb_face(np.concatenate([np.array(bbox_verts_all), mesh[:, :3]]),
                        np.concatenate([np.array(bbox_colors_all), mesh[:, 3:6]]),
                        np.array(bbox_indices_all),
                        filename)
        
        
def generate_gt_bbox_ply(args):
    split = args.split
    data_dir = cfg.SCANNETV2_PATH.split_data
    bbox_dir = '/local-scratch/qiruiw/research/ScanRefer/data/scannet/scannet_data' # TODO
    output_dir = f'/project/3dlg-hcvc/dense-scanrefer/scannet/rgb_bbox/gt/{split}' # TODO
    os.makedirs(output_dir, exist_ok=True)
    scene_ids_file = f'/local-scratch/qiruiw/research/dense-scanrefer/data/scanrefer/split_data/ScanRefer_filtered_{split}.txt'
    scene_ids = [scene_id.rstrip() for scene_id in open(scene_ids_file)]
    
    for scene_id in scene_ids:
        print(scene_id)
        # rgb_ply = os.path.join(scan_dir, split, scene_id, f'{scene_id}_vh_clean_2.ply')
        rgb_file = os.path.join(data_dir, split, f'{scene_id}.pth')
        bbox_file = os.path.join(bbox_dir, f"{scene_id}_aligned_bbox.npy")
        rgb_bbox_ply = os.path.join(output_dir, f'{scene_id}.ply')
        
        # rgb_data = PlyData.read(open(rgb_ply, 'rb'))
        scannet_data = torch.load(rgb_file)
        mesh = scannet_data['aligned_mesh'][:, :6]
        gt_bboxes = np.load(bbox_file)
        num_instances = len(gt_bboxes)
        
        bbox_verts_all = []
        bbox_colors_all = []
        bbox_indices_all = []
        for i in range(num_instances):
            gt_bbox = gt_bboxes[i]
            if not np.any(gt_bbox): continue
            gt_bbox = gt_bbox[:6]
            gt_bbox_verts, gt_bbox_colors, gt_bbox_indices = write_cylinder_bbox(gt_bbox, mode=0)
            gt_bbox_indices = [ind + len(bbox_verts_all) for ind in gt_bbox_indices]
            bbox_verts_all.extend(gt_bbox_verts)
            bbox_colors_all.extend(gt_bbox_colors)
            bbox_indices_all.extend(gt_bbox_indices)
            
        write_ply_rgb_face(np.concatenate([np.array(bbox_verts_all), mesh[:, :3]]),
                            np.concatenate([np.array(bbox_colors_all), mesh[:, 3:]]),
                            np.array(bbox_indices_all),
                            rgb_bbox_ply)
            

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-s', '--split', type=str, default='val', help='specify data split')
    parser.add_argument('-c', '--use_checkpoint', type=str, default='', help='specify checkpoint')
    parser.add_argument('-t', '--task', type=str, default='', help='specify task: semantic | instance | detection')
    args = parser.parse_args()
    
    generate_rgb_ply(args)
    # generate_gt_sem_ply(args)
    # generate_pred_sem_ply(args)
    # generate_gt_inst_ply(args)
    # generate_pred_inst_ply(args)
    # generate_gt_bbox_ply(args)