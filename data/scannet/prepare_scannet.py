'''
REFERENCE TO https://github.com/facebookresearch/votenet/blob/master/scannet/load_scannet_data.py
'''

import glob, plyfile, json, argparse, os
from omegaconf import OmegaConf
import numpy as np
import multiprocessing as mp
import torch
import scannet_utils
from plyfile import PlyData

DONOTCARE_CLASS_IDS = np.array([1, 2, 22]) # exclude wall, floor and ceiling

LABEL_MAP_FILE = 'meta_data/scannetv2-labels.combined.tsv'
MEAN_COLOR_RGB = np.array([109.8, 97.2, 83.8])


# scannet_utils.read_label_mapping(LABEL_MAP_FILE, label_from='raw_category', label_to='nyu40id')
# scannet_utils.read_label_mapping(LABEL_MAP_FILE, label_from='nyu40class', label_to='nyu40id')

### Map relevant classes to {0,1,...,19}, and ignored classes to -1
remapper = np.ones(150) * (-1)
for label, nyu40id in enumerate([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 14, 16, 24, 28, 33, 34, 36, 39]):
    remapper[nyu40id] = label
# IGNORE_CLASS_IDS = np.array([0, 1]) # wall and floor after remapping


def read_mesh_file(mesh_file):
    mesh = scannet_utils.read_mesh_vertices_rgb_normal(mesh_file) #（num_verts, 9) xyz+rgb+normal
    # mesh[:, 3:6] = mesh[:, 3:6] / 127.5 - 1 # substract rgb mean
    # mesh[:, 3:6] = (mesh[:, 3:6] - MEAN_COLOR_RGB) / 256.0 #TODO: should move to dataset
    return mesh


def read_axis_align_matrix(meta_file):
    lines = open(meta_file).readlines()
    axis_align_matrix = None
    for line in lines:
        if 'axisAlignment' in line:
            axis_align_matrix = [float(x) for x in line.rstrip().strip('axisAlignment = ').split(' ')]
    if axis_align_matrix:
        axis_align_matrix = np.array(axis_align_matrix).reshape((4,4))
    return axis_align_matrix


def align_mesh_vertices(mesh, axis_align_matrix):
    aligned_mesh = np.copy(mesh)
    if axis_align_matrix is not None:
        homo_ones = np.ones((mesh.shape[0], 1))
        aligned_mesh[:, 0:3] = np.dot(np.concatenate([mesh[:, 0:3], homo_ones], 1), axis_align_matrix.transpose())[:, :3]
    return aligned_mesh


def read_label_file(label_file):
    with open(label_file, 'rb') as f:
        plydata = PlyData.read(f)
        sem_labels = np.array(plydata['vertex']['label']) #nyu40 
        # raw_labels[raw_labels > 40] = 40 # HACK !! # 0: unannotated
        # sem_labels = remapper[raw_labels]
    return sem_labels


def read_agg_file(agg_file):
    objectId2segs = {}
    label2segs = {}
    with open(agg_file) as json_data:
        data = json.load(json_data)
        # objectId = 0
        for group in data['segGroups']:
            objectId = group['objectId'] # starts from 0
            label = group['label']
            segs = group['segments']
            if label in ['wall', 'floor', 'ceiling']:
                # print('ignore wall, floor or ceiling')
                continue
            else:
                objectId2segs[objectId] = segs
                if label in label2segs:
                    label2segs[label].extend(segs)
                else:
                    label2segs[label] = segs
                # objectId += 1

    if agg_file.split('/')[-2] == 'scene0217_00':
        objectIds = sorted(objectId2segs.keys())
        # if objectId2segs[0] == objectId2segs[objectIds[len(objectId2segs)//2]]:
        print('HACK scene0217_00')
        objectId2segs = {objectId: objectId2segs[objectId] for objectId in objectIds[:len(objectId2segs)//2]}

    return objectId2segs, label2segs


def read_seg_file(seg_file):
    seg2verts = {}
    with open(seg_file) as json_data:
        data = json.load(json_data)
        num_verts = len(data['segIndices'])
        for vert, seg in enumerate(data['segIndices']):
            if seg in seg2verts:
                seg2verts[seg].append(vert)
            else:
                seg2verts[seg] = [vert]
    return seg2verts, num_verts


def get_instance_ids(objectId2segs, seg2verts, sem_labels):
    objectId2labelId = {}
    instance_ids = np.ones(shape=(len(sem_labels))) * -1 # -1: points are not assigned to any objects ( objectId starts from 0)
    for objectId, segs in objectId2segs.items():
        for seg in segs:
            verts = seg2verts[seg]
            instance_ids[verts] = objectId
        if objectId not in objectId2labelId:
            objectId2labelId[objectId] = sem_labels[verts][0]
        #assert(len(np.unique(sem_labels[pointids])) == 1)
    return instance_ids, objectId2labelId


def get_instance_bboxes(mesh, instance_ids, objectId2labelId):
    num_instances = max(objectId2labelId.keys()) + 1
    instance_bboxes = np.zeros((num_instances, 8)) # (cx, cy, cz, dx, dy, dz, ins_label, objectId)
    for objectId in objectId2labelId:
        ins_label = objectId2labelId[objectId] # nyu40id
        obj_pc = mesh[instance_ids==objectId, 0:3] # 
        if len(obj_pc) == 0: continue
        xmin = np.min(obj_pc[:,0])
        ymin = np.min(obj_pc[:,1])
        zmin = np.min(obj_pc[:,2])
        xmax = np.max(obj_pc[:,0])
        ymax = np.max(obj_pc[:,1])
        zmax = np.max(obj_pc[:,2])
        bbox = np.array([(xmin+xmax)/2, (ymin+ymax)/2, (zmin+zmax)/2, xmax-xmin, ymax-ymin, zmax-zmin, ins_label, objectId]) 
        instance_bboxes[objectId,:] = bbox
    return instance_bboxes


def export(scene, cfg):
    mesh_file = os.path.join(cfg.SCANNETV2_PATH.raw_scans, scene, scene + '_vh_clean_2.ply')
    label_file = os.path.join(cfg.SCANNETV2_PATH.raw_scans, scene, scene + '_vh_clean_2.labels.ply')
    agg_file = os.path.join(cfg.SCANNETV2_PATH.raw_scans, scene, scene + '.aggregation.json')
    seg_file = os.path.join(cfg.SCANNETV2_PATH.raw_scans, scene, scene + '_vh_clean_2.0.010000.segs.json')
    meta_file = os.path.join(cfg.SCANNETV2_PATH.raw_scans, scene, scene + '.txt')

    # read mesh_file
    mesh = read_mesh_file(mesh_file) #（num_verts, 9) xyz+rgb+normal
    num_verts = mesh.shape[0]
    # read meta_file
    axis_align_matrix = read_axis_align_matrix(meta_file)
    aligned_mesh = align_mesh_vertices(mesh, axis_align_matrix) #（num_verts, 9) aligned_xyz+rgb+normal

    # mesh[:, :3] -= mesh[:, :3].mean(0) # substract xyz mean #TODO: should move to dataset if necessary
    # aligned_mesh[:, :3] -= aligned_mesh[:, :3].mean(0) # substract aligned xyz mean

    if os.path.isfile(agg_file):
        # read label_file
        sem_labels = read_label_file(label_file) 
        # read seg_file
        seg2verts, num = read_seg_file(seg_file)
        assert num_verts == num
        # read agg_file
        objectId2segs, label2segs = read_agg_file(agg_file)
        # get instance labels
        instance_ids, objectId2labelId = get_instance_ids(objectId2segs, seg2verts, sem_labels)
        # get instance bounding boxes
        instance_bboxes = get_instance_bboxes(mesh, instance_ids, objectId2labelId)
        # get aligned instance bounding boxes
        aligned_instance_bboxes = get_instance_bboxes(aligned_mesh, instance_ids, objectId2labelId)
    else:
        # use zero as placeholders for the test scene
        print("use placeholders")
        sem_labels = np.zeros(shape=(num_verts), dtype=np.uint32) # 0: unannotated
        instance_ids = np.ones(shape=(num_verts), dtype=np.uint32) * -1 # -1: unannotated
        instance_bboxes = np.zeros((1, 8))
        aligned_instance_bboxes = np.zeros((1, 8))

    return mesh, aligned_mesh, sem_labels, instance_ids, instance_bboxes, aligned_instance_bboxes


def process_one_scan(scan, cfg):
    mesh, aligned_mesh, sem_labels, instance_ids, instance_bboxes, aligned_instance_bboxes = export(scan, cfg)
    # print('Num of instances: ', len(np.unique(instance_ids)))
    sem_labels = remapper[sem_labels] # nyu40id -> {0, 1, ..., 19}

    if instance_bboxes.shape[0] > 1:
        num_instances = len(np.unique(instance_ids))
        # print('Num of instances: ', num_instances)

        bbox_mask = np.logical_not(np.in1d(instance_bboxes[:,-2], DONOTCARE_CLASS_IDS)) # match the mesh2cap; not care wall, floor and ceiling for instances
        instance_bboxes = instance_bboxes[bbox_mask,:]
        aligned_instance_bboxes = aligned_instance_bboxes[bbox_mask,:]
        print('Num of care instances: ', instance_bboxes.shape[0])
    else:
        print("No semantic/instance annotation for test scenes")


    torch.save({'mesh': mesh, 'aligned_mesh': aligned_mesh, 'sem_labels': sem_labels, 'instance_ids': instance_ids, 'instance_bboxes': instance_bboxes, 'aligned_instance_bboxes': aligned_instance_bboxes}, os.path.join(cfg.SCANNETV2_PATH.split_data, cfg.split, scan+'.pth'))
    
    
def process_all_scans(cfg):
    SCAN_NAMES = sorted([line.rstrip() for line in open(f'meta_data/scannetv2_{cfg.split}.txt')])     
    
    for scan in SCAN_NAMES:
        print(scan)
        process_one_scan(scan, cfg)


if __name__=='__main__':    
    parser = argparse.ArgumentParser()
    parser.add_argument('-s', '--split', help='data split (train / val / test)', default='train')
    parser.add_argument('-c', '--cfg', help='scannet configuration YAML file', default='../../conf/path.yaml')
    opt = parser.parse_args()

    cfg = OmegaConf.load(opt.cfg)
    cfg.split = opt.split
    
    os.makedirs(os.path.join(cfg.SCANNETV2_PATH.split_data, cfg.split), exist_ok=True)

    # process_one_scan('scene0217_00', cfg)

    print(f'data split: {cfg.split}')
    process_all_scans(cfg)