import os
import json
import pickle
import argparse

import numpy as np

from tqdm import tqdm
from plyfile import PlyData, PlyElement

# data
SCANNET_LIST = sorted(os.listdir("/canis/Datasets/ScanNet/public/v1/scans/"))
SCANNET_ROOT = "/canis/Datasets/ScanNet/public/v2/scans/" # TODO point this to your scannet data
SCANNET_MESH = os.path.join(SCANNET_ROOT, "{}/{}_vh_clean_2.ply") # scene_id, scene_id 
SCANNET_MESH_HIGH = os.path.join(SCANNET_ROOT, "{}/{}_vh_clean.ply") # scene_id, scene_id 
SCANNET_META = os.path.join(SCANNET_ROOT, "{}/{}.txt") # scene_id, scene_id 

OUTPUT_ROOT = "outputs/"

def write_ply(verts, colors, indices, output_file):
    if colors is None:
        colors = np.zeros_like(verts)
    if indices is None:
        indices = []

    file = open(output_file, 'w')
    file.write('ply \n')
    file.write('format ascii 1.0\n')
    file.write('element vertex {:d}\n'.format(len(verts)))
    file.write('property float x\n')
    file.write('property float y\n')
    file.write('property float z\n')
    file.write('property uchar red\n')
    file.write('property uchar green\n')
    file.write('property uchar blue\n')
    file.write('element face {:d}\n'.format(len(indices)))
    file.write('property list uchar uint vertex_indices\n')
    file.write('end_header\n')
    for vert, color in zip(verts, colors):
        file.write("{:f} {:f} {:f} {:d} {:d} {:d}\n".format(vert[0], vert[1], vert[2] , int(color[0]*255), int(color[1]*255), int(color[2]*255)))
    for ind in indices:
        file.write('3 {:d} {:d} {:d}\n'.format(ind[0], ind[1], ind[2]))
    file.close()

def write_bbox(bbox, mode, output_file):

    def create_cylinder_mesh(radius, p0, p1, stacks=10, slices=10):
        
        import math

        def compute_length_vec3(vec3):
            return math.sqrt(vec3[0]*vec3[0] + vec3[1]*vec3[1] + vec3[2]*vec3[2])
        
        def rotation(axis, angle):
            rot = np.eye(4)
            c = np.cos(-angle)
            s = np.sin(-angle)
            t = 1.0 - c
            axis /= compute_length_vec3(axis)
            x = axis[0]
            y = axis[1]
            z = axis[2]
            rot[0,0] = 1 + t*(x*x-1)
            rot[0,1] = z*s+t*x*y
            rot[0,2] = -y*s+t*x*z
            rot[1,0] = -z*s+t*x*y
            rot[1,1] = 1+t*(y*y-1)
            rot[1,2] = x*s+t*y*z
            rot[2,0] = y*s+t*x*z
            rot[2,1] = -x*s+t*y*z
            rot[2,2] = 1+t*(z*z-1)
            return rot


        verts = []
        indices = []
        diff = (p1 - p0).astype(np.float32)
        height = compute_length_vec3(diff)
        for i in range(stacks+1):
            for i2 in range(slices):
                theta = i2 * 2.0 * math.pi / slices
                pos = np.array([radius*math.cos(theta), radius*math.sin(theta), height*i/stacks])
                verts.append(pos)
        for i in range(stacks):
            for i2 in range(slices):
                i2p1 = math.fmod(i2 + 1, slices)
                indices.append( np.array([(i + 1)*slices + i2, i*slices + i2, i*slices + i2p1], dtype=np.uint32) )
                indices.append( np.array([(i + 1)*slices + i2, i*slices + i2p1, (i + 1)*slices + i2p1], dtype=np.uint32) )
        transform = np.eye(4)
        va = np.array([0, 0, 1], dtype=np.float32)
        vb = diff
        vb /= compute_length_vec3(vb)
        axis = np.cross(vb, va)
        angle = np.arccos(np.clip(np.dot(va, vb), -1, 1))
        if angle != 0:
            if compute_length_vec3(axis) == 0:
                dotx = va[0]
                if (math.fabs(dotx) != 1.0):
                    axis = np.array([1,0,0]) - dotx * va
                else:
                    axis = np.array([0,1,0]) - va[1] * va
                axis /= compute_length_vec3(axis)
            transform = rotation(axis, -angle)
        transform[:3,3] += p0
        verts = [np.dot(transform, np.array([v[0], v[1], v[2], 1.0])) for v in verts]
        verts = [np.array([v[0], v[1], v[2]]) / v[3] for v in verts]
            
        return verts, indices

    def get_bbox_edges(bbox_min, bbox_max):
        def get_bbox_verts(bbox_min, bbox_max):
            verts = [
                np.array([bbox_min[0], bbox_min[1], bbox_min[2]]),
                np.array([bbox_max[0], bbox_min[1], bbox_min[2]]),
                np.array([bbox_max[0], bbox_max[1], bbox_min[2]]),
                np.array([bbox_min[0], bbox_max[1], bbox_min[2]]),

                np.array([bbox_min[0], bbox_min[1], bbox_max[2]]),
                np.array([bbox_max[0], bbox_min[1], bbox_max[2]]),
                np.array([bbox_max[0], bbox_max[1], bbox_max[2]]),
                np.array([bbox_min[0], bbox_max[1], bbox_max[2]])
            ]
            return verts

        box_verts = get_bbox_verts(bbox_min, bbox_max)
        edges = [
            (box_verts[0], box_verts[1]),
            (box_verts[1], box_verts[2]),
            (box_verts[2], box_verts[3]),
            (box_verts[3], box_verts[0]),

            (box_verts[4], box_verts[5]),
            (box_verts[5], box_verts[6]),
            (box_verts[6], box_verts[7]),
            (box_verts[7], box_verts[4]),

            (box_verts[0], box_verts[4]),
            (box_verts[1], box_verts[5]),
            (box_verts[2], box_verts[6]),
            (box_verts[3], box_verts[7])
        ]
        return edges

    radius = 0.03
    offset = [0,0,0]
    verts = []
    indices = []
    colors = []

    box_min = np.min(bbox, axis=0)
    box_max = np.max(bbox, axis=0)
    palette = {
        0: [0, 255, 0], # gt
        1: [0, 0, 255]  # pred
    }
    chosen_color = palette[mode]
    edges = get_bbox_edges(box_min, box_max)
    for k in range(len(edges)):
        cyl_verts, cyl_ind = create_cylinder_mesh(radius, edges[k][0], edges[k][1])
        cur_num_verts = len(verts)
        cyl_color = [[c / 255 for c in chosen_color] for _ in cyl_verts]
        cyl_verts = [x + offset for x in cyl_verts]
        cyl_ind = [x + cur_num_verts for x in cyl_ind]
        verts.extend(cyl_verts)
        indices.extend(cyl_ind)
        colors.extend(cyl_color)

    write_ply(verts, colors, indices, output_file)

def read_mesh(filename):
    """ read XYZ for each vertex.
    """

    assert os.path.isfile(filename)
    with open(filename, 'rb') as f:
        plydata = PlyData.read(f)
        num_verts = plydata['vertex'].count
        vertices = np.zeros(shape=[num_verts, 6], dtype=np.float32)
        vertices[:,0] = plydata['vertex'].data['x']
        vertices[:,1] = plydata['vertex'].data['y']
        vertices[:,2] = plydata['vertex'].data['z']
        vertices[:,3] = plydata['vertex'].data['red']
        vertices[:,4] = plydata['vertex'].data['green']
        vertices[:,5] = plydata['vertex'].data['blue']

    return vertices, plydata['face']

def export_mesh(vertices, faces):
    new_vertices = []
    for i in range(vertices.shape[0]):
        new_vertices.append(
            (
                vertices[i][0],
                vertices[i][1],
                vertices[i][2],
                vertices[i][3],
                vertices[i][4],
                vertices[i][5],
            )
        )

    vertices = np.array(
        new_vertices,
        dtype=[
            ("x", np.dtype("float32")), 
            ("y", np.dtype("float32")), 
            ("z", np.dtype("float32")),
            ("red", np.dtype("uint8")),
            ("green", np.dtype("uint8")),
            ("blue", np.dtype("uint8"))
        ]
    )

    vertices = PlyElement.describe(vertices, "vertex")
    
    return PlyData([vertices, faces])

def align_mesh(scene_id):
    vertices, faces = read_mesh(SCANNET_MESH.format(scene_id, scene_id))
    for line in open(SCANNET_META.format(scene_id, scene_id)).readlines():
        if 'axisAlignment' in line:
            axis_align_matrix = np.array([float(x) for x in line.rstrip().strip('axisAlignment = ').split(' ')]).reshape((4, 4))
            break
    
    # align
    pts = np.ones((vertices.shape[0], 4))
    pts[:, :3] = vertices[:, :3]
    pts = np.dot(pts, axis_align_matrix.T)
    vertices[:, :3] = pts[:, :3]

    mesh = export_mesh(vertices, faces)

    return mesh

def align_high_mesh(scene_id):
    vertices, faces = read_mesh(SCANNET_MESH_HIGH.format(scene_id, scene_id))
    for line in open(SCANNET_META.format(scene_id, scene_id)).readlines():
        if 'axisAlignment' in line:
            axis_align_matrix = np.array([float(x) for x in line.rstrip().strip('axisAlignment = ').split(' ')]).reshape((4, 4))
            break
    
    # align
    pts = np.ones((vertices.shape[0], 4))
    pts[:, :3] = vertices[:, :3]
    pts = np.dot(pts, axis_align_matrix.T)
    vertices[:, :3] = pts[:, :3]

    mesh = export_mesh(vertices, faces)

    return mesh

def visualize_aligned_mesh(scene_id, root):
    mesh = align_mesh(scene_id)
    mesh.write(os.path.join(root, "mesh.ply"))

def visualize_aligned_high_mesh(scene_id, root):
    high_mesh = align_high_mesh(scene_id)
    high_mesh.write(os.path.join(root, "high_mesh.ply"))

def load_data(args):
    # pred_path = os.path.join(OUTPUT_ROOT, args.folder, "pred_val_0.json")
    pred_path = os.path.join(OUTPUT_ROOT, args.folder, "extra_caption.json")
    with open(pred_path) as f:
        predictions = json.load(f)

    return predictions

def visualize(args):
    print("loading data...")
    predictions = load_data(args) # key: scene_id -> object_id -> ann_id

    root = os.path.join(OUTPUT_ROOT, args.folder, "vis_captioning")
    os.makedirs(root, exist_ok=True)

    for key, value in tqdm(predictions.items()):
        scene_id, object_id = key.split("|")

        scene_root = os.path.join(root, scene_id)
        os.makedirs(scene_root, exist_ok=True)

        # print("exporting axis-aligned ScanNet meshes...")
        mesh_path = os.path.join(scene_root, "mesh.ply")
        if not os.path.exists(mesh_path): visualize_aligned_mesh(scene_id, scene_root)

        pred_bbox = np.array(value[2])
        pred_path = os.path.join(scene_root, "pred-{}.ply".format(object_id))
        write_bbox(pred_bbox, 1, pred_path)

        gt_bbox = np.array(value[3])
        gt_path = os.path.join(scene_root, "gt-{}.ply".format(object_id))
        write_bbox(gt_bbox, 0, gt_path)

    print("done!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-f", "--folder", type=str, required=True, help="path to folder with model")
    args = parser.parse_args()

    visualize(args)