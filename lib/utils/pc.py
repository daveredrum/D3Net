import numpy as np
import scipy.ndimage
import scipy.interpolate

# Point cloud IO
from plyfile import PlyData, PlyElement

# Mesh IO
import trimesh
import matplotlib.pyplot as plt


########################
# point cloud sampling #
########################

def random_sampling(pc, num_sample, replace=None, return_choices=False):
    """ Input is NxC, output is num_samplexC
    """
    if replace is None: replace = (pc.shape[0] < num_sample)
    choices = np.random.choice(pc.shape[0], num_sample, replace=replace)
    choices.sort()
    if return_choices:
        return pc[choices], choices
    else:
        return pc[choices]


def crop(pc, max_num_point, scale):
    '''
    Crop the points such that there are at most max_num_points points
    '''
    pc_offset = pc.copy()
    valid_idxs = (pc_offset.min(1) >= 0)
    assert valid_idxs.sum() == pc.shape[0]

    max_pc_range = np.array([scale] * 3)
    pc_range = pc.max(0) - pc.min(0)
    while (valid_idxs.sum() > max_num_point):
        offset = np.clip(max_pc_range - pc_range + 0.001, None, 0) * np.random.rand(3)
        pc_offset = pc + offset
        valid_idxs = (pc_offset.min(1) >= 0) * ((pc_offset < max_pc_range).sum(1) == 3)
        max_pc_range[:2] -= 32

    return pc_offset, valid_idxs


##################
# Point cloud IO #
##################

def read_ply(filename):
    """ read XYZ point cloud from filename PLY file """
    plydata = PlyData.read(filename)
    num_verts = plydata['vertex'].count
    pc = np.zeros(shape=[num_verts, 3], dtype=np.float32)
    pc[:,0] = plydata['vertex'].data['x']
    pc[:,1] = plydata['vertex'].data['y']
    pc[:,2] = plydata['vertex'].data['z']
    return pc


def read_ply_rgb(filename):
    """ read XYZ RGB for each vertex.
    Note: RGB values are in 0-255
    """
    plydata = PlyData.read(filename)
    num_verts = plydata['vertex'].count
    pc = np.zeros(shape=[num_verts, 6], dtype=np.float32)
    pc[:,0] = plydata['vertex'].data['x']
    pc[:,1] = plydata['vertex'].data['y']
    pc[:,2] = plydata['vertex'].data['z']
    pc[:,3] = plydata['vertex'].data['red']
    pc[:,4] = plydata['vertex'].data['green']
    pc[:,5] = plydata['vertex'].data['blue']
    return pc


def write_ply(points, filename, text=True):
    """ input: Nx3, write points to filename as PLY format. """
    points = [(points[i,0], points[i,1], points[i,2]) for i in range(points.shape[0])]
    vertex = np.array(points, dtype=[('x', 'f4'), ('y', 'f4'),('z', 'f4')])
    ele = PlyElement.describe(vertex, 'vertex', comments=['vertices'])
    PlyData([ele], text=text).write(filename)


def write_ply_rgb(points, colors, filename, text=True, num_classes=None):
    """ Color (N,3) points with RGB colors (N,3) within range [0,255] as ply file """
    colors = colors.astype(int)
    points = [(points[i,0], points[i,1], points[i,2], colors[i,0], colors[i,1], colors[i,2]) for i in range(points.shape[0])]
    vertex = np.array(points, dtype=[('x', 'f4'), ('y', 'f4'),('z', 'f4'),('red', 'u1'), ('green', 'u1'),('blue', 'u1')])
    ele = PlyElement.describe(vertex, 'vertex', comments=['vertices'])
    PlyData([ele], text=text).write(filename)


def write_ply_rgb_face(points, colors, faces, filename, text=True):
    """ Color (N,3) points with RGB colors (N,3) within range [0,255] as ply file """
    colors = colors.astype(int)
    points = [(points[i,0], points[i,1], points[i,2], colors[i,0], colors[i,1], colors[i,2]) for i in range(points.shape[0])]
    faces = [((faces[i,0], faces[i,1], faces[i,2]),) for i in range(faces.shape[0])]
    vertex = np.array(points, dtype=[('x', 'f4'), ('y', 'f4'),('z', 'f4'),('red', 'u1'), ('green', 'u1'),('blue', 'u1')])
    face = np.array(faces, dtype=[('vertex_indices', 'i4', (3,))])
    ele1 = PlyElement.describe(vertex, 'vertex', comments=['vertices'])
    ele2 = PlyElement.describe(face, 'face', comments=['faces'])
    PlyData([ele1, ele2], text=text).write(filename)
    
    
def write_ply_rgb_annotated(points, colors, labels, instanceIds, filename, text=True):
    colors = colors.astype(int)
    points = [(points[i,0], points[i,1], points[i,2], colors[i,0], colors[i,1], colors[i,2]) for i in range(points.shape[0])]
    vertex = np.array(points, dtype=[('x', 'f4'), ('y', 'f4'),('z', 'f4'),('red', 'u1'), ('green', 'u1'),('blue', 'u1')])
    vertex_label = np.array(labels, dtype=[('label', 'i4')])
    vertex_instance = np.array(instanceIds, dtype=[('instance', 'i4')])
    ele1 = PlyElement.describe(vertex, 'vertex', comments=['vertices'])
    ele2 = PlyElement.describe(vertex_label, 'label', comments=['labels'])
    ele3 = PlyElement.describe(vertex_instance, 'instanceId', comments=['instanceIds'])
    PlyData([ele1, ele2, ele3], text=text).write(filename)


def write_ply_colorful(points, labels, filename, num_classes=None, colormap=None):
    """ Color (N,3) points with labels (N) within range 0 ~ num_classes-1 as ply file """
    labels = labels.astype(int)
    N = points.shape[0]
    if num_classes is None:
        num_classes = np.max(labels)+1
    else:
        assert(num_classes>np.max(labels))
    
    vertex = []
    if colormap is None:
        colormap = [ply.cm.jet(i/float(num_classes)) for i in range(num_classes)]
     
    for i in range(N):
        if labels[i] >= 0:
            c = colormap[labels[i]]
        else:
            c = [0, 0, 0]
        if c[0] < 1:
            c = [int(x*255) for x in c]
        vertex.append( (points[i,0],points[i,1],points[i,2],c[0],c[1],c[2]) )
    vertex = np.array(vertex, dtype=[('x', 'f4'), ('y', 'f4'),('z', 'f4'),('red', 'u1'), ('green', 'u1'),('blue', 'u1')])
    vertex_label = np.array(labels, dtype=[('label', 'i4')])
    
    ele1 = PlyElement.describe(vertex, 'vertex', comments=['vertices'])
    ele2 = PlyElement.describe(vertex_label, 'label', comments=['labels'])
    PlyData([ele1, ele2], text=True).write(filename)
    