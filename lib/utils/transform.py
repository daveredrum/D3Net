import numpy as np
import scipy.ndimage
import scipy.interpolate


##############################
#           Rotation         #
##############################

def jitter(intensity=0.1):
    '''
    params: 
        the intensity of jittering
    return: 
        3x3 jitter matrix
    '''
    return np.eye(3) + np.random.randn(3, 3) * intensity


def flip(axis=0, random=False):
    '''
    flip the specified axis
    params: 
        axis 0:x, 1:y, 2:z
    return:
        3x3 flip matrix
    '''
    m = np.eye(3)
    m[axis][axis] *= -1 if not random else np.random.randint(0, 2) * 2 - 1
    return m


def rotx(t):
    '''
    Rotation about the x-axis. counter-clockwise
    '''
    c = np.cos(t)
    s = np.sin(t)
    return np.array([[1,  0,  0],
                    [0,  c,  -s],
                    [0,  s,  c]])


def roty(t):
    '''
    Rotation about the y-axis. clockwise
    '''
    c = np.cos(t)
    s = np.sin(t)
    return np.array([[c,  0,  s],
                    [0,  1,  0],
                    [-s, 0,  c]])
    

def roty_batch(t):
    """Get batch rotation matrix about the y-axis.
    t: (x1,x2,...xn)
    return: (x1,x2,...,xn,3,3)
    """
    input_shape = t.shape
    output = np.zeros(tuple(list(input_shape)+[3,3]))
    c = np.cos(t)
    s = np.sin(t)
    output[...,0,0] = c
    output[...,0,2] = s
    output[...,1,1] = 1
    output[...,2,0] = -s
    output[...,2,2] = c
    return output


def rotz(t):
    '''
    Rotation about the z-axis. counter-clockwise
    '''
    c = np.cos(t)
    s = np.sin(t)
    return np.array([[c, -s,  0],
                     [s,  c,  0],
                     [0,  0,  1]])


def elastic(x, gran, mag):
    '''
    Refers to https://github.com/Jia-Research-Lab/PointGroup/blob/master/data/scannetv2_inst.py
    '''
    blur0 = np.ones((3, 1, 1)).astype('float32') / 3
    blur1 = np.ones((1, 3, 1)).astype('float32') / 3
    blur2 = np.ones((1, 1, 3)).astype('float32') / 3

    bb = np.abs(x).max(0).astype(np.int32)//gran + 3
    noise = [np.random.randn(bb[0], bb[1], bb[2]).astype('float32') for _ in range(3)]
    noise = [scipy.ndimage.filters.convolve(n, blur0, mode='constant', cval=0) for n in noise]
    noise = [scipy.ndimage.filters.convolve(n, blur1, mode='constant', cval=0) for n in noise]
    noise = [scipy.ndimage.filters.convolve(n, blur2, mode='constant', cval=0) for n in noise]
    noise = [scipy.ndimage.filters.convolve(n, blur0, mode='constant', cval=0) for n in noise]
    noise = [scipy.ndimage.filters.convolve(n, blur1, mode='constant', cval=0) for n in noise]
    noise = [scipy.ndimage.filters.convolve(n, blur2, mode='constant', cval=0) for n in noise]
    ax = [np.linspace(-(b-1)*gran, (b-1)*gran, b) for b in bb]
    interp = [scipy.interpolate.RegularGridInterpolator(ax, n, bounds_error=0, fill_value=0) for n in noise]
    def g(x_):
        return np.hstack([i(x_)[:,None] for i in interp])
    return x + g(x) * mag

