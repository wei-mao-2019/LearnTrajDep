import numpy as np
import torch
from torch.autograd.variable import Variable
from utils import data_utils


def fkl(angles, parent, offset, rotInd, expmapInd):
    """
    Convert joint angles and bone lenghts into the 3d points of a person.

    adapted from
    https://github.com/una-dinosauria/human-motion-prediction/blob/master/src/forward_kinematics.py#L14

    which originaly based on expmap2xyz.m, available at
    https://github.com/asheshjain399/RNNexp/blob/7fc5a53292dc0f232867beb66c3a9ef845d705cb/structural_rnn/CRFProblems/H3.6m/mhmublv/Motion/exp2xyz.m
    Args
      angles: 99-long vector with 3d position and 3d joint angles in expmap format
      parent: 32-long vector with parent-child relationships in the kinematic tree
      offset: 96-long vector with bone lenghts
      rotInd: 32-long list with indices into angles
      expmapInd: 32-long list with indices into expmap angles
    Returns
      xyz: 32x3 3d points that represent a person in 3d space
    """

    assert len(angles) == 99

    # Structure that indicates parents for each joint
    njoints = 32
    xyzStruct = [dict() for x in range(njoints)]

    for i in np.arange(njoints):

        # if not rotInd[i]:  # If the list is empty
        #     xangle, yangle, zangle = 0, 0, 0
        # else:
        #     xangle = angles[rotInd[i][0] - 1]
        #     yangle = angles[rotInd[i][1] - 1]
        #     zangle = angles[rotInd[i][2] - 1]
        if i == 0:
            xangle = angles[0]
            yangle = angles[1]
            zangle = angles[2]
            thisPosition = np.array([xangle, yangle, zangle])
        else:
            thisPosition = np.array([0, 0, 0])

        r = angles[expmapInd[i]]

        thisRotation = data_utils.expmap2rotmat(r)

        if parent[i] == -1:  # Root node
            xyzStruct[i]['rotation'] = thisRotation
            xyzStruct[i]['xyz'] = np.reshape(offset[i, :], (1, 3)) + thisPosition
        else:
            xyzStruct[i]['xyz'] = (offset[i, :] + thisPosition).dot(xyzStruct[parent[i]]['rotation']) + \
                                  xyzStruct[parent[i]]['xyz']
            xyzStruct[i]['rotation'] = thisRotation.dot(xyzStruct[parent[i]]['rotation'])

    xyz = [xyzStruct[i]['xyz'] for i in range(njoints)]
    xyz = np.array(xyz).squeeze()
    # xyz = xyz[:, [0, 2, 1]]
    # xyz = xyz[:,[2,0,1]]

    return xyz


def _some_variables():
    """
    borrowed from
    https://github.com/una-dinosauria/human-motion-prediction/blob/master/src/forward_kinematics.py#L100

    We define some variables that are useful to run the kinematic tree

    Args
      None
    Returns
      parent: 32-long vector with parent-child relationships in the kinematic tree
      offset: 96-long vector with bone lenghts
      rotInd: 32-long list with indices into angles
      expmapInd: 32-long list with indices into expmap angles
    """

    parent = np.array([0, 1, 2, 3, 4, 5, 1, 7, 8, 9, 10, 1, 12, 13, 14, 15, 13,
                       17, 18, 19, 20, 21, 20, 23, 13, 25, 26, 27, 28, 29, 28, 31]) - 1

    offset = np.array(
        [0.000000, 0.000000, 0.000000, -132.948591, 0.000000, 0.000000, 0.000000, -442.894612, 0.000000, 0.000000,
         -454.206447, 0.000000, 0.000000, 0.000000, 162.767078, 0.000000, 0.000000, 74.999437, 132.948826, 0.000000,
         0.000000, 0.000000, -442.894413, 0.000000, 0.000000, -454.206590, 0.000000, 0.000000, 0.000000, 162.767426,
         0.000000, 0.000000, 74.999948, 0.000000, 0.100000, 0.000000, 0.000000, 233.383263, 0.000000, 0.000000,
         257.077681, 0.000000, 0.000000, 121.134938, 0.000000, 0.000000, 115.002227, 0.000000, 0.000000, 257.077681,
         0.000000, 0.000000, 151.034226, 0.000000, 0.000000, 278.882773, 0.000000, 0.000000, 251.733451, 0.000000,
         0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 99.999627, 0.000000, 100.000188, 0.000000, 0.000000,
         0.000000, 0.000000, 0.000000, 257.077681, 0.000000, 0.000000, 151.031437, 0.000000, 0.000000, 278.892924,
         0.000000, 0.000000, 251.728680, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 0.000000, 99.999888,
         0.000000, 137.499922, 0.000000, 0.000000, 0.000000, 0.000000])
    offset = offset.reshape(-1, 3)

    rotInd = [[5, 6, 4],
              [8, 9, 7],
              [11, 12, 10],
              [14, 15, 13],
              [17, 18, 16],
              [],
              [20, 21, 19],
              [23, 24, 22],
              [26, 27, 25],
              [29, 30, 28],
              [],
              [32, 33, 31],
              [35, 36, 34],
              [38, 39, 37],
              [41, 42, 40],
              [],
              [44, 45, 43],
              [47, 48, 46],
              [50, 51, 49],
              [53, 54, 52],
              [56, 57, 55],
              [],
              [59, 60, 58],
              [],
              [62, 63, 61],
              [65, 66, 64],
              [68, 69, 67],
              [71, 72, 70],
              [74, 75, 73],
              [],
              [77, 78, 76],
              []]

    expmapInd = np.split(np.arange(4, 100) - 1, 32)

    return parent, offset, rotInd, expmapInd


def _some_variables_cmu():
    """
    We define some variables that are useful to run the kinematic tree

    Args
      None
    Returns
      parent: 32-long vector with parent-child relationships in the kinematic tree
      offset: 96-long vector with bone lenghts
      rotInd: 32-long list with indices into angles
      expmapInd: 32-long list with indices into expmap angles
    """

    parent = np.array([0, 1, 2, 3, 4, 5, 6, 1, 8, 9, 10, 11, 12, 1, 14, 15, 16, 17, 18, 19, 16,
                       21, 22, 23, 24, 25, 26, 24, 28, 16, 30, 31, 32, 33, 34, 35, 33, 37]) - 1

    offset = 70 * np.array(
        [0, 0, 0, 0, 0, 0, 1.65674000000000, -1.80282000000000, 0.624770000000000, 2.59720000000000, -7.13576000000000,
         0, 2.49236000000000, -6.84770000000000, 0, 0.197040000000000, -0.541360000000000, 2.14581000000000, 0, 0,
         1.11249000000000, 0, 0, 0, -1.61070000000000, -1.80282000000000, 0.624760000000000, -2.59502000000000,
         -7.12977000000000, 0, -2.46780000000000, -6.78024000000000, 0, -0.230240000000000, -0.632580000000000,
         2.13368000000000, 0, 0, 1.11569000000000, 0, 0, 0, 0.0196100000000000, 2.05450000000000, -0.141120000000000,
         0.0102100000000000, 2.06436000000000, -0.0592100000000000, 0, 0, 0, 0.00713000000000000, 1.56711000000000,
         0.149680000000000, 0.0342900000000000, 1.56041000000000, -0.100060000000000, 0.0130500000000000,
         1.62560000000000, -0.0526500000000000, 0, 0, 0, 3.54205000000000, 0.904360000000000, -0.173640000000000,
         4.86513000000000, 0, 0, 3.35554000000000, 0, 0, 0, 0, 0, 0.661170000000000, 0, 0, 0.533060000000000, 0, 0, 0,
         0, 0, 0.541200000000000, 0, 0.541200000000000, 0, 0, 0, -3.49802000000000, 0.759940000000000,
         -0.326160000000000, -5.02649000000000, 0, 0, -3.36431000000000, 0, 0, 0, 0, 0, -0.730410000000000, 0, 0,
         -0.588870000000000, 0, 0, 0, 0, 0, -0.597860000000000, 0, 0.597860000000000])
    offset = offset.reshape(-1, 3)

    rotInd = [[6, 5, 4],
              [9, 8, 7],
              [12, 11, 10],
              [15, 14, 13],
              [18, 17, 16],
              [21, 20, 19],
              [],
              [24, 23, 22],
              [27, 26, 25],
              [30, 29, 28],
              [33, 32, 31],
              [36, 35, 34],
              [],
              [39, 38, 37],
              [42, 41, 40],
              [45, 44, 43],
              [48, 47, 46],
              [51, 50, 49],
              [54, 53, 52],
              [],
              [57, 56, 55],
              [60, 59, 58],
              [63, 62, 61],
              [66, 65, 64],
              [69, 68, 67],
              [72, 71, 70],
              [],
              [75, 74, 73],
              [],
              [78, 77, 76],
              [81, 80, 79],
              [84, 83, 82],
              [87, 86, 85],
              [90, 89, 88],
              [93, 92, 91],
              [],
              [96, 95, 94],
              []]
    posInd = []
    for ii in np.arange(38):
        if ii == 0:
            posInd.append([1, 2, 3])
        else:
            posInd.append([])

    expmapInd = np.split(np.arange(4, 118) - 1, 38)

    return parent, offset, posInd, expmapInd


def fkl_torch(angles, parent, offset, rotInd, expmapInd):
    """
    pytorch version of fkl.

    convert joint angles to joint locations
    batch pytorch version of the fkl() method above
    :param angles: N*99
    :param parent:
    :param offset:
    :param rotInd:
    :param expmapInd:
    :return: N*joint_n*3
    """
    n = angles.data.shape[0]
    j_n = offset.shape[0]
    p3d = Variable(torch.from_numpy(offset)).float().cuda().unsqueeze(0).repeat(n, 1, 1)
    angles = angles[:, 3:].contiguous().view(-1, 3)
    R = data_utils.expmap2rotmat_torch(angles).view(n, j_n, 3, 3)
    for i in np.arange(1, j_n):
        if parent[i] > 0:
            R[:, i, :, :] = torch.matmul(R[:, i, :, :], R[:, parent[i], :, :]).clone()
            p3d[:, i, :] = torch.matmul(p3d[0, i, :], R[:, parent[i], :, :]) + p3d[:, parent[i], :]
    return p3d


def main():
    # Load all the data
    parent, offset, rotInd, expmapInd = _some_variables()

    # numpy implementation
    # with h5py.File('samples.h5', 'r') as h5f:
    #     expmap_gt = h5f['expmap/gt/walking_0'][:]
    #     expmap_pred = h5f['expmap/preds/walking_0'][:]
    expmap_pred = np.array(
        [0.0000000, 0.0000000, 0.0000000, -0.0000001, -0.0000000, -0.0000002, 0.3978439, -0.4166636, 0.1027215,
         -0.7767256, -0.0000000, -0.0000000, 0.1704115, 0.3078358, -0.1861640, 0.3330379, -0.0000000, -0.0000000,
         -0.0000000, -0.0000000, -0.0000000, 0.0679339, 0.2255526, 0.2394881, -0.0989492, -0.0000000, -0.0000000,
         0.0677801, -0.3607298, 0.0503249, 0.1819232, -0.0000000, -0.0000000, -0.0000000, -0.0000000, -0.0000000,
         0.3236777, -0.0476493, -0.0651256, -0.3150051, -0.0665669, 0.3188994, -0.5980227, -0.1190833, -0.3017127,
         1.2270271, -0.1010960, 0.2072986, -0.0000000, -0.0000000, -0.0000000, -0.2578378, -0.0125206, 2.0266378,
         -0.3701521, 0.0199115, 0.5594162, -0.4625384, -0.0000000, -0.0000000, 0.1653314, -0.3952765, -0.1731570,
         -0.0000000, -0.0000000, -0.0000000, -0.0000000, -0.0000000, -0.0000000, -0.0000000, -0.0000000, -0.0000000,
         -0.0000000, -0.0000000, -0.0000000, 2.7825687, -1.4196042, -0.0936858, -1.0348599, -2.7419815, 0.4518218,
         -0.3902033, -0.0000000, -0.0000000, 0.0597317, 0.0547002, 0.0445105, -0.0000000, -0.0000000, -0.0000000,
         -0.0000000, -0.0000000, -0.0000000, -0.0000000, -0.0000000, -0.0000000, -0.0000000, -0.0000000, -0.0000000
         ])
    expmap_gt = np.array(
        [0.2240568, -0.0276901, -0.7433901, 0.0004407, -0.0020624, 0.0002131, 0.3974636, -0.4157083, 0.1030248,
         -0.7762963, -0.0000000, -0.0000000, 0.1697988, 0.3087364, -0.1863863, 0.3327336, -0.0000000, -0.0000000,
         -0.0000000, -0.0000000, -0.0000000, 0.0689423, 0.2282812, 0.2395958, -0.0998311, -0.0000000, -0.0000000,
         0.0672752, -0.3615943, 0.0505299, 0.1816492, -0.0000000, -0.0000000, -0.0000000, -0.0000000, -0.0000000,
         0.3223563, -0.0481131, -0.0659720, -0.3145134, -0.0656419, 0.3206626, -0.5979006, -0.1181534, -0.3033383,
         1.2269648, -0.1011873, 0.2057794, -0.0000000, -0.0000000, -0.0000000, -0.2590978, -0.0141497, 2.0271597,
         -0.3699318, 0.0128547, 0.5556172, -0.4714990, -0.0000000, -0.0000000, 0.1603251, -0.4157299, -0.1667608,
         -0.0000000, -0.0000000, -0.0000000, -0.0000000, -0.0000000, -0.0000000, -0.0000000, -0.0000000, -0.0000000,
         -0.0000000, -0.0000000, -0.0000000, 2.7811005, -1.4192915, -0.0932141, -1.0294687, -2.7323222, 0.4542309,
         -0.4048152, -0.0000000, -0.0000000, 0.0568960, 0.0525994, 0.0493068, -0.0000000, -0.0000000, -0.0000000,
         -0.0000000, -0.0000000, -0.0000000, -0.0000000, -0.0000000, -0.0000000, -0.0000000, -0.0000000, -0.0000000
         ])
    xyz1 = fkl(expmap_pred, parent, offset, rotInd, expmapInd)
    xyz2 = fkl(expmap_gt, parent, offset, rotInd, expmapInd)

    exp1 = Variable(torch.from_numpy(np.vstack((expmap_pred, expmap_gt))).float()).cuda()
    xyz = fkl_torch(exp1, parent, offset, rotInd, expmapInd)
    xyz = xyz.cpu().data.numpy()
    print(xyz)


if __name__ == '__main__':
    main()
