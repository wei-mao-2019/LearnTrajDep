#!/usr/bin/env python
# -*- coding: utf-8 -*-
import numpy as np
from six.moves import xrange  # pylint: disable=redefined-builtin
import torch
from torch.autograd.variable import Variable
import os
from utils import forward_kinematics


def rotmat2euler(R):
    """
    Converts a rotation matrix to Euler angles
    Matlab port to python for evaluation purposes
    https://github.com/asheshjain399/RNNexp/blob/srnn/structural_rnn/CRFProblems/H3.6m/mhmublv/Motion/RotMat2Euler.m#L1

    Args
      R: a 3x3 rotation matrix
    Returns
      eul: a 3x1 Euler angle representation of R
    """
    if R[0, 2] == 1 or R[0, 2] == -1:
        # special case
        E3 = 0  # set arbitrarily
        dlta = np.arctan2(R[0, 1], R[0, 2]);

        if R[0, 2] == -1:
            E2 = np.pi / 2;
            E1 = E3 + dlta;
        else:
            E2 = -np.pi / 2;
            E1 = -E3 + dlta;

    else:
        E2 = -np.arcsin(R[0, 2])
        E1 = np.arctan2(R[1, 2] / np.cos(E2), R[2, 2] / np.cos(E2))
        E3 = np.arctan2(R[0, 1] / np.cos(E2), R[0, 0] / np.cos(E2))

    eul = np.array([E1, E2, E3]);
    return eul


def rotmat2quat(R):
    """
    Converts a rotation matrix to a quaternion
    Matlab port to python for evaluation purposes
    https://github.com/asheshjain399/RNNexp/blob/srnn/structural_rnn/CRFProblems/H3.6m/mhmublv/Motion/rotmat2quat.m#L4

    Args
      R: 3x3 rotation matrix
    Returns
      q: 1x4 quaternion
    """
    rotdiff = R - R.T;

    r = np.zeros(3)
    r[0] = -rotdiff[1, 2]
    r[1] = rotdiff[0, 2]
    r[2] = -rotdiff[0, 1]
    sintheta = np.linalg.norm(r) / 2;
    r0 = np.divide(r, np.linalg.norm(r) + np.finfo(np.float32).eps);

    costheta = (np.trace(R) - 1) / 2;

    theta = np.arctan2(sintheta, costheta);

    q = np.zeros(4)
    q[0] = np.cos(theta / 2)
    q[1:] = r0 * np.sin(theta / 2)
    return q


def rotmat2expmap(R):
    return quat2expmap(rotmat2quat(R));


def expmap2rotmat(r):
    """
    Converts an exponential map angle to a rotation matrix
    Matlab port to python for evaluation purposes
    I believe this is also called Rodrigues' formula
    https://github.com/asheshjain399/RNNexp/blob/srnn/structural_rnn/CRFProblems/H3.6m/mhmublv/Motion/expmap2rotmat.m

    Args
      r: 1x3 exponential map
    Returns
      R: 3x3 rotation matrix
    """
    theta = np.linalg.norm(r)
    r0 = np.divide(r, theta + np.finfo(np.float32).eps)
    r0x = np.array([0, -r0[2], r0[1], 0, 0, -r0[0], 0, 0, 0]).reshape(3, 3)
    r0x = r0x - r0x.T
    R = np.eye(3, 3) + np.sin(theta) * r0x + (1 - np.cos(theta)) * (r0x).dot(r0x);
    return R


def quat2expmap(q):
    """
    Converts a quaternion to an exponential map
    Matlab port to python for evaluation purposes
    https://github.com/asheshjain399/RNNexp/blob/srnn/structural_rnn/CRFProblems/H3.6m/mhmublv/Motion/quat2expmap.m#L1

    Args
      q: 1x4 quaternion
    Returns
      r: 1x3 exponential map
    Raises
      ValueError if the l2 norm of the quaternion is not close to 1
    """
    if (np.abs(np.linalg.norm(q) - 1) > 1e-3):
        raise (ValueError, "quat2expmap: input quaternion is not norm 1")

    sinhalftheta = np.linalg.norm(q[1:])
    coshalftheta = q[0]

    r0 = np.divide(q[1:], (np.linalg.norm(q[1:]) + np.finfo(np.float32).eps));
    theta = 2 * np.arctan2(sinhalftheta, coshalftheta)
    theta = np.mod(theta + 2 * np.pi, 2 * np.pi)

    if theta > np.pi:
        theta = 2 * np.pi - theta
        r0 = -r0

    r = r0 * theta
    return r


def unNormalizeData(normalizedData, data_mean, data_std, dimensions_to_ignore, actions, one_hot):
    """Borrowed from SRNN code. Reads a csv file and returns a float32 matrix.
    https://github.com/asheshjain399/RNNexp/blob/srnn/structural_rnn/CRFProblems/H3.6m/generateMotionData.py#L12

    Args
      normalizedData: nxd matrix with normalized data
      data_mean: vector of mean used to normalize the data
      data_std: vector of standard deviation used to normalize the data
      dimensions_to_ignore: vector with dimensions not used by the model
      actions: list of strings with the encoded actions
      one_hot: whether the data comes with one-hot encoding
    Returns
      origData: data originally used to
    """
    T = normalizedData.shape[0]
    D = data_mean.shape[0]

    origData = np.zeros((T, D), dtype=np.float32)
    dimensions_to_use = []
    for i in range(D):
        if i in dimensions_to_ignore:
            continue
        dimensions_to_use.append(i)
    dimensions_to_use = np.array(dimensions_to_use)

    if one_hot:
        origData[:, dimensions_to_use] = normalizedData[:, :-len(actions)]
    else:
        origData[:, dimensions_to_use] = normalizedData

    # potentially ineficient, but only done once per experiment
    stdMat = data_std.reshape((1, D))
    stdMat = np.repeat(stdMat, T, axis=0)
    meanMat = data_mean.reshape((1, D))
    meanMat = np.repeat(meanMat, T, axis=0)
    origData = np.multiply(origData, stdMat) + meanMat
    return origData


def revert_output_format(poses, data_mean, data_std, dim_to_ignore, actions, one_hot):
    """
    Converts the output of the neural network to a format that is more easy to
    manipulate for, e.g. conversion to other format or visualization

    Args
      poses: The output from the TF model. A list with (seq_length) entries,
      each with a (batch_size, dim) output
    Returns
      poses_out: A tensor of size (batch_size, seq_length, dim) output. Each
      batch is an n-by-d sequence of poses.
    """
    seq_len = len(poses)
    if seq_len == 0:
        return []

    batch_size, dim = poses[0].shape

    poses_out = np.concatenate(poses)
    poses_out = np.reshape(poses_out, (seq_len, batch_size, dim))
    poses_out = np.transpose(poses_out, [1, 0, 2])

    poses_out_list = []
    for i in xrange(poses_out.shape[0]):
        poses_out_list.append(
            unNormalizeData(poses_out[i, :, :], data_mean, data_std, dim_to_ignore, actions, one_hot))

    return poses_out_list


def readCSVasFloat(filename):
    """
    Borrowed from SRNN code. Reads a csv and returns a float matrix.
    https://github.com/asheshjain399/NeuralModels/blob/master/neuralmodels/utils.py#L34

    Args
      filename: string. Path to the csv file
    Returns
      returnArray: the read data in a float32 matrix
    """
    returnArray = []
    lines = open(filename).readlines()
    for line in lines:
        line = line.strip().split(',')
        if len(line) > 0:
            returnArray.append(np.array([np.float32(x) for x in line]))

    returnArray = np.array(returnArray)
    return returnArray


def normalize_data(data, data_mean, data_std, dim_to_use, actions, one_hot):
    """
    Normalize input data by removing unused dimensions, subtracting the mean and
    dividing by the standard deviation

    Args
      data: nx99 matrix with data to normalize
      data_mean: vector of mean used to normalize the data
      data_std: vector of standard deviation used to normalize the data
      dim_to_use: vector with dimensions used by the model
      actions: list of strings with the encoded actions
      one_hot: whether the data comes with one-hot encoding
    Returns
      data_out: the passed data matrix, but normalized
    """
    data_out = {}
    nactions = len(actions)

    if not one_hot:
        # No one-hot encoding... no need to do anything special
        for key in data.keys():
            data_out[key] = np.divide((data[key] - data_mean), data_std)
            data_out[key] = data_out[key][:, dim_to_use]

    else:
        # TODO hard-coding 99 dimensions for un-normalized human poses
        for key in data.keys():
            data_out[key] = np.divide((data[key][:, 0:99] - data_mean), data_std)
            data_out[key] = data_out[key][:, dim_to_use]
            data_out[key] = np.hstack((data_out[key], data[key][:, -nactions:]))

    return data_out


def normalization_stats(completeData):
    """"
    Also borrowed for SRNN code. Computes mean, stdev and dimensions to ignore.
    https://github.com/asheshjain399/RNNexp/blob/srnn/structural_rnn/CRFProblems/H3.6m/processdata.py#L33

    Args
      completeData: nx99 matrix with data to normalize
    Returns
      data_mean: vector of mean used to normalize the data
      data_std: vector of standard deviation used to normalize the data
      dimensions_to_ignore: vector with dimensions not used by the model
      dimensions_to_use: vector with dimensions used by the model
    """
    data_mean = np.mean(completeData, axis=0)
    data_std = np.std(completeData, axis=0)

    dimensions_to_ignore = []
    dimensions_to_use = []

    dimensions_to_ignore.extend(list(np.where(data_std < 1e-4)[0]))
    dimensions_to_use.extend(list(np.where(data_std >= 1e-4)[0]))

    data_std[dimensions_to_ignore] = 1.0

    return data_mean, data_std, dimensions_to_ignore, dimensions_to_use


def define_actions(action):
    """
    Define the list of actions we are using.

    Args
      action: String with the passed action. Could be "all"
    Returns
      actions: List of strings of actions
    Raises
      ValueError if the action is not included in H3.6M
    """

    actions = ["walking", "eating", "smoking", "discussion", "directions",
               "greeting", "phoning", "posing", "purchases", "sitting",
               "sittingdown", "takingphoto", "waiting", "walkingdog",
               "walkingtogether"]
    if action in actions:
        return [action]

    if action == "all":
        return actions

    if action == "all_srnn":
        return ["walking", "eating", "smoking", "discussion"]

    raise (ValueError, "Unrecognized action: %d" % action)


"""all methods above are borrowed from https://github.com/una-dinosauria/human-motion-prediction"""


def define_actions_cmu(action):
    """
    Define the list of actions we are using.

    Args
      action: String with the passed action. Could be "all"
    Returns
      actions: List of strings of actions
    Raises
      ValueError if the action is not included in H3.6M
    """

    actions = ["basketball", "basketball_signal", "directing_traffic", "jumping", "running", "soccer", "walking",
               "washwindow"]
    if action in actions:
        return [action]

    if action == "all":
        return actions

    raise (ValueError, "Unrecognized action: %d" % action)


def load_data_cmu(path_to_dataset, actions, input_n, output_n, data_std=0, data_mean=0, is_test=False):
    seq_len = input_n + output_n
    nactions = len(actions)
    sampled_seq = []
    complete_seq = []
    for action_idx in np.arange(nactions):
        action = actions[action_idx]
        path = '{}/{}'.format(path_to_dataset, action)
        count = 0
        for _ in os.listdir(path):
            count = count + 1
        for examp_index in np.arange(count):
            filename = '{}/{}/{}_{}.txt'.format(path_to_dataset, action, action, examp_index + 1)
            action_sequence = readCSVasFloat(filename)
            n, d = action_sequence.shape
            even_list = range(0, n, 2)
            the_sequence = np.array(action_sequence[even_list, :])
            num_frames = len(the_sequence)
            if not is_test:
                fs = np.arange(0, num_frames - seq_len + 1)
                fs_sel = fs
                for i in np.arange(seq_len - 1):
                    fs_sel = np.vstack((fs_sel, fs + i + 1))
                fs_sel = fs_sel.transpose()
                seq_sel = the_sequence[fs_sel, :]
                if len(sampled_seq) == 0:
                    sampled_seq = seq_sel
                    complete_seq = the_sequence
                else:
                    sampled_seq = np.concatenate((sampled_seq, seq_sel), axis=0)
                    complete_seq = np.append(complete_seq, the_sequence, axis=0)
            else:
                source_seq_len = 50
                target_seq_len = 25
                total_frames = source_seq_len + target_seq_len
                batch_size = 8
                SEED = 1234567890
                rng = np.random.RandomState(SEED)
                for _ in range(batch_size):
                    idx = rng.randint(0, num_frames - total_frames)
                    seq_sel = the_sequence[
                              idx + (source_seq_len - input_n):(idx + source_seq_len + output_n), :]
                    seq_sel = np.expand_dims(seq_sel, axis=0)
                    if len(sampled_seq) == 0:
                        sampled_seq = seq_sel
                        complete_seq = the_sequence
                    else:
                        sampled_seq = np.concatenate((sampled_seq, seq_sel), axis=0)
                        complete_seq = np.append(complete_seq, the_sequence, axis=0)

    if not is_test:
        data_std = np.std(complete_seq, axis=0)
        data_mean = np.mean(complete_seq, axis=0)

    dimensions_to_ignore = []
    dimensions_to_use = []
    dimensions_to_ignore.extend(list(np.where(data_std < 1e-4)[0]))
    dimensions_to_use.extend(list(np.where(data_std >= 1e-4)[0]))
    data_std[dimensions_to_ignore] = 1.0
    data_mean[dimensions_to_ignore] = 0.0

    return sampled_seq, dimensions_to_ignore, dimensions_to_use, data_mean, data_std


def load_data_cmu_3d(path_to_dataset, actions, input_n, output_n, data_std=0, data_mean=0, is_test=False):
    seq_len = input_n + output_n
    nactions = len(actions)
    sampled_seq = []
    complete_seq = []
    for action_idx in np.arange(nactions):
        action = actions[action_idx]
        path = '{}/{}'.format(path_to_dataset, action)
        count = 0
        for _ in os.listdir(path):
            count = count + 1
        for examp_index in np.arange(count):
            filename = '{}/{}/{}_{}.txt'.format(path_to_dataset, action, action, examp_index + 1)
            action_sequence = readCSVasFloat(filename)
            n, d = action_sequence.shape
            exptmps = Variable(torch.from_numpy(action_sequence)).float().cuda()
            xyz = expmap2xyz_torch_cmu(exptmps)
            xyz = xyz.view(-1, 38 * 3)
            xyz = xyz.cpu().data.numpy()
            action_sequence = xyz

            even_list = range(0, n, 2)
            the_sequence = np.array(action_sequence[even_list, :])
            num_frames = len(the_sequence)
            if not is_test:
                fs = np.arange(0, num_frames - seq_len + 1)
                fs_sel = fs
                for i in np.arange(seq_len - 1):
                    fs_sel = np.vstack((fs_sel, fs + i + 1))
                fs_sel = fs_sel.transpose()
                seq_sel = the_sequence[fs_sel, :]
                if len(sampled_seq) == 0:
                    sampled_seq = seq_sel
                    complete_seq = the_sequence
                else:
                    sampled_seq = np.concatenate((sampled_seq, seq_sel), axis=0)
                    complete_seq = np.append(complete_seq, the_sequence, axis=0)
            else:
                source_seq_len = 50
                target_seq_len = 25
                total_frames = source_seq_len + target_seq_len
                batch_size = 8
                SEED = 1234567890
                rng = np.random.RandomState(SEED)
                for _ in range(batch_size):
                    idx = rng.randint(0, num_frames - total_frames)
                    seq_sel = the_sequence[
                              idx + (source_seq_len - input_n):(idx + source_seq_len + output_n), :]
                    seq_sel = np.expand_dims(seq_sel, axis=0)
                    if len(sampled_seq) == 0:
                        sampled_seq = seq_sel
                        complete_seq = the_sequence
                    else:
                        sampled_seq = np.concatenate((sampled_seq, seq_sel), axis=0)
                        complete_seq = np.append(complete_seq, the_sequence, axis=0)

    if not is_test:
        data_std = np.std(complete_seq, axis=0)
        data_mean = np.mean(complete_seq, axis=0)

    joint_to_ignore = np.array([0, 1, 2, 7, 8, 13, 16, 20, 29, 24, 27, 33, 36])
    dimensions_to_ignore = np.concatenate((joint_to_ignore * 3, joint_to_ignore * 3 + 1, joint_to_ignore * 3 + 2))
    dimensions_to_use = np.setdiff1d(np.arange(complete_seq.shape[1]), dimensions_to_ignore)

    data_std[dimensions_to_ignore] = 1.0
    data_mean[dimensions_to_ignore] = 0.0

    return sampled_seq, dimensions_to_ignore, dimensions_to_use, data_mean, data_std


def rotmat2euler_torch(R):
    """
    Converts a rotation matrix to euler angles
    batch pytorch version ported from the corresponding numpy method above

    :param R:N*3*3
    :return: N*3
    """
    n = R.data.shape[0]
    eul = Variable(torch.zeros(n, 3).float()).cuda()
    idx_spec1 = (R[:, 0, 2] == 1).nonzero().cpu().data.numpy().reshape(-1).tolist()
    idx_spec2 = (R[:, 0, 2] == -1).nonzero().cpu().data.numpy().reshape(-1).tolist()
    if len(idx_spec1) > 0:
        R_spec1 = R[idx_spec1, :, :]
        eul_spec1 = Variable(torch.zeros(len(idx_spec1), 3).float()).cuda()
        eul_spec1[:, 2] = 0
        eul_spec1[:, 1] = -np.pi / 2
        delta = torch.atan2(R_spec1[:, 0, 1], R_spec1[:, 0, 2])
        eul_spec1[:, 0] = delta
        eul[idx_spec1, :] = eul_spec1

    if len(idx_spec2) > 0:
        R_spec2 = R[idx_spec2, :, :]
        eul_spec2 = Variable(torch.zeros(len(idx_spec2), 3).float()).cuda()
        eul_spec2[:, 2] = 0
        eul_spec2[:, 1] = np.pi / 2
        delta = torch.atan2(R_spec2[:, 0, 1], R_spec2[:, 0, 2])
        eul_spec2[:, 0] = delta
        eul[idx_spec2] = eul_spec2

    idx_remain = np.arange(0, n)
    idx_remain = np.setdiff1d(np.setdiff1d(idx_remain, idx_spec1), idx_spec2).tolist()
    if len(idx_remain) > 0:
        R_remain = R[idx_remain, :, :]
        eul_remain = Variable(torch.zeros(len(idx_remain), 3).float()).cuda()
        eul_remain[:, 1] = -torch.asin(R_remain[:, 0, 2])
        eul_remain[:, 0] = torch.atan2(R_remain[:, 1, 2] / torch.cos(eul_remain[:, 1]),
                                       R_remain[:, 2, 2] / torch.cos(eul_remain[:, 1]))
        eul_remain[:, 2] = torch.atan2(R_remain[:, 0, 1] / torch.cos(eul_remain[:, 1]),
                                       R_remain[:, 0, 0] / torch.cos(eul_remain[:, 1]))
        eul[idx_remain, :] = eul_remain

    return eul


def rotmat2quat_torch(R):
    """
    Converts a rotation matrix to quaternion
    batch pytorch version ported from the corresponding numpy method above
    :param R: N * 3 * 3
    :return: N * 4
    """
    rotdiff = R - R.transpose(1, 2)
    r = torch.zeros_like(rotdiff[:, 0])
    r[:, 0] = -rotdiff[:, 1, 2]
    r[:, 1] = rotdiff[:, 0, 2]
    r[:, 2] = -rotdiff[:, 0, 1]
    r_norm = torch.norm(r, dim=1)
    sintheta = r_norm / 2
    r0 = torch.div(r, r_norm.unsqueeze(1).repeat(1, 3) + 0.00000001)
    t1 = R[:, 0, 0]
    t2 = R[:, 1, 1]
    t3 = R[:, 2, 2]
    costheta = (t1 + t2 + t3 - 1) / 2
    theta = torch.atan2(sintheta, costheta)
    q = Variable(torch.zeros(R.shape[0], 4)).float().cuda()
    q[:, 0] = torch.cos(theta / 2)
    q[:, 1:] = torch.mul(r0, torch.sin(theta / 2).unsqueeze(1).repeat(1, 3))

    return q


def expmap2quat_torch(exp):
    """
    Converts expmap to quaternion
    batch pytorch version ported from the corresponding numpy method above
    :param R: N*3
    :return: N*4
    """
    theta = torch.norm(exp, p=2, dim=1).unsqueeze(1)
    v = torch.div(exp, theta.repeat(1, 3) + 0.0000001)
    sinhalf = torch.sin(theta / 2)
    coshalf = torch.cos(theta / 2)
    q1 = torch.mul(v, sinhalf.repeat(1, 3))
    q = torch.cat((coshalf, q1), dim=1)
    return q


def expmap2rotmat_torch(r):
    """
    Converts expmap matrix to rotation
    batch pytorch version ported from the corresponding method above
    :param r: N*3
    :return: N*3*3
    """
    theta = torch.norm(r, 2, 1)
    r0 = torch.div(r, theta.unsqueeze(1).repeat(1, 3) + 0.0000001)
    r1 = torch.zeros_like(r0).repeat(1, 3)
    r1[:, 1] = -r0[:, 2]
    r1[:, 2] = r0[:, 1]
    r1[:, 5] = -r0[:, 0]
    r1 = r1.view(-1, 3, 3)
    r1 = r1 - r1.transpose(1, 2)
    n = r1.data.shape[0]
    R = Variable(torch.eye(3, 3).repeat(n, 1, 1)).float().cuda() + torch.mul(
        torch.sin(theta).unsqueeze(1).repeat(1, 9).view(-1, 3, 3), r1) + torch.mul(
        (1 - torch.cos(theta).unsqueeze(1).repeat(1, 9).view(-1, 3, 3)), torch.matmul(r1, r1))
    return R


def expmap2xyz_torch(expmap):
    """
    convert expmaps to joint locations
    :param expmap: N*99
    :return: N*32*3
    """
    parent, offset, rotInd, expmapInd = forward_kinematics._some_variables()
    xyz = forward_kinematics.fkl_torch(expmap, parent, offset, rotInd, expmapInd)
    return xyz


def expmap2xyz_torch_cmu(expmap):
    parent, offset, rotInd, expmapInd = forward_kinematics._some_variables_cmu()
    xyz = forward_kinematics.fkl_torch(expmap, parent, offset, rotInd, expmapInd)
    return xyz


def load_data(path_to_dataset, subjects, actions, sample_rate, seq_len, input_n=10, data_mean=None, data_std=None):
    """
    adapted from
    https://github.com/una-dinosauria/human-motion-prediction/src/data_utils.py#L216

    :param path_to_dataset: path of dataset
    :param subjects:
    :param actions:
    :param sample_rate:
    :param seq_len: past frame length + future frame length
    :param is_norm: normalize the expmap or not
    :param data_std: standard deviation of the expmap
    :param data_mean: mean of the expmap
    :param input_n: past frame length
    :return:
    """

    sampled_seq = []
    complete_seq = []
    # actions_all = define_actions("all")
    # one_hot_all = np.eye(len(actions_all))
    for subj in subjects:
        for action_idx in np.arange(len(actions)):
            action = actions[action_idx]
            if not (subj == 5):
                for subact in [1, 2]:  # subactions

                    print("Reading subject {0}, action {1}, subaction {2}".format(subj, action, subact))

                    filename = '{0}/S{1}/{2}_{3}.txt'.format(path_to_dataset, subj, action, subact)
                    action_sequence = readCSVasFloat(filename)
                    n, d = action_sequence.shape
                    even_list = range(0, n, sample_rate)
                    the_sequence = np.array(action_sequence[even_list, :])
                    num_frames = len(the_sequence)
                    fs = np.arange(0, num_frames - seq_len + 1)
                    fs_sel = fs
                    for i in np.arange(seq_len - 1):
                        fs_sel = np.vstack((fs_sel, fs + i + 1))
                    fs_sel = fs_sel.transpose()
                    seq_sel = the_sequence[fs_sel, :]
                    if len(sampled_seq) == 0:
                        sampled_seq = seq_sel
                        complete_seq = the_sequence
                    else:
                        sampled_seq = np.concatenate((sampled_seq, seq_sel), axis=0)
                        complete_seq = np.append(complete_seq, the_sequence, axis=0)
            else:
                print("Reading subject {0}, action {1}, subaction {2}".format(subj, action, 1))
                filename = '{0}/S{1}/{2}_{3}.txt'.format(path_to_dataset, subj, action, 1)
                action_sequence = readCSVasFloat(filename)
                n, d = action_sequence.shape
                even_list = range(0, n, sample_rate)
                the_sequence1 = np.array(action_sequence[even_list, :])
                num_frames1 = len(the_sequence1)

                print("Reading subject {0}, action {1}, subaction {2}".format(subj, action, 2))
                filename = '{0}/S{1}/{2}_{3}.txt'.format(path_to_dataset, subj, action, 2)
                action_sequence = readCSVasFloat(filename)
                n, d = action_sequence.shape
                even_list = range(0, n, sample_rate)
                the_sequence2 = np.array(action_sequence[even_list, :])
                num_frames2 = len(the_sequence2)

                fs_sel1, fs_sel2 = find_indices_srnn(num_frames1, num_frames2, seq_len, input_n=input_n)
                seq_sel1 = the_sequence1[fs_sel1, :]
                seq_sel2 = the_sequence2[fs_sel2, :]
                if len(sampled_seq) == 0:
                    sampled_seq = seq_sel1
                    sampled_seq = np.concatenate((sampled_seq, seq_sel2), axis=0)
                    complete_seq = the_sequence1
                    complete_seq = np.append(complete_seq, the_sequence2, axis=0)

    # if is not testing or validation then get the data statistics
    if not (subj == 5 and subj == 11):
        data_std = np.std(complete_seq, axis=0)
        data_mean = np.mean(complete_seq, axis=0)

    dimensions_to_ignore = []
    dimensions_to_use = []
    dimensions_to_ignore.extend(list(np.where(data_std < 1e-4)[0]))
    dimensions_to_use.extend(list(np.where(data_std >= 1e-4)[0]))
    data_std[dimensions_to_ignore] = 1.0
    data_mean[dimensions_to_ignore] = 0.0

    return sampled_seq, dimensions_to_ignore, dimensions_to_use, data_mean, data_std


def load_data_3d(path_to_dataset, subjects, actions, sample_rate, seq_len):
    """

    adapted from
    https://github.com/una-dinosauria/human-motion-prediction/src/data_utils.py#L216
    :param path_to_dataset:
    :param subjects:
    :param actions:
    :param sample_rate:
    :param seq_len:
    :return:
    """

    sampled_seq = []
    complete_seq = []
    for subj in subjects:
        for action_idx in np.arange(len(actions)):
            action = actions[action_idx]
            if not (subj == 5):
                for subact in [1, 2]:  # subactions

                    print("Reading subject {0}, action {1}, subaction {2}".format(subj, action, subact))

                    filename = '{0}/S{1}/{2}_{3}.txt'.format(path_to_dataset, subj, action, subact)
                    action_sequence = readCSVasFloat(filename)
                    n, d = action_sequence.shape
                    even_list = range(0, n, sample_rate)
                    num_frames = len(even_list)
                    the_sequence = np.array(action_sequence[even_list, :])
                    the_seq = Variable(torch.from_numpy(the_sequence)).float().cuda()
                    # remove global rotation and translation
                    the_seq[:, 0:6] = 0
                    p3d = expmap2xyz_torch(the_seq)
                    the_sequence = p3d.view(num_frames, -1).cpu().data.numpy()

                    fs = np.arange(0, num_frames - seq_len + 1)
                    fs_sel = fs
                    for i in np.arange(seq_len - 1):
                        fs_sel = np.vstack((fs_sel, fs + i + 1))
                    fs_sel = fs_sel.transpose()
                    seq_sel = the_sequence[fs_sel, :]
                    if len(sampled_seq) == 0:
                        sampled_seq = seq_sel
                        complete_seq = the_sequence
                    else:
                        sampled_seq = np.concatenate((sampled_seq, seq_sel), axis=0)
                        complete_seq = np.append(complete_seq, the_sequence, axis=0)
            else:
                print("Reading subject {0}, action {1}, subaction {2}".format(subj, action, 1))
                filename = '{0}/S{1}/{2}_{3}.txt'.format(path_to_dataset, subj, action, 1)
                action_sequence = readCSVasFloat(filename)
                n, d = action_sequence.shape
                even_list = range(0, n, sample_rate)

                num_frames1 = len(even_list)
                the_sequence1 = np.array(action_sequence[even_list, :])
                the_seq1 = Variable(torch.from_numpy(the_sequence1)).float().cuda()
                the_seq1[:, 0:6] = 0
                p3d1 = expmap2xyz_torch(the_seq1)
                the_sequence1 = p3d1.view(num_frames1, -1).cpu().data.numpy()

                print("Reading subject {0}, action {1}, subaction {2}".format(subj, action, 2))
                filename = '{0}/S{1}/{2}_{3}.txt'.format(path_to_dataset, subj, action, 2)
                action_sequence = readCSVasFloat(filename)
                n, d = action_sequence.shape
                even_list = range(0, n, sample_rate)

                num_frames2 = len(even_list)
                the_sequence2 = np.array(action_sequence[even_list, :])
                the_seq2 = Variable(torch.from_numpy(the_sequence2)).float().cuda()
                the_seq2[:, 0:6] = 0
                p3d2 = expmap2xyz_torch(the_seq2)
                the_sequence2 = p3d2.view(num_frames2, -1).cpu().data.numpy()

                # print("action:{}".format(action))
                # print("subact1:{}".format(num_frames1))
                # print("subact2:{}".format(num_frames2))
                fs_sel1, fs_sel2 = find_indices_srnn(num_frames1, num_frames2, seq_len)
                seq_sel1 = the_sequence1[fs_sel1, :]
                seq_sel2 = the_sequence2[fs_sel2, :]
                if len(sampled_seq) == 0:
                    sampled_seq = seq_sel1
                    sampled_seq = np.concatenate((sampled_seq, seq_sel2), axis=0)
                    complete_seq = the_sequence1
                    complete_seq = np.append(complete_seq, the_sequence2, axis=0)
                else:
                    sampled_seq = np.concatenate((sampled_seq, seq_sel1), axis=0)
                    sampled_seq = np.concatenate((sampled_seq, seq_sel2), axis=0)
                    complete_seq = np.append(complete_seq, the_sequence1, axis=0)
                    complete_seq = np.append(complete_seq, the_sequence2, axis=0)

    # ignore constant joints and joints at same position with other joints
    joint_to_ignore = np.array([0, 1, 6, 11, 16, 20, 23, 24, 28, 31])
    dimensions_to_ignore = np.concatenate((joint_to_ignore * 3, joint_to_ignore * 3 + 1, joint_to_ignore * 3 + 2))
    dimensions_to_use = np.setdiff1d(np.arange(complete_seq.shape[1]), dimensions_to_ignore)

    return sampled_seq, dimensions_to_ignore, dimensions_to_use


def get_dct_matrix(N):
    dct_m = np.eye(N)
    for k in np.arange(N):
        for i in np.arange(N):
            w = np.sqrt(2 / N)
            if k == 0:
                w = np.sqrt(1 / N)
            dct_m[k, i] = w * np.cos(np.pi * (i + 1 / 2) * k / N)
    idct_m = np.linalg.inv(dct_m)
    return dct_m, idct_m


def find_indices_srnn(frame_num1, frame_num2, seq_len, input_n=10):
    """
    Adapted from https://github.com/una-dinosauria/human-motion-prediction/blob/master/src/seq2seq_model.py#L478

    which originaly from
    In order to find the same action indices as in SRNN.
    https://github.com/asheshjain399/RNNexp/blob/master/structural_rnn/CRFProblems/H3.6m/processdata.py#L325
    """

    # Used a fixed dummy seed, following
    # https://github.com/asheshjain399/RNNexp/blob/srnn/structural_rnn/forecastTrajectories.py#L29
    SEED = 1234567890
    rng = np.random.RandomState(SEED)

    T1 = frame_num1 - 150
    T2 = frame_num2 - 150  # seq_len
    idxo1 = None
    idxo2 = None
    for _ in np.arange(0, 4):
        idx_ran1 = rng.randint(16, T1)
        idx_ran2 = rng.randint(16, T2)
        idxs1 = np.arange(idx_ran1 + 50 - input_n, idx_ran1 + 50 - input_n + seq_len)
        idxs2 = np.arange(idx_ran2 + 50 - input_n, idx_ran2 + 50 - input_n + seq_len)
        if idxo1 is None:
            idxo1 = idxs1
            idxo2 = idxs2
        else:
            idxo1 = np.vstack((idxo1, idxs1))
            idxo2 = np.vstack((idxo2, idxs2))
    return idxo1, idxo2


if __name__ == "__main__":
    r = np.random.rand(2, 3) * 10
    # r = np.array([[0.4, 1.5, -0.0], [0, 0, 1.4]])
    r1 = r[0]
    R1 = expmap2rotmat(r1)
    q1 = rotmat2quat(R1)
    # R1 = np.array([[0, 0, 1], [0, 1, 0], [1, 0, 0]])
    e1 = rotmat2euler(R1)

    r2 = r[1]
    R2 = expmap2rotmat(r2)
    q2 = rotmat2quat(R2)
    # R2 = np.array([[0, 0, -1], [0, 1, 0], [1, 0, 0]])
    e2 = rotmat2euler(R2)

    r = Variable(torch.from_numpy(r)).cuda().float()
    # q = expmap2quat_torch(r)
    R = expmap2rotmat_torch(r)
    q = rotmat2quat_torch(R)
    # R = Variable(torch.from_numpy(
    #     np.array([[[0, 0, 1], [0, 1, 0], [1, 0, 0]], [[0, 0, -1], [0, 1, 0], [1, 0, 0]]]))).cuda().float()
    eul = rotmat2euler_torch(R)
    eul = eul.cpu().data.numpy()
    R = R.cpu().data.numpy()
    q = q.cpu().data.numpy()

    if np.max(np.abs(eul[0] - e1)) < 0.000001:
        print('e1 clear')
    else:
        print('e1 error {}'.format(np.max(np.abs(eul[0] - e1))))
    if np.max(np.abs(eul[1] - e2)) < 0.000001:
        print('e2 clear')
    else:
        print('e2 error {}'.format(np.max(np.abs(eul[1] - e2))))

    if np.max(np.abs(R[0] - R1)) < 0.000001:
        print('R1 clear')
    else:
        print('R1 error {}'.format(np.max(np.abs(R[0] - R1))))

    if np.max(np.abs(R[1] - R2)) < 0.000001:
        print('R2 clear')
    else:
        print('R2 error {}'.format(np.max(np.abs(R[1] - R2))))

    if np.max(np.abs(q[0] - q1)) < 0.000001:
        print('q1 clear')
    else:
        print('q1 error {}'.format(np.max(np.abs(q[0] - q1))))
