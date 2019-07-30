import pickle as pkl
from os import walk
from torch.utils.data import Dataset
import numpy as np
from utils import data_utils

from matplotlib import pyplot as plt


class Pose3dPW3D(Dataset):

    def __init__(self, path_to_data, input_n=20, output_n=10, dct_n=15, split=0):
        """

        :param path_to_data:
        :param input_n:
        :param output_n:
        :param dct_n:
        :param split:
        """
        self.path_to_data = path_to_data
        self.split = split
        self.dct_n = dct_n

        # since baselines (http://arxiv.org/abs/1805.00655.pdf and https://arxiv.org/pdf/1705.02445.pdf)
        # use observed 50 frames but our method use 10 past frames in order to make sure all methods are evaluated
        # on same sequences, we first crop the sequence with 50 past frames and then use the last 10 frame as input
        if split == 1:
            their_input_n = 50
        else:
            their_input_n = input_n
        seq_len = their_input_n + output_n

        if split == 0:
            self.data_path = path_to_data + '/train/'
        elif split == 1:
            self.data_path = path_to_data + '/test/'
        elif split == 2:
            self.data_path = path_to_data + '/validation/'
        all_seqs = []
        files = []
        for (dirpath, dirnames, filenames) in walk(self.data_path):
            files.extend(filenames)
        for f in files:
            with open(self.data_path + f, 'rb') as f:
                data = pkl.load(f, encoding='latin1')
                joint_pos = data['jointPositions']
                for i in range(len(joint_pos)):
                    seqs = joint_pos[i]
                    seqs = seqs - seqs[:, 0:3].repeat(24, axis=0).reshape(-1, 72)
                    n_frames = seqs.shape[0]
                    fs = np.arange(0, n_frames - seq_len + 1)
                    fs_sel = fs
                    for j in np.arange(seq_len - 1):
                        fs_sel = np.vstack((fs_sel, fs + j + 1))
                    fs_sel = fs_sel.transpose()
                    seq_sel = seqs[fs_sel, :]
                    if len(all_seqs) == 0:
                        all_seqs = seq_sel
                    else:
                        all_seqs = np.concatenate((all_seqs, seq_sel), axis=0)

        self.all_seqs = all_seqs[:, (their_input_n - input_n):, :]

        self.dim_used = np.array(range(3, all_seqs.shape[2]))
        all_seqs = all_seqs[:, (their_input_n - input_n):, 3:]
        n, seq_len, dim_len = all_seqs.shape
        all_seqs = all_seqs.transpose(0, 2, 1)
        all_seqs = all_seqs.reshape(-1, input_n + output_n)
        all_seqs = all_seqs.transpose()

        dct_m_in, _ = data_utils.get_dct_matrix(input_n + output_n)
        dct_m_out, _ = data_utils.get_dct_matrix(input_n + output_n)
        pad_idx = np.repeat([input_n - 1], output_n)
        i_idx = np.append(np.arange(0, input_n), pad_idx)
        input_dct_seq = np.matmul(dct_m_in[0:dct_n, :], all_seqs[i_idx, :])
        input_dct_seq = input_dct_seq.transpose().reshape(-1, dim_len, dct_n)
        # input_dct_seq = input_dct_seq.reshape(-1, dim_len * dct_used)

        output_dct_seq = np.matmul(dct_m_out[0:dct_n, :], all_seqs)
        output_dct_seq = output_dct_seq.transpose().reshape(-1, dim_len, dct_n)
        # output_dct_seq = output_dct_seq.reshape(-1, dim_len * dct_used)

        self.input_dct_seq = input_dct_seq
        self.output_dct_seq = output_dct_seq

    def __len__(self):
        return np.shape(self.input_dct_seq)[0]

    def __getitem__(self, item):
        return self.input_dct_seq[item], self.output_dct_seq[item], self.all_seqs[item]
