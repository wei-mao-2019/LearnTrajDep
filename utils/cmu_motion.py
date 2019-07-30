from torch.utils.data import Dataset
import numpy as np
from utils import data_utils


class CMU_Motion(Dataset):

    def __init__(self, path_to_data, actions, input_n=20, output_n=10, dct_n=20, split=0, data_mean=0, data_std=0,
                 dim_used=0):
        """

        :param path_to_data:
        :param actions:
        :param input_n:
        :param output_n:
        :param dct_n:
        :param split:
        :param data_mean:
        :param data_std:
        :param dim_used:
        """
        self.path_to_data = path_to_data
        self.split = split
        # actions = 'walking'
        actions = data_utils.define_actions_cmu(actions)

        if split == 0:
            path_to_data = path_to_data + '/train/'
            is_test = False
        else:
            path_to_data = path_to_data + '/test/'
            is_test = True

        all_seqs, dim_ignore, dim_use, data_mean, data_std = data_utils.load_data_cmu(path_to_data, actions,
                                                                                      input_n, output_n,
                                                                                      data_std=data_std,
                                                                                      data_mean=data_mean,
                                                                                      is_test=is_test)
        if not is_test:
            dim_used = dim_use[6:]

        self.all_seqs = all_seqs
        self.dim_used = dim_used
        all_seqs = all_seqs[:, :, dim_used]
        all_seqs = all_seqs.transpose(0, 2, 1)
        all_seqs = all_seqs.reshape(-1, input_n + output_n)
        all_seqs = all_seqs.transpose()

        dct_m_in, _ = data_utils.get_dct_matrix(input_n + output_n)
        dct_m_out, _ = data_utils.get_dct_matrix(input_n + output_n)
        pad_idx = np.repeat([input_n - 1], output_n)
        i_idx = np.append(np.arange(0, input_n), pad_idx)
        input_dct_seq = np.matmul(dct_m_in[:dct_n, :], all_seqs[i_idx, :])
        input_dct_seq = input_dct_seq.transpose().reshape(-1, len(dim_used), dct_n)

        output_dct_seq = np.matmul(dct_m_out[:dct_n, :], all_seqs)
        output_dct_seq = output_dct_seq.transpose().reshape(-1, len(dim_used), dct_n)

        self.input_dct_seq = input_dct_seq
        self.output_dct_seq = output_dct_seq
        self.data_mean = data_mean
        self.data_std = data_std

    def __len__(self):
        return np.shape(self.input_dct_seq)[0]

    def __getitem__(self, item):
        return self.input_dct_seq[item], self.output_dct_seq[item], self.all_seqs[item]
