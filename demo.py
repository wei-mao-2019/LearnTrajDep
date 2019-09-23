#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import print_function, absolute_import, division

import os
import time
import torch
import torch.nn as nn
import torch.optim
from torch.utils.data import DataLoader
from torch.autograd import Variable
from torch.nn import functional
import numpy as np
from progress.bar import Bar
import pandas as pd
from matplotlib import pyplot as plt

from utils import loss_funcs, utils as utils
from utils.opt import Options
from utils.h36motion import H36motion
import utils.model as nnmodel
import utils.data_utils as data_utils
import utils.viz as viz


def main(opt):
    is_cuda = torch.cuda.is_available()

    # create model
    print(">>> creating model")
    input_n = opt.input_n
    output_n = opt.output_n
    sample_rate = opt.sample_rate

    model = nnmodel.GCN(input_feature=(input_n + output_n), hidden_feature=opt.linear_size, p_dropout=opt.dropout,
                        num_stage=opt.num_stage, node_n=48)
    if is_cuda:
        model.cuda()
    model_path_len = './checkpoint/pretrained/h36m_in10_out25.pth.tar'
    print(">>> loading ckpt len from '{}'".format(model_path_len))
    if is_cuda:
        ckpt = torch.load(model_path_len)
    else:
        ckpt = torch.load(model_path_len, map_location='cpu')
    err_best = ckpt['err']
    start_epoch = ckpt['epoch']
    model.load_state_dict(ckpt['state_dict'])
    print(">>> ckpt len loaded (epoch: {} | err: {})".format(start_epoch, err_best))

    # data loading
    print(">>> loading data")
    acts = data_utils.define_actions('all')
    test_data = dict()
    for act in acts:
        test_dataset = H36motion(path_to_data=opt.data_dir, actions=act, input_n=input_n, output_n=output_n, split=1,
                                 sample_rate=sample_rate)
        test_data[act] = DataLoader(
            dataset=test_dataset,
            batch_size=opt.test_batch,
            shuffle=False,
            num_workers=opt.job,
            pin_memory=True)
    dim_used = test_dataset.dim_used
    print(">>> data loaded !")

    model.eval()
    fig = plt.figure()
    ax = plt.gca(projection='3d')
    for act in acts:
        for i, (inputs, targets, all_seq) in enumerate(test_data[act]):
            inputs = Variable(inputs).float()
            all_seq = Variable(all_seq).float()
            if is_cuda:
                inputs = inputs.cuda()
                all_seq = all_seq.cuda()

            outputs = model(inputs)

            n, seq_len, dim_full_len = all_seq.data.shape
            dim_used_len = len(dim_used)

            _, idct_m = data_utils.get_dct_matrix(seq_len)
            idct_m = Variable(torch.from_numpy(idct_m)).float().cuda()
            outputs_t = outputs.view(-1, seq_len).transpose(0, 1)
            outputs_exp = torch.matmul(idct_m, outputs_t).transpose(0, 1).contiguous().view(-1, dim_used_len,
                                                                                            seq_len).transpose(1, 2)
            pred_expmap = all_seq.clone()
            dim_used = np.array(dim_used)
            pred_expmap[:, :, dim_used] = outputs_exp
            targ_expmap = all_seq
            pred_expmap = pred_expmap.cpu().data.numpy()
            targ_expmap = targ_expmap.cpu().data.numpy()
            for k in range(8):
                plt.cla()
                figure_title = "action:{}, seq:{},".format(act, (k + 1))
                viz.plot_predictions(targ_expmap[k, :, :], pred_expmap[k, :, :], fig, ax, figure_title)
                plt.pause(1)


if __name__ == "__main__":
    option = Options().parse()
    main(option)
