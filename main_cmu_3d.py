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

from utils import loss_funcs, utils as utils
from utils.opt import Options
from utils.cmu_motion_3d import CMU_Motion3D
import utils.model as nnmodel
import utils.data_utils as data_utils


def main(opt):
    start_epoch = 0
    err_best = 10000
    lr_now = opt.lr
    is_cuda = torch.cuda.is_available()

    # save option in log
    script_name = os.path.basename(__file__).split('.')[0]
    script_name = script_name + '_in{:d}_out{:d}_dctn{:d}'.format(opt.input_n, opt.output_n, opt.dct_n)

    # create model
    print(">>> creating model")
    input_n = opt.input_n
    output_n = opt.output_n
    dct_n = opt.dct_n

    model = nnmodel.GCN(input_feature=dct_n, hidden_feature=opt.linear_size,
                        p_dropout=opt.dropout, num_stage=opt.num_stage, node_n=75)

    if is_cuda:
        model.cuda()
    print(">>> total params: {:.2f}M".format(sum(p.numel() for p in model.parameters()) / 1000000.0))
    optimizer = torch.optim.Adam(model.parameters(), lr=opt.lr)
    if opt.is_load:
        model_path_len = 'checkpoint/test/ckpt_main_gcn_muti_att_best.pth.tar'
        print(">>> loading ckpt len from '{}'".format(model_path_len))
        if is_cuda:
            ckpt = torch.load(model_path_len)
        else:
            ckpt = torch.load(model_path_len, map_location='cpu')
        start_epoch = ckpt['epoch']
        err_best = ckpt['err']
        lr_now = ckpt['lr']
        model.load_state_dict(ckpt['state_dict'])
        optimizer.load_state_dict(ckpt['optimizer'])
        print(">>> ckpt len loaded (epoch: {} | err: {})".format(start_epoch, err_best))

    # data loading
    print(">>> loading data")
    train_dataset = CMU_Motion3D(path_to_data=opt.data_dir_cmu, actions=opt.actions, input_n=input_n, output_n=output_n,
                                 split=0, dct_n=opt.dct_n)
    data_std = train_dataset.data_std
    data_mean = train_dataset.data_mean
    dim_used = train_dataset.dim_used

    acts = data_utils.define_actions_cmu(opt.actions)
    test_data = dict()
    for act in acts:
        test_dataset = CMU_Motion3D(path_to_data=opt.data_dir_cmu, actions=act, input_n=input_n, output_n=output_n,
                                    split=1, data_mean=data_mean, data_std=data_std, dim_used=dim_used, dct_n=dct_n)
        test_data[act] = DataLoader(
            dataset=test_dataset,
            batch_size=opt.test_batch,
            shuffle=False,
            num_workers=opt.job,
            pin_memory=True)
    # val_dataset = CMU_Motion(path_to_data=opt.data_dir, actions='all', input_n=input_n, output_n=output_n,
    #                         split=2,
    #                         sample_rate=sample_rate, is_norm_dct=opt.is_norm_dct, is_norm_motion=opt.is_norm,
    #                         data_mean=data_mean, data_std=data_std, data_mean_dct=data_mean_dct,
    #                         data_std_dct=data_std_dct)
    # load dadasets for training
    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=opt.train_batch,
        shuffle=True,
        num_workers=opt.job,
        pin_memory=True)
    # val_loader = DataLoader(
    #     dataset=val_dataset,
    #     batch_size=opt.test_batch,
    #     shuffle=False,
    #     num_workers=opt.job,
    #     pin_memory=True)
    print(">>> data loaded !")
    print(">>> train data {}".format(train_dataset.__len__()))
    print(">>> test data {}".format(test_dataset.__len__()))
    # print(">>> validation data {}".format(val_dataset.__len__()))

    for epoch in range(start_epoch, opt.epochs):

        if (epoch + 1) % opt.lr_decay == 0:
            lr_now = utils.lr_decay(optimizer, lr_now, opt.lr_gamma)
        print('==========================')
        print('>>> epoch: {} | lr: {:.5f}'.format(epoch + 1, lr_now))
        ret_log = np.array([epoch + 1])
        head = np.array(['epoch'])
        # per epoch
        lr_now, t_l, t_e, t_3d = train(train_loader, model, optimizer, lr_now=lr_now, max_norm=opt.max_norm,
                                       is_cuda=is_cuda, dim_used=dim_used, dct_n=opt.dct_n)
        ret_log = np.append(ret_log, [lr_now, t_l, t_3d])
        head = np.append(head, ['lr', 't_l', 't_3d'])

        # v_l, v_e, v_3d = val(val_loader, model, optimizer, adj, input_n=input_n, output_n=output_n,
        #                      lr_now=lr_now, max_norm=opt.max_norm, is_cuda=is_cuda, epoch=epoch + 1,
        #                      dim_used=train_dataset.dim_used, is_norm_dct=opt.is_norm_dct,
        #                      is_norm=opt.is_norm, data_mean=data_mean, data_std=data_std,
        #                      data_mean_dct=data_mean_dct, data_std_dct=data_std_dct)
        #
        # ret_log = np.append(ret_log, [v_l, v_e, v_3d])
        # head = np.append(head, ['v_l', 'v_e', 'v_3d'])

        test_3d_temp = np.array([])
        test_3d_head = np.array([])
        for act in acts:
            test_3d = test(test_data[act], model, input_n=input_n, output_n=output_n,
                           is_cuda=is_cuda, dim_used=dim_used, dct_n=opt.dct_n)
            test_3d_temp = np.append(test_3d_temp, test_3d)
            test_3d_head = np.append(test_3d_head,
                                     [act + '3d80', act + '3d160', act + '3d320', act + '3d400'])
            if output_n > 10:
                test_3d_head = np.append(test_3d_head,
                                         [act + '3d560', act + '3d1000'])
        ret_log = np.append(ret_log, test_3d_temp)
        head = np.append(head, test_3d_head)

        # update log file
        df = pd.DataFrame(np.expand_dims(ret_log, axis=0))
        if epoch == start_epoch:
            df.to_csv(opt.ckpt + '/' + script_name + '.csv', header=head, index=False)
        else:
            with open(opt.ckpt + '/' + script_name + '.csv', 'a') as f:
                df.to_csv(f, header=False, index=False)
        # save ckpt
        if not np.isnan(t_3d):
            is_best = t_3d < err_best
            err_best = min(t_3d, err_best)
        else:
            is_best = False
        file_name = ['ckpt_' + script_name + '_best.pth.tar', 'ckpt_' + script_name + '_last.pth.tar']
        utils.save_ckpt({'epoch': epoch + 1,
                         'lr': lr_now,
                         'err': test_3d[0],
                         'state_dict': model.state_dict(),
                         'optimizer': optimizer.state_dict()},
                        ckpt_path=opt.ckpt,
                        is_best=is_best,
                        file_name=file_name)


def train(train_loader, model, optimizer, lr_now=None, max_norm=True, is_cuda=False, dim_used=[], dct_n=15):
    t_l = utils.AccumLoss()
    t_e = utils.AccumLoss()
    t_3d = utils.AccumLoss()

    model.train()
    st = time.time()
    bar = Bar('>>>', fill='>', max=len(train_loader))
    for i, (inputs, targets, all_seq) in enumerate(train_loader):
        bt = time.time()

        if is_cuda:
            inputs = Variable(inputs.cuda()).float()
            all_seq = Variable(all_seq.cuda(async=True)).float()

        outputs = model(inputs)

        n, seq_len, dim_full_len = all_seq.data.shape
        dim_used = np.array(dim_used)
        dim_used_len = len(dim_used)

        _, idct_m = data_utils.get_dct_matrix(seq_len)
        idct_m = Variable(torch.from_numpy(idct_m)).float().cuda()
        outputs_t = outputs.view(-1, dct_n).transpose(0, 1)
        outputs_p3d = torch.matmul(idct_m[:, :dct_n], outputs_t).transpose(0, 1).contiguous()
        outputs_p3d = outputs_p3d.view(-1, dim_used_len, seq_len).transpose(1, 2).contiguous().view(-1, 3)

        targ_p3d = all_seq[:, :, dim_used].clone().contiguous().view(-1, 3)

        loss = torch.mean(torch.norm(targ_p3d - outputs_p3d, 2, 1))

        # calculate loss and backward
        optimizer.zero_grad()
        loss.backward()
        if max_norm:
            nn.utils.clip_grad_norm(model.parameters(), max_norm=1)
        optimizer.step()
        t_l.update(loss.cpu().data.numpy()[0] * n, n)

        bar.suffix = '{}/{}|batch time {:.4f}s|total time{:.2f}s'.format(i + 1, len(train_loader), time.time() - bt,
                                                                         time.time() - st)
        bar.next()
    bar.finish()
    return lr_now, t_l.avg, t_e.avg, t_3d.avg


def test(train_loader, model, input_n=20, output_n=50, is_cuda=False, dim_used=[], dct_n=15):
    N = 0
    if output_n == 25:
        eval_frame = [1, 3, 7, 9, 13, 24]
    elif output_n == 10:
        eval_frame = [1, 3, 7, 9]
    t_3d = np.zeros(len(eval_frame))

    model.eval()
    st = time.time()
    bar = Bar('>>>', fill='>', max=len(train_loader))
    for i, (inputs, targets, all_seq) in enumerate(train_loader):
        bt = time.time()

        if is_cuda:
            inputs = Variable(inputs.cuda()).float()
            all_seq = Variable(all_seq.cuda(async=True)).float()

        outputs = model(inputs)

        n, seq_len, dim_full_len = all_seq.data.shape
        dim_used = np.array(dim_used)
        dim_used_len = len(dim_used)

        _, idct_m = data_utils.get_dct_matrix(seq_len)
        idct_m = Variable(torch.from_numpy(idct_m)).float().cuda()
        outputs_t = outputs.view(-1, dct_n).transpose(0, 1)
        outputs_3d = torch.matmul(idct_m[:, :dct_n], outputs_t).transpose(0, 1).contiguous().view \
            (-1, dim_used_len, seq_len).transpose(1, 2)
        pred_3d = all_seq.clone()
        dim_used = np.array(dim_used)

        # deal with joints at same position
        joint_to_ignore = np.array([16, 20, 29, 24, 27, 33, 36])
        index_to_ignore = np.concatenate((joint_to_ignore * 3, joint_to_ignore * 3 + 1, joint_to_ignore * 3 + 2))
        joint_equal = np.array([15, 15, 15, 23, 23, 32, 32])
        index_to_equal = np.concatenate((joint_equal * 3, joint_equal * 3 + 1, joint_equal * 3 + 2))

        pred_3d[:, :, dim_used] = outputs_3d
        pred_3d[:, :, index_to_ignore] = pred_3d[:, :, index_to_equal]
        pred_p3d = pred_3d.contiguous().view(n, seq_len, -1, 3)[:, input_n:, :, :]
        targ_p3d = all_seq.contiguous().view(n, seq_len, -1, 3)[:, input_n:, :, :]

        for k in np.arange(0, len(eval_frame)):
            j = eval_frame[k]
            t_3d[k] += torch.mean(torch.norm(
                targ_p3d[:, j, :, :].contiguous().view(-1, 3) - pred_p3d[:, j, :, :].contiguous().view(-1, 3), 2,
                1)).cpu().data.numpy()[0] * n

        N += n

        bar.suffix = '{}/{}|batch time {:.4f}s|total time{:.2f}s'.format(i + 1, len(train_loader), time.time() - bt,
                                                                         time.time() - st)
        bar.next()
    bar.finish()
    return t_3d / N


# def val(train_loader, model, optimizer, input_n=20, output_n=50, lr_now=None, max_norm=True, is_cuda=False,
#         epoch=1, dim_used=[], is_norm_dct=False, is_norm=False, data_mean_dct=0, data_std_dct=0, data_mean=0,
#         data_std=0):
#     t_l = utils.AccumLoss()
#     t_e = utils.AccumLoss()
#     t_3d = utils.AccumLoss()
#
#     data_mean_dct = Variable(torch.from_numpy(data_mean_dct)).float().cuda()
#     data_std_dct = Variable(torch.from_numpy(data_std_dct)).float().cuda()
#     data_mean = Variable(torch.from_numpy(data_mean)).float().cuda()
#     data_std = Variable(torch.from_numpy(data_std)).float().cuda()
#
#     model.eval()
#     st = time.time()
#     bar = Bar('>>>', fill='>', max=len(train_loader))
#     for i, (inputs, targets, all_seq, one_hot) in enumerate(train_loader):
#         bt = time.time()
#
#         if is_cuda:
#             inputs = Variable(inputs.cuda()).float()
#             targets = Variable(targets.cuda(async=True)).float()
#             all_seq = Variable(all_seq.cuda(async=True)).float()
#
#         outputs = model(inputs)
#         n = outputs.shape[0]
#         outputs = outputs.view(n, -1)
#         targets = targets.view(n, -1)
#         if is_norm_dct:
#             outputs = torch.mul(outputs, data_std_dct.repeat(n, 1)) + data_mean_dct.repeat(n, 1)
#             targets = torch.mul(targets, data_std_dct.repeat(n, 1)) + data_mean_dct.repeat(n, 1)
#
#         loss = torch.mean(torch.norm(outputs - targets, 2, 1))
#
#         n, _, _ = all_seq.data.shape
#
#         m_err = loss_funcs.mpjpe_error(outputs, all_seq, input_n, dim_used, is_norm=is_norm, data_mean=data_mean,
#                                        data_std=data_std)
#         e_err = loss_funcs.euler_error(outputs, all_seq, input_n, dim_used, is_norm=is_norm, data_mean=data_mean,
#                                        data_std=data_std)
#
#         # update the training loss
#         t_l.update(loss.cpu().data.numpy()[0] * n, n)
#         t_e.update(e_err.cpu().data.numpy()[0] * n, n)
#         t_3d.update(m_err.cpu().data.numpy()[0] * n, n)
#
#         bar.suffix = '{}/{}|batch time {:.4f}s|total time{:.2f}s'.format(i, len(train_loader), time.time() - bt,
#                                                                          time.time() - st)
#         bar.next()
#     bar.finish()
#     return t_l.avg, t_e.avg, t_3d.avg


if __name__ == "__main__":
    option = Options().parse()
    main(option)
