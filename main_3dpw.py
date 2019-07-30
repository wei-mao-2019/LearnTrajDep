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
from utils.pose3dpw import Pose3dPW
import utils.model as nnmodel
import utils.data_utils as data_utils


def main(opt):
    start_epoch = 0
    err_best = 10000
    lr_now = opt.lr
    is_cuda = torch.cuda.is_available()

    script_name = os.path.basename(__file__).split('.')[0]
    script_name = script_name + '_in{:d}_out{:d}_dctn_{:d}'.format(opt.input_n, opt.output_n, opt.dct_n)

    # create model
    print(">>> creating model")
    input_n = opt.input_n
    output_n = opt.output_n
    dct_n = opt.dct_n

    model = nnmodel.GCN(input_feature=dct_n, hidden_feature=opt.linear_size,
                        p_dropout=opt.dropout, num_stage=opt.num_stage, node_n=69)

    if is_cuda:
        model.cuda()

    print(">>> total params: {:.2f}M".format(sum(p.numel() for p in model.parameters()) / 1000000.0))
    optimizer = torch.optim.Adam(model.parameters(), lr=opt.lr)
    if opt.is_load:
        model_path_len = 'checkpoint/test/ckpt_main_last.pth.tar'
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
    train_dataset = Pose3dPW(path_to_data=opt.data_dir_3dpw, input_n=input_n, output_n=output_n, dct_n=dct_n, split=0)
    dim_used = train_dataset.dim_used
    test_dataset = Pose3dPW(path_to_data=opt.data_dir_3dpw, input_n=input_n, output_n=output_n, dct_n=dct_n, split=1)
    val_dataset = Pose3dPW(path_to_data=opt.data_dir_3dpw, input_n=input_n, output_n=output_n, dct_n=dct_n, split=2)

    # load dadasets for training
    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=opt.train_batch,
        shuffle=True,
        num_workers=opt.job,
        pin_memory=True)
    test_loader = DataLoader(
        dataset=test_dataset,
        batch_size=opt.test_batch,
        shuffle=False,
        num_workers=opt.job,
        pin_memory=True)
    val_loader = DataLoader(
        dataset=val_dataset,
        batch_size=opt.test_batch,
        shuffle=False,
        num_workers=opt.job,
        pin_memory=True)
    print(">>> data loaded !")
    print(">>> train data {}".format(train_dataset.__len__()))
    print(">>> test data {}".format(test_dataset.__len__()))
    print(">>> validation data {}".format(val_dataset.__len__()))

    for epoch in range(start_epoch, opt.epochs):

        if (epoch + 1) % opt.lr_decay == 0:
            lr_now = utils.lr_decay(optimizer, lr_now, opt.lr_gamma)
        print('==========================')
        print('>>> epoch: {} | lr: {:.5f}'.format(epoch + 1, lr_now))
        ret_log = np.array([epoch + 1])
        head = np.array(['epoch'])
        # per epoch
        lr_now, t_l, t_err = train(train_loader, model,
                                   optimizer,
                                   input_n=input_n,
                                   dct_n=dct_n,
                                   dim_used=dim_used,
                                   lr_now=lr_now,
                                   max_norm=opt.max_norm,
                                   is_cuda=is_cuda)
        ret_log = np.append(ret_log, [lr_now, t_l, t_err])
        head = np.append(head, ['lr', 't_l', 't_err'])

        v_err = val(val_loader, model,
                    input_n=input_n,
                    dct_n=dct_n,
                    dim_used=dim_used,
                    is_cuda=is_cuda)

        ret_log = np.append(ret_log, v_err)
        head = np.append(head, ['v_err'])

        test_3d = test(test_loader, model,
                       input_n=input_n,
                       output_n=output_n,
                       dct_n=dct_n,
                       dim_used=dim_used,
                       is_cuda=is_cuda)
        # ret_log = np.append(ret_log, test_l)
        ret_log = np.append(ret_log, test_3d)
        if output_n == 15:
            head = np.append(head, ['1003d', '2003d', '3003d', '4003d', '5003d'])
        elif output_n == 30:
            head = np.append(head, ['1003d', '2003d', '3003d', '4003d', '5003d', '6003d', '7003d', '8003d', '9003d',
                                    '10003d'])

        # update log file
        df = pd.DataFrame(np.expand_dims(ret_log, axis=0))
        if epoch == start_epoch:
            df.to_csv(opt.ckpt + '/' + script_name + '.csv', header=head, index=False)
        else:
            with open(opt.ckpt + '/' + script_name + '.csv', 'a') as f:
                df.to_csv(f, header=False, index=False)
        # save ckpt
        is_best = v_err < err_best
        err_best = min(v_err, err_best)
        file_name = ['ckpt_' + script_name + '_best.pth.tar', 'ckpt_' + script_name + '_last.pth.tar']
        utils.save_ckpt({'epoch': epoch + 1,
                         'lr': lr_now,
                         'err': test_3d[0],
                         'state_dict': model.state_dict(),
                         'optimizer': optimizer.state_dict()},
                        ckpt_path=opt.ckpt,
                        is_best=is_best,
                        file_name=file_name)


def train(train_loader, model, optimizer, input_n=10, dct_n=20, dim_used=[], lr_now=None, max_norm=True, is_cuda=False):
    sen_l = utils.AccumLoss()
    eul_err = utils.AccumLoss()

    model.train()
    st = time.time()
    bar = Bar('>>>', fill='>', max=len(train_loader))
    for i, (inputs, targets, all_seq) in enumerate(train_loader):

        batch_size = inputs.shape[0]
        # batch size is 1 do not train
        if batch_size == 1:
            break

        bt = time.time()

        if is_cuda:
            inputs = Variable(inputs.cuda()).float()
            # targets = Variable(targets.cuda(async=True)).float()
            all_seq = Variable(all_seq.cuda(async=True)).float()
        else:
            inputs = Variable(inputs).float()
            # targets = Variable(targets).float()
            all_seq = Variable(all_seq).float()
        outputs = model(inputs)
        loss = loss_funcs.sen_loss(outputs, all_seq, dim_used, dct_n)

        # calculate loss and backward
        optimizer.zero_grad()
        loss.backward()
        if max_norm:
            nn.utils.clip_grad_norm(model.parameters(), max_norm=1)
        optimizer.step()
        n, seq_len, _ = all_seq.data.shape
        # update the training loss
        e_err = loss_funcs.euler_error(outputs, all_seq, input_n, dim_used, dct_n)
        sen_l.update(loss.cpu().data.numpy()[0] * n * seq_len, n * seq_len)
        eul_err.update(e_err.cpu().data.numpy()[0] * n * seq_len, n * seq_len)

        bar.suffix = '{}/{}|batch time {:.4f}s|total time{:.2f}s'.format(i, len(train_loader), time.time() - bt,
                                                                         time.time() - st)
        bar.next()
    bar.finish()
    return lr_now, sen_l.avg, eul_err.avg


def test(train_loader, model, input_n=20, dct_n=20, dim_used=[], output_n=50, is_cuda=False):
    N = 0
    if output_n == 15:
        eval_frame = [2, 5, 8, 11, 14]
    elif output_n == 30:
        eval_frame = [2, 5, 8, 11, 14, 17, 20, 23, 26, 29]
    t_err = np.zeros(len(eval_frame))

    model.eval()
    st = time.time()
    bar = Bar('>>>', fill='>', max=len(train_loader))
    for i, (inputs, targets, all_seq) in enumerate(train_loader):
        bt = time.time()

        if is_cuda:
            inputs = Variable(inputs.cuda()).float()
            # targets = Variable(targets.cuda(async=True)).float()
            all_seq = Variable(all_seq.cuda(async=True)).float()
        else:
            inputs = Variable(inputs).float()
            # targets = Variable(targets).float()
            all_seq = Variable(all_seq).float()
        outputs = model(inputs)

        n, seq_len, dim_full_len = all_seq.data.shape

        _, idct_m = data_utils.get_dct_matrix(seq_len)
        idct_m = Variable(torch.from_numpy(idct_m)).float().cuda()
        outputs_t = outputs.view(-1, dct_n).transpose(0, 1)
        outputs_exp = torch.matmul(idct_m[:, :dct_n], outputs_t).transpose(0, 1).contiguous().view(-1, dim_full_len - 3,
                                                                                                   seq_len).transpose(1,
                                                                                                                      2)
        pred_exp = all_seq.clone()
        pred_exp[:, :, dim_used] = outputs_exp
        pred_exp = pred_exp.contiguous().view(n, seq_len, -1)[:, input_n:, :]
        targ_exp = all_seq.contiguous().view(n, seq_len, -1)[:, input_n:, :]

        for k in np.arange(0, len(eval_frame)):
            j = eval_frame[k]
            t_err[k] += torch.mean(torch.norm(targ_exp[:, j, :] - pred_exp[:, j, :], p=2, dim=1)).cpu().data.numpy()[
                            0] * n

        # update the training loss
        N += n

        bar.suffix = '{}/{}|batch time {:.4f}s|total time{:.2f}s'.format(i, len(train_loader), time.time() - bt,
                                                                         time.time() - st)
        bar.next()
    bar.finish()
    return t_err / N


def val(train_loader, model, input_n=10, dct_n=20, dim_used=[], is_cuda=False):
    t_err = utils.AccumLoss()

    model.eval()
    st = time.time()
    bar = Bar('>>>', fill='>', max=len(train_loader))
    for i, (inputs, targets, all_seq) in enumerate(train_loader):
        bt = time.time()

        if is_cuda:
            inputs = Variable(inputs.cuda()).float()
            # targets = Variable(targets.cuda(async=True)).float()
            all_seq = Variable(all_seq.cuda(async=True)).float()
        else:
            inputs = Variable(inputs).float()
            # targets = Variable(targets).float()
            all_seq = Variable(all_seq).float()
        outputs = model(inputs)
        e_err = loss_funcs.euler_error(outputs, all_seq, input_n, dim_used, dct_n)

        n, seq_len, _ = all_seq.data.shape
        # update the training loss
        t_err.update(e_err.cpu().data.numpy()[0] * n * seq_len, n * seq_len)

        bar.suffix = '{}/{}|batch time {:.4f}s|total time{:.2f}s'.format(i, len(train_loader), time.time() - bt,
                                                                         time.time() - st)
        bar.next()
    bar.finish()
    return t_err.avg


if __name__ == "__main__":
    option = Options().parse()
    main(option)
