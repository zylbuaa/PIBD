#!/usr/bin/env python3
# -*- encodinng: uft-8 -*-
'''
@file: valid_utils.py
@author:zyl
@contact:zhangyilan@buaa.edu.cn
@time:2024/4/27 21:15
'''

import torch
import os
from utils.core_utils import _get_splits,_init_model, _init_loaders, _extract_survival_metadata, _init_loss_function, _summary


def _get_val_results(args,model,train_loader,val_loader,log_file,loss_fn):
    all_survival = _extract_survival_metadata(train_loader, val_loader)
    _, val_cindex, val_cindex_ipcw, val_BS, val_IBS, val_iauc, total_loss = _summary(args.dataset_factory, model, args.omics_format, val_loader, loss_fn, all_survival)

    print('Best Val c-index: {:.4f} | Best Val c-index2: {:.4f} | Best Val IBS: {:.4f} | Best Val iauc: {:.4f}'.format(
        val_cindex,
        val_cindex_ipcw,
        val_IBS,
        val_iauc
    ))
    log_file.write(
        'Best Val c-index: {:.4f} | Best Val c-index2: {:.4f} | Best Val IBS: {:.4f} | Best Val iauc: {:.4f}\n'.format(
            val_cindex,
            val_cindex_ipcw,
            val_IBS,
            val_iauc
        ))

    return val_cindex, val_cindex_ipcw, val_BS, val_IBS, val_iauc, total_loss


def _val(datasets,cur,args,log_file):
    '''

        :param datasets: tuple
        :param cur: Int
        :param args: argspace.Namespace
        :param log_file: file
        :return:
        '''

    # ----> gets splits and summarize
    train_split, val_split = _get_splits(datasets, cur, args)

    # ----> initialize model
    model = _init_model(args)

    # ----> load params of model

    path = os.path.join(args.results_dir, "model_best_s{}.pth".format(cur))
    model.load_state_dict(torch.load(path), strict=True)
    print("Loaded model from {}".format(path))
    log_file.write("Loaded model from {}\n".format(path))

    # ----> init loss function
    loss_fn = _init_loss_function(args)

    # ----> initialize loaders
    train_loader, val_loader = _init_loaders(args, train_split, val_split)

    # ----> val
    val_cindex, val_cindex_ipcw, val_BS, val_IBS, val_iauc, total_loss = _get_val_results(args, model, train_loader,
                                                                                          val_loader, log_file, loss_fn)

    return (val_cindex, val_cindex_ipcw, val_BS, val_IBS, val_iauc, total_loss)