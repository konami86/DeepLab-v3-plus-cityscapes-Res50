#!/usr/bin/python
# -*- encoding: utf-8 -*-


from logger import *
from models.deeplabv3plus import Deeplab_v3plus
from cityscapes import CityScapes
from evaluate import MscEval
from optimizer import Optimizer
from loss import OhemCELoss
from configs import config_factory

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.nn.functional as F
import torch.distributed as dist

import os
import logging
import time
import datetime
import argparse


cfg = config_factory['resnet_cityscapes']
if not osp.exists(cfg.respth): os.makedirs(cfg.respth)


def parse_args():
    parse = argparse.ArgumentParser()
    parse.add_argument(
            '--local_rank',
            dest = 'local_rank',
            type = int,
            default = -1,
            )
    return parse.parse_args()


def train(verbose=True, **kwargs):
    args = kwargs['args']
    torch.cuda.set_device(args.local_rank)
    dist.init_process_group(
                backend = 'nccl',
                init_method = 'tcp://127.0.0.1:{}'.format(cfg.port),
                world_size = torch.cuda.device_count(),
                rank = args.local_rank
                )
    setup_logger(cfg.respth)
    logger = logging.getLogger()

    ## dataset
    ds = CityScapes(cfg, mode='train')
    sampler = torch.utils.data.distributed.DistributedSampler(ds)
    dl = DataLoader(ds,
                    batch_size = cfg.ims_per_gpu,
                    shuffle = False,
                    sampler = sampler,
                    num_workers = cfg.n_workers,
                    pin_memory = True,
                    drop_last = True)

    ## model
    net = Deeplab_v3plus(cfg)
    net.train()
    net.cuda()
    net = nn.parallel.DistributedDataParallel(net,
            device_ids = [args.local_rank, ],
            output_device = args.local_rank
            )
    n_min = cfg.ims_per_gpu*cfg.crop_size[0]*cfg.crop_size[1]//16
    criteria = OhemCELoss(thresh=cfg.ohem_thresh, n_min=n_min).cuda()

    ## optimizer
    optim = Optimizer(
            net,
            cfg.lr_start,
            cfg.momentum,
            cfg.weight_decay,
            cfg.warmup_steps,
            cfg.warmup_start_lr,
            cfg.max_iter,
            cfg.lr_power
            )

    ## train loop
    loss_avg = []
    st = glob_st = time.time()
    diter = iter(dl)
    n_epoch = 0
    #count for the epoch finished
    #已经跑结束的epoch
    epochF = 0
    bestMIOU = 0
    for it in range(cfg.max_iter):
        try:
            im, lb = next(diter)
            if not im.size()[0]==cfg.ims_per_gpu: continue
        except StopIteration:
            n_epoch += 1
            sampler.set_epoch(n_epoch)
            diter = iter(dl)
            im, lb = next(diter)
        im = im.cuda()
        lb = lb.cuda()

        H, W = im.size()[2:]
        lb = torch.squeeze(lb, 1)

        optim.zero_grad()
        logits = net(im)
        loss = criteria(logits, lb)
        loss.backward()
        optim.step()

        loss_avg.append(loss.item())
        ## print training log message 
        if it%cfg.msg_iter==0 and not it==0:
            loss_avg = sum(loss_avg) / len(loss_avg)
            lr = optim.lr
            ed = time.time()
            t_intv, glob_t_intv = ed - st, ed - glob_st
            eta = int((cfg.max_iter - it) * (glob_t_intv / it))
            eta = str(datetime.timedelta(seconds = eta))
            msg = ', '.join([
                    'iter: {it}/{max_it}',
                    'lr: {lr:4f}',
                    'loss: {loss:.4f}',
                    'eta: {eta}',
                    'time: {time:.4f}',
                ]).format(
                    it = it,
                    max_it = cfg.max_iter,
                    lr = lr,
                    loss = loss_avg,
                    time = t_intv,
                    eta = eta
                )
            logger.info(msg)
            loss_avg = []
            st = ed
         #每隔一段时间评估一次
        if n_epoch > epochF and n_epoch > 20:
            #置为相等的了
            epochF = n_epoch
        #if (n_epoch > 35) and it%(5*cfg.msg_iter) == 0 and not it==0:
            # net.cpu()
            # save_pth = osp.join(cfg.respth, 'model_final_best.pth')
            # state = net.module.state_dict() if hasattr(net, 'module') else net.state_dict()
            # if dist.get_rank()==0: torch.save(state, save_pth)
            # logger.info('training done, model saved to: {}'.format(save_pth))
            # logger.info('evaluating the final model')
            # net.cuda()
            net.eval()
            evaluator = MscEval(cfg)
            mIOU = evaluator(net)
            logger.info('mIOU is: {}'.format(mIOU))

            # 保存check point 
            save_pth = osp.join(cfg.respth, 'checkpoint.pth.tar')
            state = net.module.state_dict() if hasattr(net, 'module') else net.state_dict()
            if dist.get_rank()==0:
                stateF = {
                        'model': state,
                        'lr': optim.lr,
                        'mIOU': mIOU,
                        'it': it,
                        'epoch': n_epoch,
                        'optimizer': optim.optim.state_dict(),
                    }
                torch.save(stateF, save_pth)

            if mIOU > bestMIOU:
                logger.info('Get a new best mIMOU:{} at epoch:{}'.format(bestMIOU, n_epoch))
                #print('Get a new best mIMOU:{}'.format(bestMIOU))
                bestMIOU = mIOU
                #net.cpu()
                save_pth = osp.join(cfg.respth, 'model_final_{}.pth'.format(n_epoch))
                state = net.module.state_dict() if hasattr(net, 'module') else net.state_dict()
                if dist.get_rank()==0: torch.save(state, save_pth)
                #重新加载到cuda
                #net.cuda()
            
            net.train()
            

if __name__ == "__main__":
    args = parse_args()
    train(args=args)
