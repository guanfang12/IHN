from __future__ import print_function, division
import sys
sys.path.append('core')

import argparse
import os
import cv2
import time
import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from torch.utils.data import DataLoader
from network import IHN
import evaluate
import datasets_4cor_img as datasets
from utils import *

from torch.utils.tensorboard import SummaryWriter

try:
    from torch.cuda.amp import GradScaler
except:
    # dummy GradScaler for PyTorch < 1.6
    class GradScaler:
        def __init__(self):
            pass
        def scale(self, loss):
            return loss
        def unscale_(self, optimizer):
            pass
        def step(self, optimizer):
            optimizer.step()
        def update(self):
            pass


# exclude extremly large displacements
MAX_FLOW = 400
SUM_FREQ = 100
VAL_FREQ = 5000


def sequence_loss(four_pred, flow_gt, gamma=0.8, max_flow=MAX_FLOW):
    """ Loss function defined over sequence of flow predictions """

    n_predictions = len(four_pred)    
    flow_loss = 0.0

    flow_4cor = torch.zeros((four_pred[0].shape[0], 2, 2, 2)).cuda()
    flow_4cor[:, :, 0, 0] = flow_gt[:, :, 0, 0]
    flow_4cor[:, :, 0, 1] = flow_gt[:, :, 0, -1]
    flow_4cor[:, :, 1, 0] = flow_gt[:, :, -1, 0]
    flow_4cor[:, :, 1, 1] = flow_gt[:, :, -1, -1]

    for i in range(n_predictions):
        i_weight = gamma**(n_predictions - i - 1)
        
        i_loss = (four_pred[i] - flow_4cor).abs()
        flow_loss += i_weight * i_loss.mean()

    epe = torch.sum((four_pred[-1] - flow_4cor)**2, dim=1).sqrt()

    metrics = { # 表示指标，用于给人看效果如何
        'epe': epe.mean().item(), # 所有角点的平均估计误差
        '1px': (epe < 1).float().mean().item(), # 表示像素误差小于1的所占百分比
        '3px': (epe < 3).float().mean().item(),
        '5px': (epe < 5).float().mean().item(),
    }

    return flow_loss, metrics


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def fetch_optimizer(args, model):
    """ Create the optimizer and learning rate scheduler """
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wdecay, eps=args.epsilon)

    scheduler = optim.lr_scheduler.OneCycleLR(optimizer, args.lr, args.num_steps+100,
        pct_start=0.05, cycle_momentum=False, anneal_strategy='linear')

    return optimizer, scheduler
    

class Logger:
    def __init__(self, model, scheduler):
        self.model = model
        self.scheduler = scheduler
        self.total_steps = 0
        self.running_loss = {}
        self.writer = None

    def _print_training_status(self):
        metrics_data = [self.running_loss[k]/SUM_FREQ for k in sorted(self.running_loss.keys())]
        training_str = "[{:6d}, {:10.7f}] ".format(self.total_steps+1, self.scheduler.get_last_lr()[0])
        metrics_str = ("{:10.4f}, "*len(metrics_data)).format(*metrics_data)
        
        # print the training status
        print(training_str + metrics_str)

        if self.writer is None:
            self.writer = SummaryWriter('save')

        for k in self.running_loss:
            self.writer.add_scalar(k, self.running_loss[k]/SUM_FREQ, self.total_steps)
            self.running_loss[k] = 0.0

    def push(self, metrics):
        self.total_steps += 1

        for key in metrics:
            if key not in self.running_loss:
                self.running_loss[key] = 0.0

            self.running_loss[key] += metrics[key]

        if self.total_steps % SUM_FREQ == SUM_FREQ-1:
            self._print_training_status()
            self.running_loss = {}

    def write_dict(self, results):
        if self.writer is None:
            self.writer = SummaryWriter()

        for key in results:
            self.writer.add_scalar(key, results[key], self.total_steps)

    def close(self):
        self.writer.close()


def train(args):

    model = nn.DataParallel(IHN(args), device_ids=args.gpuid)
    print("Parameter Count: %d" % count_parameters(model))

    if args.restore_ckpt is not None:
        model.load_state_dict(torch.load(args.restore_ckpt), strict=False)
        print("Have load state_dict from: {}".format(args.restore_ckpt))

    model.cuda()
    model.train()

    # if args.dataset != 'mscoco':
    #     model.module.freeze_bn()

    # # 自定义梯度回调函数
    # def print_grad_layerwise(grad):
    #     print("{} ".format(grad.shape))

    # for name, param in model.module.named_parameters():
    #     hook = param.register_hook(print_grad_layerwise)
    #     hook.owner = name
    #     print(hook.owner)

    train_loader = datasets.fetch_dataloader(args)
    optimizer, scheduler = fetch_optimizer(args, model)

    total_steps = 0
    scaler = GradScaler(enabled=args.mixed_precision)
    logger = Logger(model, scheduler)

    VAL_FREQ = 5000
    add_noise = True

    should_keep_training = True
    while should_keep_training:

        for i_batch, data_blob in enumerate(train_loader):
            optimizer.zero_grad()
            image1, image2, flow_gt,  H  = [x.cuda() for x in data_blob]
            # image1, image2, flow, valid = [x.cuda() for x in data_blob]

            if args.add_noise:
                stdv = np.random.uniform(0.0, 5.0)
                image1 = (image1 + stdv * torch.randn(*image1.shape).cuda()).clamp(0.0, 255.0)
                image2 = (image2 + stdv * torch.randn(*image2.shape).cuda()).clamp(0.0, 255.0)

            four_pred = model(image1, image2, iters_lev0=args.iters_lev0, iters_lev1=args.iters_lev1)            

            loss, metrics = sequence_loss(four_pred, flow_gt, args.gamma)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)                
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)
            
            scaler.step(optimizer)
            scheduler.step()
            scaler.update()
            # exit()

            logger.push(metrics)

            if total_steps % VAL_FREQ == VAL_FREQ - 1:
                PATH = 'checkpoints/%d_%s.pth' % (total_steps+1, args.name)
                torch.save(model.state_dict(), PATH)
                print("state_dict has saved at: {}".format(PATH))

                results = {}
                results.update(evaluate.validate_process(model.module, total_steps, args))

                logger.write_dict(results)
                
                # model.train()  # evaluate.validate_process 中已经model.train() 
                # if args.stage != 'chairs':
                #     model.module.freeze_bn()
            
            total_steps += 1

            if total_steps > args.num_steps:
                should_keep_training = False
                break

    logger.close()
    PATH = 'checkpoints/%s.pth' % args.name
    torch.save(model.state_dict(), PATH)

    return PATH


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', default='results/mscoco_lev1/IHN.pth',help="restore checkpoint")
    parser.add_argument('--iters_lev0', type=int, default=6)
    parser.add_argument('--iters_lev1', type=int, default=6)
    parser.add_argument('--mixed_precision', default=False, action='store_true',
                        help='use mixed precision')
    parser.add_argument('--dropout', type=float, default=0.0)
    parser.add_argument('--gpuid', type=int, nargs='+', default=[0])
    parser.add_argument('--savemat', type=str,  default='mscoco.mat')
    parser.add_argument('--savedict', type=str, default='mscoco.npy')
    parser.add_argument('--dataset', type=str, default='mscoco', help='dataset')    
    parser.add_argument('--lev0', default=True, action='store_true',
                        help='warp no')
    parser.add_argument('--lev1', default=False, action='store_true',
                        help='warp once')
    parser.add_argument('--weight', default=False, action='store_true',
                        help='weight')
    parser.add_argument('--model_name_lev0', default='', help='specify model0 name')
    parser.add_argument('--model_name_lev1', default='', help='specify model0 name')

    parser.add_argument('--name', default='IHN', help="name your experiment")
    parser.add_argument('--restore_ckpt', help="restore checkpoint")
    parser.add_argument('--validation', type=str, nargs='+')

    parser.add_argument('--lr', type=float, default=0.00025)
    parser.add_argument('--num_steps', type=int, default=120000)
    parser.add_argument('--batch_size', type=int, default=8)

    parser.add_argument('--wdecay', type=float, default=.00005)
    parser.add_argument('--epsilon', type=float, default=1e-8)
    parser.add_argument('--clip', type=float, default=1.0)
    parser.add_argument('--gamma', type=float, default=0.85, help='exponential weighting')
    parser.add_argument('--add_noise', default=False, action='store_true')
    args = parser.parse_args()

    setup_seed(2023)

    if not os.path.isdir('checkpoints'):
        os.mkdir('checkpoints')

    train(args)