#-*- coding:utf-8 -*-

from __future__ import division
from __future__ import absolute_import
from __future__ import print_function

import os
import time
import torch
import argparse
import torch.nn as nn
import torch.optim as optim
import torch.nn.init as init
import torch.utils.data as data
import numpy as np
from torch.autograd import Variable
import torch.backends.cudnn as cudnn

from data.config import cfg
from models.Res50_RFP import build_model
from layers.modules import MultiBoxLoss
from data.factory import dataset_factory, detection_collate

#os.environ['CUDA_LAUNCH_BLOCKING'] = '1'


def str2bool(v):
    return v.lower() in ("yes", "true", "t", "1")

parser = argparse.ArgumentParser(
    description='face Detector Training With Pytorch')
train_set = parser.add_mutually_exclusive_group()
parser.add_argument('--dataset',
                    default='face',
                    choices=['hand', 'face', 'head'],
                    help='Train target')
parser.add_argument('--batch_size',
                    default=32, type=int,
                    help='Batch size for training')
parser.add_argument('--resume',
                    default=None, type=str,
                    help='Checkpoint state_dict file to resume training from')
parser.add_argument('--num_workers',
                    default=8, type=int,
                    help='Number of workers used in dataloading')
parser.add_argument('--cuda',
                    default=True, type=str2bool,
                    help='Use CUDA to train model')
parser.add_argument('--lr', '--learning-rate',
                    default=5e-3, type=float,
                    help='initial learning rate')
parser.add_argument('--momentum',
                    default=0.9, type=float,
                    help='Momentum value for optim')
parser.add_argument('--weight_decay',
                    default=5e-4, type=float,
                    help='Weight decay for SGD')
parser.add_argument('--gamma',
                    default=0.1, type=float,
                    help='Gamma update for SGD')
parser.add_argument('--multigpu',
                    default=True, type=str2bool,
                    help='Use mutil Gpu training')
parser.add_argument('--save_folder',
                    default='weights/',
                    help='Directory for saving checkpoint models')
args = parser.parse_args()


if torch.cuda.is_available():
    if args.cuda:
        torch.set_default_tensor_type('torch.cuda.FloatTensor')
    if not args.cuda:
        print("WARNING: It looks like you have a CUDA device, but aren't " +
              "using CUDA.\nRun with --cuda for optimal training speed.")
        torch.set_default_tensor_type('torch.FloatTensor')
else:
    torch.set_default_tensor_type('torch.FloatTensor')

if not os.path.exists(args.save_folder):
    os.makedirs(args.save_folder)


train_dataset, val_dataset = dataset_factory(args.dataset)

train_loader = data.DataLoader(train_dataset, args.batch_size,
                               num_workers=args.num_workers,
                               shuffle=True,
                               collate_fn=detection_collate,
                               pin_memory=True)

val_batchsize = args.batch_size // 2
val_loader = data.DataLoader(val_dataset, val_batchsize,
                             num_workers=args.num_workers,
                             shuffle=False,
                             collate_fn=detection_collate,
                             pin_memory=True)

min_loss = np.inf
start_epoch = 0
facedet = build_model('train', cfg.NUM_CLASSES)
net = facedet


if args.resume:
    print('Resuming training, loading {}...'.format(args.resume))
    start_epoch = net.load_weights(args.resume)

if args.cuda:
    if args.multigpu:
        net = torch.nn.DataParallel(facedet)
    net = net.cuda()
    cudnn.benckmark = True

if not args.resume:
    print('Initializing weights...')
    facedet.layer6.apply(facedet.weights_init)
    facedet.layer7.apply(facedet.weights_init)
    facedet.fpn.apply(facedet.weights_init)
    facedet.MR1.apply(facedet.weights_init)
    facedet.MR2.apply(facedet.weights_init)
    facedet.MR3.apply(facedet.weights_init)
    facedet.MR4.apply(facedet.weights_init)
    facedet.MR5.apply(facedet.weights_init)
    facedet.MR6.apply(facedet.weights_init)
    facedet.loc.apply(facedet.weights_init)
    facedet.conf.apply(facedet.weights_init)

optimizer = optim.SGD(net.parameters(), lr=args.lr, momentum=args.momentum,
                      weight_decay=args.weight_decay)
criterion = MultiBoxLoss(cfg, args.dataset, args.cuda)
print('Loading wider dataset...')
print('Using the specified args:')
print(args)
epoch_size = len(train_dataset) // args.batch_size

def train():
    step_index = 0
    iteration = 0
    net.train()
    for epoch in range(start_epoch, cfg.EPOCHES):
        losses = 0
        for batch_idx, (images, targets) in enumerate(train_loader):
            if args.cuda:
                images = Variable(images.cuda())
                targets = [Variable(ann.cuda(), volatile=True)
                           for ann in targets]
            else:
                images = Variable(images)
                targets = [Variable(ann, volatile=True) for ann in targets]

            if iteration in cfg.LR_STEPS:
                step_index += 1
            adjust_learning_rate(optimizer, args.gamma, epoch, step_index,iteration, epoch_size)

            t0 = time.time()
            out = net(images)
            # backprop
            optimizer.zero_grad()
            loss_l, loss_c = criterion(out, targets)
            loss = loss_l + loss_c
            loss.backward()
            optimizer.step()
            t1 = time.time()
            losses += loss.data[0]

            if iteration % 20 == 0:
                tloss = losses / (batch_idx + 1)
                print('Timer: %.4f' % (t1 - t0))
                print('epoch:' + repr(epoch) + ' || iter:' +
                      repr(iteration) + ' || Loss:%.4f' % (tloss))
                print('->> conf loss:{:.4f} || loc loss:{:.4f}'.format(
                    loss_c.data[0], loss_l.data[0]))
                print('->>lr:{:.6f}'.format(optimizer.param_groups[0]['lr']))

            if iteration != 0 and iteration % 10000 == 0:
                print('Saving state, iter:', iteration)
                file = 'Res50_RFP' +'_' + repr(iteration) + '.pth'
                torch.save(facedet.state_dict(),
                           os.path.join(args.save_folder, file))
            iteration += 1

        val(epoch)
        #if iteration == cfg.MAX_STEPS:
        #    break


def val(epoch):
    net.eval()
    loc_loss = 0
    conf_loss = 0
    step = 0
    t1 = time.time()
    for batch_idx, (images, targets) in enumerate(val_loader):
        if args.cuda:
            images = Variable(images.cuda())
            targets = [Variable(ann.cuda(), volatile=True)
                       for ann in targets]
        else:
            images = Variable(images)
            targets = [Variable(ann, volatile=True) for ann in targets]

        out = net(images)
        loss_l, loss_c = criterion(out, targets)
        loss = loss_l + loss_c
        loc_loss += loss_l.data[0]
        conf_loss += loss_c.data[0]
        step += 1

    tloss = (loc_loss + conf_loss) / step
    t2 = time.time()
    print('Timer: %.4f' % (t2 - t1))
    print('test epoch:' + repr(epoch) + ' || Loss:%.4f' % (tloss))

    global min_loss
    if tloss < min_loss:
        print('Saving best state,epoch', epoch)
        file = 'Res50_RFP.pth'
        torch.save(facedet.state_dict(), os.path.join(
            args.save_folder, file))
        min_loss = tloss

    states = {
        'epoch': epoch,
        'weight': facedet.state_dict(),
    }
    file = 'Res50_RFP_checkpoint.pth'
    torch.save(states, os.path.join(
        args.save_folder, file))

def adjust_learning_rate(optimizer, gamma, epoch, step_index, iteration, epoch_size):
    """Sets the learning rate 
    # Adapted from PyTorch Imagenet example:
    # https://github.com/pytorch/examples/blob/master/imagenet/main.py
    """
    if epoch < 6:
        lr = 1e-6 + (args.lr-1e-6) * iteration / (epoch_size * 5) 
    else:
        lr = args.lr * (gamma ** (step_index))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return lr

if __name__ == '__main__':
    train()
