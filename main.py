# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
from pathlib import Path
import argparse
import json
import math
import os
import random
import signal
import subprocess
import sys
import time
from RegSeg_8 import Encoder
import numpy as np

from PIL import Image, ImageOps, ImageFilter
from torch import nn, optim
import torch
import torchvision
import torchvision.transforms as transforms
from torchvision.transforms import InterpolationMode
# from 3d_dataloder import SliceDataset



def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--data', default='./dataset/isic2018',type=Path, metavar='',
                        help='')
    parser.add_argument('--workers', default=2, type=int, metavar='N',
                        help='number of data loader workers')
    parser.add_argument('--epochs', default=100, type=int, metavar='N',
                        help='number of total epochs to run')
    #epochs:100
    parser.add_argument('--batch_size', default=8, type=int, metavar='N',
                        help='mini-batch size')
    parser.add_argument('--learning-rate-weights', default=0.2, type=float, metavar='LR',
                        help='base learning rate for weights')
    parser.add_argument('--learning_rate_biases', default=0.0048, type=float, metavar='LR',
                        help='base learning rate for biases and batch norm parameters')
    parser.add_argument('--weight_decay', default=1e-6, type=float, metavar='W',
                        help='weight decay')
    parser.add_argument('--lambd', default=0.0051, type=float, metavar='L',
                        help='weight on off-diagonal terms')

    parser.add_argument('--projector', default='128-128-128', type=str,
                        metavar='MLP', help='projector MLP')
    parser.add_argument('--print_freq', default=100, type=int, metavar='N',
                        help='print frequency')
    parser.add_argument('--checkpoint_dir', default='./checkpoint/', type=Path,
                        metavar='DIR', help='path to checkpoint directory')
    parser.add_argument('--rank', default=0, type=int, help='node rank for distributed training')
    parser.add_argument('--ngpus_per_node', default=2, type=int)
    parser.add_argument('--gpu', default=None, type=int, help='GPU id to use.')
    config = parser.parse_args()

    return config

def main():
    #args = parser.parse_args()
    args = parse_args() #parser.parse_arge()函数是用来解析命令行参数的
    args.ngpus_per_node = torch.cuda.device_count()#获取当前节点gpu可用数量

    if 'SLURM_JOB_ID' in os.environ:    #？？？？？？？？？？？？？？？？？？？？？？？？？？？？？？？？？？？？？？？/
        # single-node and multi-node distributed training on SLURM cluster
        # requeue job on SLURM preemption
        signal.signal(signal.SIGUSR1, handle_sigusr1)
        signal.signal(signal.SIGTERM, handle_sigterm)
        # find a common host name on all nodes
        # assume scontrol returns hosts in the same order on all nodes
        cmd = 'scontrol show hostnames ' + os.getenv('SLURM_JOB_NODELIST')
        stdout = subprocess.check_output(cmd.split())
        host_name = stdout.decode().splitlines()[0]
        args.rank = int(os.getenv('SLURM_NODEID')) * args.ngpus_per_node
        args.world_size = int(os.getenv('SLURM_NNODES')) * args.ngpus_per_node
        args.dist_url = f'tcp://{host_name}:58472'
    else:
        # single-node distributed training
        args.rank = 0
        args.dist_url = 'tcp://localhost:58472'
        args.world_size = args.ngpus_per_node
    torch.multiprocessing.spawn(main_worker, (args,), args.ngpus_per_node)


def main_worker(gpu, args):
    args.rank += gpu
    # print(gpu)
    #使用pytorch的分布式训练工具初始化进程组
    #使用的后端是nccl  backend参数指定了进程中的进程数量  word_size参数指定了进程组中的进程数量 rank`参数指定了当前进程在进程组中的排名
    torch.distributed.init_process_group(
        backend='nccl', init_method=args.dist_url,
        world_size=args.world_size, rank=args.rank)

    if args.rank == 0:
        args.checkpoint_dir.mkdir(parents=True, exist_ok=True)#创建一个文件用于储存检查点

        stats_file = open(args.checkpoint_dir / 'stats.txt', 'a', buffering=1)#打开一个文本文件用于记录统计信息  ‘a’表示以追加模式打开文件  buffering=1表示启用缓存区
        print(' '.join(sys.argv))
        print(' '.join(sys.argv), file=stats_file)

    torch.cuda.set_device(gpu)
    torch.backends.cudnn.benchmark = True

    model = BarlowTwins(args).cuda(gpu)
    model = nn.SyncBatchNorm.convert_sync_batchnorm(model)#将模型中的普通BatchNorm层转化为同步批量归一化（SyncBatchNorm）层，以便在分布训练中使用
    #同步批量归一化层可以在多个GPU之间同步均值和方差，从而加速模型的收敛速度。这个函数会遍历模型中的所有BatchNorm层，并将其转换为SyncBatchNorm层。
    param_weights = []
    param_biases = []
    for param in model.parameters():#将所有维度为1的视为偏置  不为1的视为权重
        if param.ndim == 1:
            param_biases.append(param)
        else:
            param_weights.append(param)
    parameters = [{'params': param_weights}, {'params': param_biases}]

    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[gpu], find_unused_parameters=True)
    #它将模型包装在`DistributedDataParallel`中，以便在多个GPU上进行并行训练。其中`device_ids`参数指定了使用哪些GPU进行训练
    optimizer = LARS(parameters, lr=0, weight_decay=args.weight_decay,
                     weight_decay_filter=True,
                     lars_adaptation_filter=True)

    # automatically resume from checkpoint if it exists
    #如果检查点文件存在，则会加载之前训练的模型参数、优化器状态和训练轮数等信息，从上次训练结束的地方继续训练
    #这样做的好处是，如果训练过程中出现意外情况导致程序中断，可以通过加载检查点文件恢复训练，避免浪费之前的训练成果。
    if (args.checkpoint_dir / 'checkpoint.pth').is_file():
        ckpt = torch.load(args.checkpoint_dir / 'checkpoint.pth',
                          map_location='cpu')
        start_epoch = ckpt['epoch']
        model.load_state_dict(ckpt['model'])
        optimizer.load_state_dict(ckpt['optimizer'])
    else:
        start_epoch = 0

    dataset = torchvision.datasets.ImageFolder(args.data / "pretrain", Transform())

    # dataset = torchvision.datasets.ImageFolder(args.data, Transform())
    # dataset = torchvision.datasets.DatasetFolder(root=args.data / "pretrain", transform=Transform())
    sampler = torch.utils.data.distributed.DistributedSampler(dataset)
    assert args.batch_size % args.world_size == 0 #确保每个GPU上的批次大小是整数。
    per_device_batch_size = args.batch_size // args.world_size
    loader = torch.utils.data.DataLoader(
        dataset, batch_size=per_device_batch_size, num_workers=args.workers,
        pin_memory=True, sampler=sampler, drop_last = True)

########################
    start_time = time.time()
    scaler = torch.cuda.amp.GradScaler()
    # torch.cuda.reset_peak_memory_stats()
    #由于半精度浮点数的精度较低，可能会导致数值不稳定的问题，因此需要使用GradScaler来缩放梯度，以保证训练的稳定性和精度。
    for epoch in range(start_epoch, args.epochs):
        torch.cuda.empty_cache()
        sampler.set_epoch(epoch)
        for step, ((y1, y2), _) in enumerate(loader, start=epoch * len(loader)): #len(loader)是每个批次中数据的数量
            y1 = y1.cuda(gpu, non_blocking=True)
            y2 = y2.cuda(gpu, non_blocking=True)
            torch.cuda.empty_cache()
            adjust_learning_rate(args, optimizer, loader, step)
            optimizer.zero_grad()
            with torch.cuda.amp.autocast():#使用了PyTorch的自动混合精度功能   这个上下文管理器会自动将输入的浮点数转换为半精度浮点数（float16），以减少计算所需的内存和计算时间
                loss = model.forward(y1, y2)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            scaler = torch.cuda.amp.GradScaler()
            optimizer.zero_grad()
            torch.cuda.empty_cache()
            if step % args.print_freq == 0:
                if args.rank == 0:
                    stats = dict(epoch=epoch, step=step,
                                 lr_weights=optimizer.param_groups[0]['lr'],
                                 lr_biases=optimizer.param_groups[1]['lr'],
                                 loss=loss.item(),
                                 time=int(time.time() - start_time))
                    print(json.dumps(stats))
                    print(json.dumps(stats), file=stats_file)
        # param_size = 0
        # for name, param in model.named_parameters():
        #     print(param.size())
        # print(param_size,'++++++++++++++')
        if args.rank == 0:
            # save checkpoint
            state = dict(epoch=epoch + 1, model=model.state_dict(),
                         optimizer=optimizer.state_dict())
            torch.save(state, args.checkpoint_dir / 'checkpoint.pth')

    if args.rank == 0:
        # save final model
        #torch.save(model.module.backbone.state_dict(),
          #         args.checkpoint_dir / 'resnet50.pth')
        torch.save(model.module.backbone.state_dict(),
    args.checkpoint_dir / 'RegSeg_8_isic18_encoder.pth')

# ########################################
#
#
#
# class WarmUpCosine(torch.optim.lr_scheduler.LambdaLR):
#     def __init__(self, optimizer, learning_rate_base, total_steps, warmup_learning_rate, warmup_steps):
#         self.learning_rate_base = learning_rate_base
#         self.total_steps = total_steps
#         self.warmup_learning_rate = warmup_learning_rate
#         self.warmup_steps = warmup_steps
#         self.pi = torch.tensor(np.pi)
#
#         def lr_lambda(step):
#             if self.total_steps < self.warmup_steps:
#                 raise ValueError("Total_steps must be larger or equal to warmup_steps.")
#
#             learning_rate = (
#                     0.5
#                     * self.learning_rate_base
#                     * (
#                             1
#                             + torch.cos(
#                         self.pi * (step - self.warmup_steps) / float(self.total_steps - self.warmup_steps)
#                                       )
#                       )
#             )
#             if self.warmup_steps > 0:
#                 if self.learning_rate_base < self.warmup_learning_rate:
#                     raise ValueError(
#                         "Learning_rate_base must be larger or equal to warmup_learning_rate."
#                     )
#                 slope = (
#                                 self.learning_rate_base - self.warmup_learning_rate
#                         ) / self.warmup_steps
#                 warmup_rate = slope * step + self.warmup_learning_rate
#                 learning_rate = torch.where(
#                     step < self.warmup_steps, warmup_rate, learning_rate
#                 )
#
#             return torch.where(
#                 step > self.total_steps, torch.tensor(0.0), learning_rate
#             )
#
#         super(WarmUpCosine, self).__init__(optimizer, lr_lambda)
#
# lr_decayed_fn = WarmUpCosine(
#     learning_rate_base=1e-3,
#     total_steps=EPOCHS * STEPS_PER_EPOCH,
#     warmup_learning_rate=0.0,
#     warmup_steps=WARMUP_STEPS
# )
# optimizer = tf.keras.optimizers.SGD(learning_rate=lr_decayed_fn, momentum=0.9)
# ########################################


def adjust_learning_rate(args, optimizer, loader, step):
    max_steps = args.epochs * len(loader)
    warmup_steps = 10 * len(loader)
    base_lr = args.batch_size / 256
    if step < warmup_steps:
        lr = base_lr * step / warmup_steps
    else:
        step -= warmup_steps
        max_steps -= warmup_steps
        q = 0.5 * (1 + math.cos(math.pi * step / max_steps))
        end_lr = base_lr * 0.001
        lr = base_lr * q + end_lr * (1 - q)
    optimizer.param_groups[0]['lr'] = lr * args.learning_rate_weights
    optimizer.param_groups[1]['lr'] = lr * args.learning_rate_biases


def handle_sigusr1(signum, frame):
    os.system(f'scontrol requeue {os.getenv("SLURM_JOB_ID")}')
    exit()


def handle_sigterm(signum, frame):
    pass


def off_diagonal(x):#计算的是非对角线元素
    # return a flattened view of the off-diagonal elements of a square matrix
    n, m = x.shape
    assert n == m
    return x.flatten()[:-1].view(n - 1, n + 1)[:, 1:].flatten()


class BarlowTwins(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
       # self.backbone = torc,hvision.models.resnet50(zero_init_residual=True)
        self.backbone = Encoder()

        self.backbone.fc = nn.Identity()
        #模型的最后一层fcn被替换成identity
        # 这意味着模型不会对特征进行任何降维或变换，而是直接输出ResNet50提取的特征。

        # projector 构建一个mlp的projector
        sizes = [256] + list(map(int, args.projector.split('-')))  #别忘记将最后一层的神经元个数设置为128
        #size表示每一层神经元的个数 如第一层神经元的个数为128


        layers = []
        for i in range(len(sizes) - 2):#两个
            # print('2----', sizes, sizes[-2], sizes[-1])
            layers.append(nn.Linear(sizes[i], sizes[i + 1], bias=False))
            #nn.linear(in_features,out_features,bias)  in_features是输入特征图的数量
            layers.append(nn.BatchNorm1d(sizes[i + 1]))
            layers.append(nn.ReLU(inplace=True))
        layers.append(nn.Linear(sizes[-2], sizes[-1], bias=False))#最后一层
        self.projector = nn.Sequential(*layers)
        #`nn.Sequential(*layers)` 是 PyTorch 中的一个模型容器，它可以将多个层按照顺序组合成一个神经网络模型

        # normalization layer for the representations z1 and z2
        self.bn = nn.BatchNorm1d(sizes[-1], affine=False)

    def forward(self, y1, y2):
        #print("y1------shape",type(y1))
        z1 = self.backbone(y1)
        #print("z1----------shape", type(z1))
        z2 = self.backbone(y2)
        
        # print("z1********", z1.shape) torch.Size([2, 256, 8, 8])  b,c,w,h 代码中batch_size设置的是8  但是这里并行计算时有4个显卡  所以每个gpu上放2张 所以这里的b是2
        # print("z2********", z2.shape)
        # print('q---', z1.shape)


        global_avg_pool_layer = nn.AdaptiveAvgPool2d(output_size=(1, 1))#全局平均池化  v2
        
        z1 = global_avg_pool_layer(z1)
        # print('1111111',z1.shape)
        z1 = z1.squeeze()
        # print('2222222',z1.shape)
        z2 = global_avg_pool_layer(z2)
        # print('3333333',z2.shape)
        z2 = z2.squeeze()
        # print('4444444',z2.shape)




        # print("z1----", z1.shape)
        # print("z2-----", z2.shape)
        # exit(0)
        # z1 = z1.permute(0, 2, 3, 1)
        # print('q---', z1.shape) torch.Size([2, 8, 8, 256])
        # z1 = z1.reshape(-1, 256)#z2.reshape(-1, 256) 将 z2 重塑为一个形状为 (batch_size * num_channels * height * width, 256)
        # print('q---', z1.shape)

        # z2 = z2.permute(0, 2, 3, 1)
        # z2 = z2.reshape(-1, 256)
        # print("z1=======",z1.shape)torch.Size([128, 256])
        # print("z2=======", z2.shape)
     
        z1 = self.projector(z1)
        z2 = self.projector(z2)

        # print("z1-----", z1.shape)
        # print("z2-----", z2.shape)torch.Size([128, 128])
       
        # empirical cross-correlation matrix
        c = self.bn(z1).T @ self.bn(z2)

        # sum the cross-correlation matrix between all gpus
        c.div_(self.args.batch_size)
        torch.distributed.all_reduce(c)

        on_diag = torch.diagonal(c).add_(-1).pow_(2).sum()#对角线元素-1的平方然后求和
        off_diag = off_diagonal(c).pow_(2).sum()
        loss = on_diag + self.args.lambd * off_diag#loss公式
        return loss


class LARS(optim.Optimizer):
    def __init__(self, params, lr, weight_decay=0, momentum=0.9, eta=0.001,
                 weight_decay_filter=False, lars_adaptation_filter=False):
        defaults = dict(lr=lr, weight_decay=weight_decay, momentum=momentum,
                        eta=eta, weight_decay_filter=weight_decay_filter,
                        lars_adaptation_filter=lars_adaptation_filter)
        super().__init__(params, defaults)

    def exclude_bias_and_norm(self, p):
        return p.ndim == 1

    @torch.no_grad()
    def step(self):
        for g in self.param_groups:
            for p in g['params']:
                dp = p.grad

                if dp is None:
                    continue

                if not g['weight_decay_filter'] or not self.exclude_bias_and_norm(p):
                    dp = dp.add(p, alpha=g['weight_decay'])

                if not g['lars_adaptation_filter'] or not self.exclude_bias_and_norm(p):
                    param_norm = torch.norm(p)
                    update_norm = torch.norm(dp)
                    one = torch.ones_like(param_norm)
                    q = torch.where(param_norm > 0.,
                                    torch.where(update_norm > 0,
                                                (g['eta'] * param_norm / update_norm), one), one)
                    dp = dp.mul(q)

                param_state = self.state[p]
                if 'mu' not in param_state:
                    param_state['mu'] = torch.zeros_like(p)
                mu = param_state['mu']
                mu.mul_(g['momentum']).add_(dp)

                p.add_(mu, alpha=-g['lr'])


class GaussianBlur(object):
    def __init__(self, p):
        self.p = p

    def __call__(self, img):
        if random.random() < self.p:
            sigma = random.random() * 1.9 + 0.1
            return img.filter(ImageFilter.GaussianBlur(sigma))
        else:
            return img


class Solarization(object):
    def __init__(self, p):
        self.p = p

    def __call__(self, img):
        if random.random() < self.p:
            return ImageOps.solarize(img)
        else:
            return img


class Transform:#这里的数据增强方式需要改动下   可能看后续结果再改动也ok
    def __init__(self):
        self.transform = transforms.Compose([
            transforms.RandomResizedCrop(256, interpolation=InterpolationMode.BICUBIC),#将图像随机裁剪成指定大小并，进行双三次插值。
            #这里源代码里面设置的是256
            transforms.RandomHorizontalFlip(p=0.5),#以0.5的概率对图像进行水平翻转。
            transforms.RandomApply(
                [transforms.ColorJitter(brightness=0.4, contrast=0.4,
                                        saturation=0.2, hue=0.1)],
                p=0.8
            ),#以0.8的概率对图像进行颜色抖动，包括亮度、对比度、饱和度和色调。

            transforms.RandomGrayscale(p=0.2),#以一定的概率将图像转换为灰度图像，可以增加模型对于颜色变化的鲁棒性。
            GaussianBlur(p=1.0),#以一定的概率对图像进行高斯模糊，
            Solarization(p=0.0),#以一定的概率对图像进行反转
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])
        self.transform_prime = transforms.Compose([
            transforms.RandomResizedCrop(256, interpolation=InterpolationMode.BICUBIC),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomApply(
                [transforms.ColorJitter(brightness=0.4, contrast=0.4,
                                        saturation=0.2, hue=0.1)],
                p=0.8
            ),
            transforms.RandomGrayscale(p=0.2),
            GaussianBlur(p=0.1),
            Solarization(p=0.2),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])

    def __call__(self, x):
        y1 = self.transform(x)
        y2 = self.transform_prime(x)
        return y1, y2


if __name__ == '__main__':
    main()
