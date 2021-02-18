import argparse
from pathlib import Path
import shutil
import os
import os.path as osp
import random
import numpy as np
from tqdm import tqdm

import torch
import torch.cuda.amp as amp
from torch.cuda.amp.grad_scaler import GradScaler
from torch.utils.data import DataLoader
import torch.optim as optim
from torch.optim.lr_scheduler import MultiStepLR, OneCycleLR

from transformations import get_train_transforms
from utils.utils import load_yaml, ProbsAverageMeter
from utils.log_utils import Writer

from factory import get_fpn_net

from losses.AnchorFreeloss import AnchorFreeLoss
from datasets.CAVIARDataset import CAVIARDataset as Dataset

from utils.LossMetric import LossMetric

class Trainer:
    def __init__(self,cfg):
        self.cfg = cfg
        self.paths = cfg['paths']
        self.net_params = cfg['net']
        self.train_params = cfg['train']
        self.trans_params = cfg['train']['transforms']

        self.checkpoints = self.paths['checkpoints']
        Path(self.checkpoints).mkdir(parents=True, exist_ok=True)
        shutil.copyfile('config.yaml', f'{self.checkpoints}/config.yaml')

        self.update_interval = self.paths['update_interval']
        
        # amp training
        self.use_amp = self.train_params['mixed_precision']
        self.scaler = GradScaler() if self.use_amp else None

        # data setup
        # TODO - add datasets
        self.train_dataset = Dataset(self.paths['data'], augment=get_train_transforms())
        print(f'Train dataset: {len(self.train_dataset)} samples')
        self.val_dataset = Dataset(self.paths['data'])
        print(f'Val dataset: {len(self.val_dataset)} samples')

        self.criterion = AnchorFreeLoss(self.train_params)

        self.writer = Writer(self.paths['log_dir'])
        print('Tensorboard logs are saved to: {}'.format(self.paths['log_dir']))

        self.sched_type = self.train_params['scheduler']
        self.scheduler = None
        self.optimizer = None

    def train(self):
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = True

        batch_size = self.train_params['batch_size']
        num_workers = self.train_params['num_workers']
        pin_memory = self.train_params['pin_memory']
        print('Batch-size = {}'.format(batch_size))

        train_loader = DataLoader(
            self.train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=pin_memory,
            drop_last=True
        )
        val_loader = DataLoader(
            self.val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=pin_memory,
            drop_last=False
        )

        # net setup
        print('Preparing net: ')
        net = get_fpn_net(self.net_params)
        # TODO - pretrained / restore
        net.cuda()

        # train setup
        lr = self.train_params['lr']
        epochs = self.train_params['epochs']
        weight_decay = self.train_params['weight_decay']

        self.optimizer = optim.Adam(net.parameters(), lr=lr, weight_decay=weight_decay, eps=1e-4)

        first_epoch = 0
        # scheduler 
        if self.sched_type == 'ocp':
            last_epoch = -1 if first_epoch == 0 else first_epoch*len(train_loader)
            self.scheduler = OneCycleLR(
                self.optimizer,
                max_lr=lr,
                epochs=epochs,
                last_epoch=last_epoch,
                steps_per_epoch=len(train_loader),
                pct_start=self.train_params['ocp_params']['max_lr_pct']
            )
        elif self.sched_type == 'multi_step':
            self.scheduler = MultiStepLR(
                self.optimizer,
                milestones=self.train_params['multi_params']['milestones'],
                gamma=self.train_params['multi_params']['gamma'],
                last_epoch=first_epoch
            )
        
        #start training

        net.train()
        val_rate = self.train_params['val_rate']
        test_rate = self.train_params['test_rate']

        for epoch in range(first_epoch, epochs):
            self.train_epoch(net, train_loader, epoch)

            if self.sched_type != 'ocp':
                self.writer.log_lr(epoch, self.scheduler.get_last_lr()[0])
                self.scheduler.step()

            if (epoch + 1) % val_rate == 0 or epoch == epochs -1:
                # TODO - eval and test
                pass

    def train_epoch(self, net, loader, epoch):
        net.train()
        # TODO = LossMetric class
        loss_metric = LossMetric(self.cfg)
        probs = ProbsAverageMeter()
        
        for mini_batch_i, read_mini_batch in tqdm(enumerate(loader), desc=f'Epoch {epoch}:', ascii=True, total=len(loader)):
            data, labels = read_mini_batch
            data.cuda()
            labels.cuda()

            with amp.autocast():
                out = net(data)
                loss_dict, hm_probs = self.criterion(out, labels)
                loss = loss_metric.calculate_loss(loss_dict)
            self.optimizer.zero_grad()
            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()

            probs.update(hm_probs)

            if self.sched_type == 'ocp':
                self.writer.log_lr(epoch * len(loader) + mini_batch_i, self.scheduler.get_last_lr()[0])
                self.scheduler.step()

            loss_metric.add_sample(loss_dict)
            
            if mini_batch_i % self.update_interval == 0:
                self.writer.log_training(epoch * len(loader) + mini_batch_i, loss_metric)
        self.writer.log_probs(epoch, probs.get_average())



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-gpu", type=int, default=0, help='gpu to run on.')

    args = parser.parse_args()

    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)
    os.system('echo $CUDA_VISIBLE_DEVICES')

    cfg_path = 'config.yaml'
    cfg = load_yaml(cfg_path)

    trainer = Trainer(cfg)
    trainer.train()
