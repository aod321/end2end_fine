import os
import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST
from torchvision import transforms
import pytorch_lightning as pl
from dataloaders import HelenLoader
from helper_funcs import F1Score, calc_centroid
from model import Stage1Model, Stage2Model, SelectNet, ReverseNet
import argparse
from torchvision import transforms
from preprocess import ToPILImage, ToTensor, Resize
from collections import OrderedDict


def args(func):
    def wrapper(*args, **kw):
        args[0].parser.add_argument("--batch_size", default=16, type=int, help="Batch size to use during training.")
        args[0].parser.add_argument("--workers", default=4, type=int, help="Dataloader Workers.")
        args[0].parser.add_argument("--cuda", default=0, type=int, help="Choose GPU with cuda number")
        func(*args, **kw)
        args_a = args[0].parser.parse_args()      
        return args_a
    return wrapper


class TrainModule(pl.LightningModule):

    def __init__(self):
        super(TrainModule, self).__init__()
        self.parser = argparse.ArgumentParser()

        self.args = self.set_args()

        self.device = torch.device("cuda:%d" % self.args.cuda if torch.cuda.is_available() else "cpu")

        self.dataset_root_dir = "/home/yinzi/data1/datas"

        self.model1 = Stage1Model()

        self.model2 = Stage2Model()

        self.select_net = SelectNet()            # a fixed affine_model used to crop parts

        self.reverse_net = ReverseNet()          # a fixed affine_model used to recover parts to a whole image

        self.loss_func = nn.CrossEntropyLoss()

        self.metric = F1Score(self.device)

    def forward(self, x):
        return self.model1(x)

    def training_step(self, batch, batch_nb, optim_idx):
        image = batch['image'].to(self.device)
        label = batch['labels'].to(self.device)
        orig = batch['orig'].to(self.device)
        orig_label = batch['orig_label'].to(self.device)
        
        stage1 = self.forward(image)

        loss1 = self.loss_func(stage1, label.long())

        cens = calc_centroid(F.interpolate(stage1, (512, 512), mode='nearest'))
        stage2_parts, stage2_labels = self.select_net(img=orig, label=orig_label, points=cens)
        stage2_predicts = self.model2(stage2_parts)
        loss2 = []
        for i in range(6):
            loss2.append(self.loss_func(stage2_predicts[i], stage2_labels[:, i].long()))
        loss = loss1 + sum(loss2)
        logger_logs = {'training_loss': loss}
        output = {
            'loss': loss,  # required
            'progress_bar': {'training_loss': loss},  # optional (MUST ALL BE TENSORS)
            'log': logger_logs
        }
        return output

    def validation_step(self, batch, batch_nb):
        # OPTIONAL
        image = batch['image'].to(self.device)
        label = batch['labels'].to(self.device)
        orig = batch['orig'].to(self.device)
        orig_label = batch['orig_label'].to(self.device)

        stage1 = self.forward(image)
        cens = calc_centroid(F.interpolate(stage1, (512, 512), mode='nearest'))

        stage2_parts, stage2_labels = self.select_net(img=orig, label=orig_label, points=cens)
        stage2_predicts = self.model2(stage2_parts)

        final_predicts = self.reverse_net(stage2_predicts, self.select_net.theta)  # Shape(N, 6, 512, 512)
        loss = self.loss_func(final_predicts, orig_label)
        # f1_score = self.metric(final_predicts, orig_label)
        # f1_list = f1_score[0]
        output = OrderedDict({
            'val_loss': loss
        })
        # f1_score = self.metric(final_predicts, orig_label)
        # f1_list = f1_score[0]
        return output

    def validation_end(self, outputs):
        # OPTIONAL
        metric_value = 0
        tqdm_dict = {}
        for output in outputs:
            metric_value += output['val_loss']
        tqdm_dict['val_loss'] = metric_value / len(outputs)
        # F1, F1_overall = self.metric.output_f1_score()
        result = {'progress_bar': tqdm_dict, 'log': tqdm_dict, 'val_loss': tqdm_dict["val_loss"]}
        return result

    def test_step(self, batch, batch_nb):
        # OPTIONAL
        image = batch['image'].to(self.device)
        orig = batch['orig_image'].to(self.device)
        orig_label = batch['orig_label'].to(self.device)
        stage1 = self.forward(image)
        cens = calc_centroid(F.interpolate(stage1, (512, 512), mode='nearest'))
        stage2_parts, stage2_labels = self.select_net(img=orig, label=orig_label, points=cens)
        stage2_predicts = self.model2d(stage2_parts)

        final_predicts = self.reverse_netd(stage2_predicts, self.select_net.theta)
        f1_score = self.metric(final_predicts, orig_label)
        f1_list = f1_score[0]
        return f1_list

    def test_end(self, outputs):
        # OPTIONAL
        F1, F1_overall = self.metric.output_f1_score()
        F1['overall'] = F1_overall
        return F1, F1_overall

    def configure_optimizers(self):
        pass
        # # REQUIRED
        # # can return multiple optimizers and learning_rate schedulers
        # optim_1 = torch.optim.Adam(self.model1.parameters(), lr=self.args.lr1)
        # optim_2 = torch.optim.Adam(self.model2.parameters(), lr=self.args.lr2)
        # return [optim_1, optim_2]

    @pl.data_loader
    def train_dataloader(self):
        # REQUIRED
        transformers = transforms.Compose([
            ToPILImage(),
            Resize((80, 80)),
            ToTensor()
        ]),
        loader = HelenLoader(self.dataset_root_dir, transforms=transformers,
                             batch_size=self.args.batch_size, workers=self.args.workers, mode='train')
        return loader.get_dataloader()

    @pl.data_loader
    def val_dataloader(self):
        # OPTIONAL
        transformers = transforms.Compose([
            ToPILImage(),
            Resize((80, 80)),
            ToTensor()
        ]),
        loader = HelenLoader(self.dataset_root_dir, transforms=transformers,
                             batch_size=self.args.batch_size, workers=self.args.workers, mode='val')
        return loader.get_dataloader()

    @pl.data_loader
    def test_dataloader(self):
        # OPTIONAL
        transformers = transforms.Compose([
            ToPILImage(),
            Resize((80, 80)),
            ToTensor()
        ]),
        loader = HelenLoader(self.dataset_root_dir, transforms=transformers,
                             batch_size=self.args.batch_size, workers=self.args.workers, mode='test')
        return loader.get_dataloader()

    @args
    def set_args(self):
        self.parser.add_argument("--lr1", default=0, type=float, help="Stage1 Learning rate for optimizer")
        self.parser.add_argument("--lr2", default=0, type=float, help="Stage2 Learning rate for optimizer")
