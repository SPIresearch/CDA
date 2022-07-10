import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.optim.lr_scheduler as lr_scheduler
import numpy as np
from .basemodel import FeatureNet, PredNet
from cross_entropy_loss import CrossEntropyLoss
from avgmeter import AverageMeter
from accuracy import accuracy
import time
import datetime
from itertools import cycle
# ======================================================================================================================

class Baseline(nn.Module):
    def __init__(self, opt):
        super(Baseline, self).__init__()
        self.opt = opt
        self.device = opt.device
        self.use_gpu = True if self.device == 'cuda' else False
        self.train_log = opt.outf + '/train.log'
        self.model_path = opt.outf + '/model.pth'
        self.max_epoch = opt.num_epoch 
        self.net = FeatureNet(opt)
        self.cls = PredNet(opt)
        self.optimizer = torch.optim.Adam(self.net.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999),
                                            weight_decay=opt.weight_decay)

        self.lr_scheduler = lr_scheduler.ExponentialLR(optimizer=self.optimizer, gamma=0.5 ** (1 / 100))
        self.criterion = CrossEntropyLoss(
            num_classes= opt.class_num
        )

    def learn(self, epoch, source1_dataloader, source2_dataloader):
        self.epoch = epoch
        self.train()
        losses = AverageMeter()
        accs = AverageMeter()
        batch_time = AverageMeter()
        data_time = AverageMeter()
        num_batches = len(source1_dataloader)
        end = time.time()
        source2_dataloader = iter(cycle(source2_dataloader))
        for index, data_s1 in enumerate(source1_dataloader):
            data_time.update(time.time()-end)
            self.optimizer.zero_grad()
            x1, y1, idx = data_s1
            x2, y2, idx = next(source2_dataloader)
            x = torch.cat([x1, x2],0)
            y = torch.cat([y1, y2],0)
            x, y = x.to(self.device), y.to(self.device)
            feat = self.net(x)
            y_h = self.cls(feat)
            loss =self.criterion(y_h, y)
            
            loss.backward()
            self.optimizer.step()
            batch_time.update(time.time()-end)
            losses.update(loss.item(), y.size(0))
            accs.update(accuracy(y_h, y)[0].item())

            if (index+1)%10 == 0:
                eta_seconds = batch_time.avg * (num_batches-(index+1) + (self.max_epoch -(epoch+1))*num_batches)
                eta_str = str(datetime.timedelta(seconds=int(eta_seconds)))
                print('Epoch: [{0}/{1}][{2}/{3}]\t'
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                      'Loss {cnn_loss.val:.4f} ({cnn_loss.avg:.4f})\t'
                      'Acc {acc.val:.2f} ({acc.avg:.2f})\t'
                      'Lr {lr:.6f}\t'
                      'eta {eta}'.format(
                       epoch+1, self.max_epoch , index+1, num_batches,
                       batch_time=batch_time,
                       data_time=data_time,
                       cnn_loss = losses,
                       acc=accs,
                       lr=self.optimizer.param_groups[0]['lr']   ,
                       eta=eta_str
                      )
                )
        self.lr_scheduler.step()

    def save(self):
        torch.save(self.state_dict(), self.model_path)

    def load(self):
        print('===> Loading model from {}'.format(self.model_path))
        try:
            self.load_state_dict(torch.load(self.model_path))
            print('<=== Success!')
        except:
            print('<==== Failed!')

    @torch.no_grad()
    def test(self, epoch, dataloader):
        self.eval()
        y_hs = []
        y_s = []
        for data in dataloader:
            x, y, idx = data
            x , y = x.to(self.device),y.to(self.device)
            feat = self.net(x)
            y_h = self.cls(feat)
            y_hs.append(y_h)
            y_s.append(y)

        y_hs = torch.cat(y_hs, 0)
        y_s = torch.cat(y_s, 0)
        acc = accuracy(y_hs,y_s)
        print(acc)

    def init_weight(self, net=None):
        if net is None:
            net = self
        for m in net.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, mean=0, std=0.01)
#                 nn.init.normal_(m.weight, mean=0, std=0.1)
#                 nn.init.xavier_normal_(m.weight, gain=10)
                nn.init.constant_(m.bias, val=0)

def to_np(x):
    return x.detach().cpu().numpy()



class Baseline1s(nn.Module):
    def __init__(self, opt):
        super(Baseline1s, self).__init__()
        self.opt = opt
        self.device = opt.device
        self.use_gpu = True if self.device == 'cuda' else False
        self.train_log = opt.outf + '/train.log'
        self.model_path = opt.outf + '/model.pth'
        self.max_epoch = opt.num_epoch 
        self.net = FeatureNet(opt)
        self.cls = PredNet(opt)
        self.optimizer = torch.optim.Adam(self.net.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999),
                                            weight_decay=opt.weight_decay)

        self.lr_scheduler = lr_scheduler.ExponentialLR(optimizer=self.optimizer, gamma=0.5 ** (1 / 100))
        self.criterion = CrossEntropyLoss(
            num_classes= opt.class_num
        )

    def learn(self, epoch, source_dataloader):
        self.epoch = epoch
        self.train()
        losses = AverageMeter()
        accs = AverageMeter()
        batch_time = AverageMeter()
        data_time = AverageMeter()
        num_batches = len(source_dataloader)
        end = time.time()
        for index, data_s1 in enumerate(source_dataloader):
            data_time.update(time.time()-end)
            self.optimizer.zero_grad()
            x, y, idx = data_s1
            x, y = x.to(self.device), y.to(self.device)
            feat = self.net(x)
            y_h = self.cls(feat)
            loss =self.criterion(y_h, y)
            
            loss.backward()
            self.optimizer.step()
            batch_time.update(time.time()-end)
            losses.update(loss.item(), y.size(0))
            accs.update(accuracy(y_h, y)[0].item())

            if (index+1)%50 == 0:
                eta_seconds = batch_time.avg * (num_batches-(index+1) + (self.max_epoch -(epoch+1))*num_batches)
                eta_str = str(datetime.timedelta(seconds=int(eta_seconds)))
                print('Epoch: [{0}/{1}][{2}/{3}]\t'
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                      'Loss {cnn_loss.val:.4f} ({cnn_loss.avg:.4f})\t'
                      'Acc {acc.val:.2f} ({acc.avg:.2f})\t'
                      'Lr {lr:.6f}\t'
                      'eta {eta}'.format(
                       epoch+1, self.max_epoch , index+1, num_batches,
                       batch_time=batch_time,
                       data_time=data_time,
                       cnn_loss = losses,
                       acc=accs,
                       lr=self.optimizer.param_groups[0]['lr']   ,
                       eta=eta_str
                      )
                )
        self.lr_scheduler.step()

    def save(self):
        torch.save(self.state_dict(), self.model_path)

    def load(self):
        print('===> Loading model from {}'.format(self.model_path))
        try:
            self.load_state_dict(torch.load(self.model_path))
            print('<=== Success!')
        except:
            print('<==== Failed!')

    @torch.no_grad()
    def test(self, epoch, dataloader):
        self.eval()
        y_hs = []
        y_s = []
        feats = []
        for data in dataloader:
            x, y, idx = data
            x , y = x.to(self.device),y.to(self.device)
            feat = self.net(x)
            y_h = self.cls(feat)
            y_hs.append(y_h)
            y_s.append(y)
            feats.append(feat.squeeze().cpu().numpy())

        y_hs = torch.cat(y_hs, 0)
        y_s = torch.cat(y_s, 0)
        feats = np.concatenate(feats, axis=0)
        acc = accuracy(y_hs,y_s)
        return acc[0].item(), feats

    def init_weight(self, net=None):
        if net is None:
            net = self
        for m in net.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, mean=0, std=0.01)
#                 nn.init.normal_(m.weight, mean=0, std=0.1)
#                 nn.init.xavier_normal_(m.weight, gain=10)
                nn.init.constant_(m.bias, val=0)

def to_np(x):
    return x.detach().cpu().numpy()