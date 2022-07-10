import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim.lr_scheduler as lr_scheduler
import numpy as np
from .basemodel import FeatureNet,  nnUnsqueeze, nnSqueeze, to_np, to_tensor, PredNet,PredNet_Norm
from .mdd import ClfMDD
from .grl import WarmStartGradientReverseLayer
from cross_entropy_loss import CrossEntropyLoss
from itertools import cycle
from accuracy import accuracy
from torch.utils.tensorboard import SummaryWriter
from torchvision.transforms.functional import rotate


class CDA(nn.Module):
    def __init__(self, opt):
        super(CDA, self).__init__()
        self.opt = opt
        self.device = opt.device
        self.writer = SummaryWriter()
        self.train_log = opt.outf + '/train.log'
        self.model_path = opt.outf + '/model.pth'
        self.best_acc_tgt = 0
        self.alpha = 0.01

        self.netE = FeatureNet(opt)
        self.netF1 = PredNet(opt)
        self.netF2 = PredNet(opt)
        self.netF_adv1 = PredNet(opt) #
        self.netF_adv2 = PredNet(opt) #

        self.mdd1 = ClfMDD(opt.class_num)
        self.mdd2 = ClfMDD(opt.class_num)
        self.mdd3 = ClfMDD(opt.class_num)
        self.criterion = CrossEntropyLoss(
        num_classes= opt.class_num      
        )
        self.grl_layer1 = WarmStartGradientReverseLayer(alpha=1.0, lo=0.0, hi=0.1, max_iters=1000,auto_step=False)
        self.grl_layer2 = WarmStartGradientReverseLayer(alpha=1.0, lo=0.0, hi=0.1, max_iters=1000,auto_step=False)

        parameters = list(self.netE.parameters()) + list(self.netF1.parameters()) + list(self.netF2.parameters()) + list(self.netF_adv1.parameters()) + list(self.netF_adv2.parameters()) +list(self.grl_layer1.parameters())+list(self.grl_layer2.parameters())
        # self.optimizer = torch.optim.Adam(parameters, lr=opt.lr, betas=(opt.beta1, 0.999),
        #                                     weight_decay=opt.weight_decay)

        self.optimizer = torch.optim.SGD(parameters, lr=0.01, weight_decay=0.0005 ,momentum=0.9)
        # self.lr_scheduler = lr_scheduler.StepLR(optimizer=self.optimizer, step_size=50)
        self.lr_scheduler = lr_scheduler.ExponentialLR(optimizer=self.optimizer, gamma=0.5 ** (1 / 50))
        self.loss_S2 = torch.tensor(0.)
        self.loss_S1 = torch.tensor(0.)
        self.loss_names = ['S1', 'S2']

        self.K = opt.K
        self.register_buffer("queue1", torch.randn(opt.K, 100))
        self.queue1 = nn.functional.normalize(self.queue1, dim=0)
        self.register_buffer("queue2", torch.randn(opt.K, 100))
        self.queue2 = nn.functional.normalize(self.queue2, dim=0)
        self.register_buffer("queue_ptr", torch.zeros(1, dtype=torch.long))

        self.criterion = CrossEntropyLoss(
            num_classes= opt.class_num
        )
        self.acc_s1 = 0.0
        self.acc_s2 = 0.0
        self.acc_t = 0.0

    def set_input(self, input):
        data_s1, data_s2, data_t = input
        x_t, y_t, idx_t = data_t
        x1, y1, idx1 = data_s1
        x2, y2, idx2 = data_s2
        angle = np.random.uniform(self.opt.ang_min,self.opt.ang_max,1)[0]
        x_t = rotate(x_t, angle)
        self.x_seq = torch.cat([x1, x_t,x2],0).to(self.device)
        self.y_seq = torch.cat([y1, y_t, y2,],0).to(self.device)
    
        self.s1_size = x1.size(0)
        self.s2_size = x2.size(0)
        self.t_size = x_t.size(0)
        self.num_source_h = self.s1_size
        self.num_target = self.t_size
    
    def forward_stage0(self):
        self.optimizer.zero_grad()
        x_t = self.x_seq[self.num_source_h:-self.num_source_h]
        self.e_seq = self.netE(x_t)

        self.f_t1 = self.netF1(self.e_seq)
        self.f_t2 = self.netF2(self.e_seq)

        e_seq1, e_seq2 = self.queue1.clone().detach(), self.queue2.clone().detach()

        f_s1 = self.netF1(e_seq1)
        f_s2 = self.netF2(e_seq2)
        
        f_s1_adv = self.grl_layer1(e_seq1)
        f_s2_adv = self.grl_layer2(e_seq2)
        f_s1_adv = self.netF_adv1(f_s1_adv)
        f_s2_adv = self.netF_adv2(f_s2_adv)

        f_s12_adv = self.grl_layer2(e_seq1)
        f_s21_adv = self.grl_layer1(e_seq2)
        f_s12_adv = self.netF_adv2(f_s12_adv)
        f_s21_adv = self.netF_adv1(f_s21_adv)


        s_seq_adv1 = self.grl_layer1(self.e_seq)
        s_seq_adv2 = self.grl_layer2(self.e_seq)

        self.f_t1_adv = self.netF_adv1(s_seq_adv1)
        self.f_t2_adv = self.netF_adv2(s_seq_adv2)

        self.loss_E1 = self.mdd1(f_s1, f_s1_adv, self.f_t1, self.f_t1_adv)
        self.loss_E2 = self.mdd2(f_s2, f_s2_adv,self.f_t2, self.f_t2_adv)


        self.grl_layer1.step()
        self.grl_layer2.step()

        self.loss_adv = self.loss_E1+self.loss_E2 + 10* (r1_reg(self.loss_E1 + self.loss_E2,self.e_seq) - 1).pow(2)

        self.loss_S1 = self.loss_adv
        self.loss_S1.backward()
        self.optimizer.step()

    def forward_stage1(self):
        self.optimizer.zero_grad()
        x_s1, x_s2 = self.x_seq[:self.num_source_h], self.x_seq[-self.num_source_h:]
        x_s12 = torch.cat([x_s1, x_s2],0)
        self.e_seq = self.netE(x_s12)

        e_s1 = self.e_seq[:self.num_source_h]
        e_s2 = self.e_seq[-self.num_source_h:]

        self._dequeue_and_enqueue(e_s1, e_s2)
        
        e_seq1, e_seq2 = self.queue1.clone().detach(), self.queue2.clone().detach()

        self.f1 = self.netF1(e_s1)
        t1 = self.netF1(e_seq2)

        self.f2 = self.netF2(e_s2)
        t2 = self.netF2(e_seq1)

        self.f1_adv = self.netF_adv1(self.grl_layer1(e_s1))
        self.f2_adv = self.netF_adv2(self.grl_layer2(e_s2))
        
        self.t1_adv = self.netF_adv1(self.grl_layer1(e_seq2))
        self.t2_adv = self.netF_adv2(self.grl_layer2(e_seq1))

        self.loss_E_pred1 = self.criterion(self.f1,self.y_seq[:self.num_source_h])
        self.loss_E_pred2 = self.criterion(self.f2,self.y_seq[-self.num_source_h:])

        self.loss_E1 = self.mdd1(self.f1, self.f1_adv, t1, self.t1_adv)
        self.loss_E2 = self.mdd2(self.f2, self.f2_adv, t2, self.t2_adv)

        self.grl_layer1.step()
        self.grl_layer2.step()

        self.loss_E_adv = self.loss_E1+self.loss_E2
        self.loss_S2 = self.loss_E_pred1 + self.loss_E_pred2 + self.opt.lam * self.loss_E_adv
        self.loss_S2.backward()
        self.optimizer.step()
        
    def optimize_parameters_0(self):
        self.forward_stage0()
        #print(self.f_t.size(), self.y_seq[self.num_source_h:-self.num_source_h:].size())
        self.acc_t = accuracy((self.f_t1 + self.f_t2),self.y_seq[self.num_source_h:-self.num_source_h:])[0].item()

    def optimize_parameters_1(self):
        self.forward_stage1()
        self.acc_s1 = accuracy(self.f1[:self.num_source_h],self.y_seq[:self.num_source_h])[0].item()
        self.acc_s2 = accuracy(self.f2[-self.num_source_h:],self.y_seq[-self.num_source_h:])[0].item()


    def learn_1(self, epoch, source1_dataloader, source2_dataloader, target_dataloader):
        self.epoch = epoch

        self.train()

        loss_curve = {
            loss: []
            for loss in self.loss_names
        }
        source2_dataloader = iter(cycle(source2_dataloader))
        target_dataloader = iter(cycle(target_dataloader))
        source1_dataloader = source1_dataloader

        for data_s1 in source1_dataloader:
            data_s2 = next(source2_dataloader)
            data_t = next(target_dataloader)
            self.set_input(input=(data_s1, data_s2, data_t))

            self.optimize_parameters_1()
        
            for loss in self.loss_names:
                loss_curve[loss].append(getattr(self, 'loss_' + loss).detach().item())


        self.loss_msg = '[Train][{}] Loss:'.format(epoch)
        for loss in self.loss_names:
            self.loss_msg += ' {} {:.3f}'.format(loss, np.mean(loss_curve[loss]))

        self.acc_msg = '[Train][{}] Acc: source1 {:.3f}  source2 {:.3f} target {:.3f} '.format(
            epoch, self.acc_s1, self.acc_s2, self.acc_t)
        self.print_log()
        self.lr_scheduler.step()

    def learn_0(self, epoch, source1_dataloader, source2_dataloader, target_dataloader):
        self.epoch = epoch

        self.train()

        loss_curve = {
            loss: []
            for loss in self.loss_names
        }
        source2_dataloader = iter(cycle(source2_dataloader))
        source1_dataloader = iter(cycle(source1_dataloader))
        n_iter = 0
        num_batches = len(target_dataloader)
        for data_t in target_dataloader:
            data_s1 = next(source1_dataloader)
            data_s2 = next(source2_dataloader)

            self.set_input(input=(data_s1, data_s2, data_t))

            self.optimize_parameters_0()
            n = epoch//2 * num_batches + n_iter
            self.writer.add_scalar('mdddiff/train', self.loss_adv.cpu().detach().numpy(), n)

            for loss in self.loss_names:
                loss_curve[loss].append(getattr(self, 'loss_' + loss).detach().item())

            n_iter = n_iter+1

        self.loss_msg = '[Train][{}] Loss:'.format(epoch)
        for loss in self.loss_names:
            self.loss_msg += ' {} {:.3f}'.format(loss, np.mean(loss_curve[loss]))

        self.acc_msg = '[Train][{}] Acc: source1 {:.3f}  source2 {:.3f} target {:.3f} '.format(
            epoch, self.acc_s1, self.acc_s2, self.acc_t)
        self.print_log()
        self.lr_scheduler.step()
    
    def learn(self, epoch, source1_dataloader, source2_dataloader, target_dataloader):
        if epoch%2 ==0:
            self.learn_1(epoch, source1_dataloader, source2_dataloader, target_dataloader)
        else:
            self.learn_0(epoch, source1_dataloader, source2_dataloader, target_dataloader)

    def save(self,epoch):
        path = self.opt.outf + '/{}_{}_{}model.pth'.format(self.opt.s_pos, self.opt.split, epoch)
        torch.save(self.state_dict(), path)

    def print_log(self):
        print(self.loss_msg)
        print(self.acc_msg)

    def load(self):
        print('===> Loading model from {}'.format(self.model_path))
        try:
            self.load_state_dict(torch.load(self.model_path))
            print('<=== Success!')
        except:
            print('<==== Failed!')

    def set_data_stats(self, dm, ds):
        self.data_m, self.data_s = dm, ds


    @torch.no_grad()
    def test(self, epoch, dataloader):
        self.eval()
        y_hs = []
        y_s = []
        for data in dataloader:
            x, y, idx = data
            x , y = x.to(self.device),y.to(self.device)
            e = self.netE(x)
            f = self.netF1(e) + self.netF2(e)
            y_hs.append(f)
            y_s.append(y)

        y_hs = torch.cat(y_hs, 0)
        y_s = torch.cat(y_s, 0)
        acc = accuracy(y_hs,y_s)[0]
        print('acc:{}'.format(acc.cpu().item()))
       
    @torch.no_grad()
    def get_feats(self, dataloader, path):
        self.eval()
        feats = []
        for data in dataloader:
            x, y, idx = data
            x , y = x.to(self.device),y.to(self.device)
            e = self.netE(x)
            f = self.netF1(e) + self.netF2(e)
            feats.append(e)
        feats = torch.cat(feats, 0)
        np.save(path, feats.cpu().numpy())
       
    
    @torch.no_grad()
    def _dequeue_and_enqueue(self, keys1, keys2):
        # gather keys before updating queue
        batch_size = keys1.shape[0]
        ptr = int(self.queue_ptr)
        assert self.K % batch_size == 0  # for simplicity
        # replace the keys at ptr (dequeue and enqueue)
        self.queue1[ptr:ptr + batch_size] = keys1
        self.queue2[ptr:ptr + batch_size] = keys2
        ptr = (ptr + batch_size) % self.K  # move pointer
        self.queue_ptr[0] = ptr

def r1_reg(d_out, x_in):
    batch_size = x_in.size(0)
    grad_dout = torch.autograd.grad(
        outputs=d_out.sum(), inputs=x_in,
        create_graph=True, retain_graph=True, only_inputs=True
    )[0]
    grad_dout2 = grad_dout.pow(2)
    assert(grad_dout2.size() == x_in.size())
    reg = 0.5 * grad_dout2.view(batch_size, -1).sum(1).mean(0)
    return reg

