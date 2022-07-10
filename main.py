import os
from easydict import EasyDict
import torch
import numpy as np
from torch.utils.data import DataLoader
from models import Baseline, CDA
from mnist import RotateEMNIST, EMNIST
import random


def set_opt():
    opt = EasyDict()
    opt.seed = 2000
    opt.domain_num = 9
    opt.half_source_num = 1
    opt.train_domain_num = 3
    opt.test_domain_num = 4 
    opt.batch_size = 512
    opt.uda = True
    opt.class_num = 47
    opt.device = 'cuda'
    opt.nz =100
    opt.model = 'CDA'
    opt.exp =  opt.model
    opt.outf = './dump/' + opt.exp
    opt.num_epoch = 200
    opt.lam = 1.0
    opt.lr = 1e-4
    opt.gamma = 100
    opt.beta1 = 0.9
    opt.lmbd = 0.25
    opt.weight_decay = 5e-4
    opt.K = 65536
    opt.c = 0.01

    opt.s_pos = 75
    opt.split = 1

    if opt.s_pos ==100:
        opt.s1_ang =  0#(22.5,22.5) (0,0)(45,45) (67.5,67.5)
        opt.s2_ang =  180#(157.5,157.5) (180,180)(135,135) (112.5,112.5)
    elif  opt.s_pos ==75:
        opt.s1_ang =  22.5
        opt.s2_ang =  157.5
    elif opt.s_pos == 50:
        opt.s1_ang =  45
        opt.s2_ang =  135
    elif opt.s_pos == 25:
        opt.s1_ang =  67.5
        opt.s2_ang =  112.5

    if opt.split == 1:
            opt.ang_min = 0
            opt.ang_max = 180
    elif opt.split == 2:
        opt.ang_min = 0
        opt.ang_max = opt.s2_ang
    elif opt.split == 3:
        opt.ang_min = opt.s1_ang
        opt.ang_max = opt.s2_ang
    elif opt.split == 4:
        opt.ang_min = 0
        opt.ang_max = opt.s1_ang

    print(opt.s1_ang, opt.s2_ang)
    print(opt.ang_min, opt.ang_max)
    return opt

def setup_seed(opt):
    seed = opt.seed
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = True

def get_dataloader(opt):
    dataset_s1 = RotateEMNIST(opt.s1_ang) # 45, 135,   22.5, 157.5 ,  67.5, 112.5
    dataset_s2 = RotateEMNIST(opt.s2_ang)
    dataset_t = EMNIST()
    dataset_test = RotateEMNIST(0,train=False)
    print(len(dataset_s1))

    dataloader_s1 = DataLoader(dataset_s1, batch_size=opt.batch_size, shuffle=True,drop_last=True,num_workers=2, pin_memory=True)
    dataloader_s2 = DataLoader(dataset_s2, batch_size=opt.batch_size, shuffle=True,drop_last=True,num_workers=2, pin_memory=True)
    dataloader_t = DataLoader(dataset_t, batch_size=opt.batch_size, shuffle=True,drop_last=True,num_workers=2, pin_memory=True)
    dataloader_test = DataLoader(dataset_test, batch_size=opt.batch_size, shuffle=True,drop_last=True,num_workers=2, pin_memory=True)
    return dataloader_s1, dataloader_s2, dataloader_t, dataloader_test

if __name__ =='__main__':
    
    opt = set_opt()
    setup_seed(opt)
    np.random.seed(opt.seed)
    torch.manual_seed(opt.seed)
    print('mdoel:' , opt.model)
    if opt.model!='CUA':
        dataloader_s1, dataloader_s2, dataloader_t, dataloader_test = get_dataloader(opt)
        if opt.model == 'base':
            model = Baseline(opt).cuda()
            opt.uda = False
        elif opt.model == 'CDA':
            model = CDA(opt).cuda()

        if opt.uda:
            for epoch in range(opt.num_epoch):
                model.learn(epoch, dataloader_s1, dataloader_s2, dataloader_t)
                if (epoch + 1) % 10 == 0 or (epoch + 1) == opt.num_epoch:
                    model.save(epoch)
                if (epoch + 1) % 10 == 0 or (epoch+1)==opt.num_epoch:    
                    model.test(epoch, dataloader_test)
        else:
            for epoch in range(opt.num_epoch):
                model.learn(epoch, dataloader_s1, dataloader_s2)
                if (epoch + 1) % 10 == 0 or (epoch + 1) == opt.num_epoch:
                    model.save()
                if (epoch + 1) % 10 == 0 or (epoch+1)==opt.num_epoch:    
                    model.test(epoch, dataloader_test)
   
    

