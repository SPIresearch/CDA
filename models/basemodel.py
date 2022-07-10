import torch.nn as nn
import torch
import numpy as np
import torch.nn.functional as F
from torch.autograd import Function

def to_np(x):
    return x.detach().cpu().numpy()


def to_tensor(x, device="cuda"):
    if isinstance(x, np.ndarray):
        x = torch.from_numpy(x).to(device)
    else:
        x = x.to(device)
    return x


def chamfer_dist(p1, p2):
    diff = p1[:, :, None, :] - p2[:, None, :, :]
    d = (diff ** 2).sum(3)
    d1 = torch.min(d, 1)[0]
    d2 = torch.min(d, 2)[0]
    return d1.mean(1) + d2.mean(1)

class nnSqueeze(nn.Module):
    def __init__(self):
        super(nnSqueeze, self).__init__()

    def forward(self, x):
        return torch.squeeze(x)


class nnUnsqueeze(nn.Module):
    def __init__(self):
        super(nnUnsqueeze, self).__init__()

    def forward(self, x):
        return x[:, :, None, None]

class Encoder(nn.Module):
    def __init__(self, opt):
        super(Encoder, self).__init__()

        nh = 256

        nz = opt.nz

        self.conv = nn.Sequential(
            nn.Conv2d(1, nh, 3, 2, 1), nn.BatchNorm2d(nh), nn.ReLU(True),   # 32 x 32
            nn.Conv2d(nh, nh, 3, 2, 1), nn.BatchNorm2d(nh), nn.ReLU(True),  # 16 x 16
            nn.Conv2d(nh, nh, 3, 2, 1), nn.BatchNorm2d(nh), nn.ReLU(True),  # 8 x 8
            nn.Conv2d(nh, nz, 4, 1, 0), nn.ReLU(True),  # 1 x 1
        )

        self.fc_pred = nn.Sequential(
            nn.Conv2d(nz, nh, 1, 1, 0), nn.BatchNorm2d(nh), nn.ReLU(True),
            nn.Conv2d(nh, nh, 1, 1, 0), nn.BatchNorm2d(nh), nn.ReLU(True),
            nnSqueeze(),
            nn.Linear(nh, opt.class_num),
        )

    def forward(self, x):
        """
        :param x: B x 1 x 32 x 32
        :param x: B x nu
        :return:
        """
        v = self.conv(x)
        x = self.fc_pred(v)
        return x, v


        



class DiscConv(nn.Module):
    def __init__(self, nin, nout):
        super(DiscConv, self).__init__()
        nh = 512
        self.net = nn.Sequential(
            nn.Conv2d(nin, nh, 1, 1, 0), nn.BatchNorm2d(nh), nn.LeakyReLU(),
            nn.Conv2d(nh, nh, 1, 1, 0), nn.BatchNorm2d(nh), nn.LeakyReLU(),
            nn.Conv2d(nh, nh, 1, 1, 0), nn.BatchNorm2d(nh), nn.LeakyReLU(),
            nnSqueeze(),
            nn.Linear(nh, nout),
        )

    def forward(self, x):
        return self.net(x)


class FeatureNet(nn.Module):
    def __init__(self, opt):
        super(FeatureNet, self).__init__()

        nh = 256

        nz = opt.nz

        self.conv = nn.Sequential(
            nn.Conv2d(1, nh, 3, 2, 1), nn.BatchNorm2d(nh), nn.ReLU(True),   # 32 x 32
            nn.Conv2d(nh, nh, 3, 2, 1), nn.BatchNorm2d(nh), nn.ReLU(True),  # 16 x 16
            nn.Conv2d(nh, nh, 3, 2, 1), nn.BatchNorm2d(nh), nn.ReLU(True),  # 8 x 8
            nn.Conv2d(nh, nz, 4, 1, 0), nn.ReLU(True), 
            nnSqueeze() # 1 x 1
        )
    def forward(self, x):
        x = self.conv(x)
        return x 

class FeatureNetDRA(nn.Module):
    def __init__(self, opt):
        super(FeatureNetDRA, self).__init__()

        nh = 256

        nz = opt.nz


        self.conv1 = nn.Sequential(nn.Conv2d(1, nh, 3, 2, 1), nn.BatchNorm2d(nh), nn.ReLU(True))   # 32 x 32
        self.conv2 = nn.Sequential(nn.Conv2d(nh, nh, 3, 1, 1), nn.BatchNorm2d(nh), nn.ReLU(True))  # 16 x 16
        self.conv3 = nn.Sequential(nn.Conv2d(nh, nh, 3, 2, 1), nn.BatchNorm2d(nh), nn.ReLU(True))  # 8 x 8
        self.conv4 = nn.Sequential(nn.Conv2d(nh, nz, 4, 1, 0), nn.ReLU(True))
        self.pool = nn.AvgPool2d(2,2)

        self.convs1 = nn.Conv2d(nh, nh, 1, 1, 0, bias=False)
        self.convs2 = nn.Conv2d(nh, nh, 1, 1, 0, bias=False)
        self.convs3 = nn.Conv2d(nh, nh, 1, 1, 0, bias=False)
        self.convs4 = nn.Conv2d(nh, nh, 1, 1, 0, bias=False)

        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(nh, nh//16, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(nh//16, 4, bias=False),
        )
        self.sf = nn.Softmax(dim=1)

    def forward(self, x):
        
        x = self.conv1(x)
        b, c, h, w = x.size()

        y = self.fc(self.avg_pool(x).view(b, c)).view(b, 4, 1, 1, 1)
        y = self.sf(y)

        res = self.convs1(x)*y[:,0] + self.convs2(x)*y[:,1] + \
            self.convs3(x)*y[:,2] + self.convs4(x)*y[:,3]
        
        x = res + self.conv2(x)
        x = self.pool(x)
        x = self.conv3(x)
        x = self.conv4(x)

        return torch.squeeze(x)


from torch.nn import utils
class PredNet_Norm(nn.Module):
    def __init__(self, opt):
        super(PredNet_Norm, self).__init__()

        nh = 256

        nz = opt.nz

        self.fc_pred = nn.Sequential(
           utils.spectral_norm( nn.Linear(nz, nh)), nn.BatchNorm1d(nh), nn.ReLU(True),
            utils.spectral_norm( nn.Linear(nh, nh)), nn.BatchNorm1d(nh), nn.ReLU(True),
            utils.spectral_norm( nn.Linear(nh, opt.class_num))
        )

    def forward(self, x):
        x = self.fc_pred(x)
        return x

class PredNet(nn.Module):
    def __init__(self, opt):
        super(PredNet, self).__init__()

        nh = 256

        nz = opt.nz

        self.fc_pred = nn.Sequential(
            nn.Linear(nz, nh), nn.BatchNorm1d(nh), nn.ReLU(True),
            nn.Linear(nh, nh), nn.BatchNorm1d(nh), nn.ReLU(True),
            nn.Linear(nh, opt.class_num)
        )

    def forward(self, x):
        x = self.fc_pred(x)
        return x


class GradReverse(Function):
    @staticmethod
    def forward(grev, x, lmbd):
        #self.lmbd = lmbd
        grev.lmbd = lmbd
        return x.view_as(x)

    @staticmethod
    def backward(grev, grad_output):
        #print (grev.lmbd)
        return (grad_output*-grev.lmbd), None
def grad_reverse(x,lambd=1.0):
    return GradReverse.apply(x,lambd)

class ResPredNet(nn.Module):
    def __init__(self, opt):
        super(ResPredNet, self).__init__()

        nh = 256

        nz = opt.nz

        self.fc_pred = nn.Sequential(
            nn.Linear(nz, nh), nn.BatchNorm1d(nh), nn.ReLU(True),
            nn.Linear(nh, nh), nn.BatchNorm1d(nh), nn.ReLU(True),
            nn.Linear(nh, opt.class_num)
        )

    def set_lambda(self, lambd):
        self.lambd = lambd

    def forward(self, x, reverse=False):
        if reverse:
            x = grad_reverse(x, self.lambd)
        x = self.fc_pred(x)
        return x
