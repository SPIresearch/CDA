from typing import Optional, List, Dict, Tuple, Callable
import torch.nn as nn
import torch.nn.functional as F
import torch
from cross_entropy_loss import CrossEntropyLoss

class MarginDisparityDiscrepancy(nn.Module):
    def __init__(self, source_disparity: Callable, target_disparity: Callable,
                 margin: Optional[float] = 4, reduction: Optional[str] = 'mean'):
        super(MarginDisparityDiscrepancy, self).__init__()
        self.margin = margin
        self.reduction = reduction
        self.source_disparity = source_disparity
        self.target_disparity = target_disparity

    def forward(self, y_s: torch.Tensor, y_s_adv: torch.Tensor, y_t: torch.Tensor, y_t_adv: torch.Tensor,
                w_s: Optional[torch.Tensor] = None, w_t: Optional[torch.Tensor] = None) -> torch.Tensor:
        source_loss = 1.38 * self.source_disparity(y_s, y_s_adv)
        target_loss = self.target_disparity(y_t, y_t_adv)
        if w_s is None:
            w_s = torch.ones_like(source_loss)
        source_loss = source_loss * w_s
        if w_t is None:
            w_t = torch.ones_like(target_loss)
        target_loss = target_loss * w_t

        if self.reduction == 'mean':
            loss = source_loss.mean() + target_loss.mean()
        elif self.reduction == 'sum':
            loss = source_loss.sum() + target_loss.sum()
        return loss


class ClfMDD(MarginDisparityDiscrepancy):
    def __init__(self, number_class, margin: Optional[float] = 4, **kwargs):
        
        def source_discrepancy(y: torch.Tensor, y_adv: torch.Tensor):
            _, prediction = y.max(dim=1)
            return self.ce(y_adv, prediction, reduction='none')

        def target_discrepancy(y: torch.Tensor, y_adv: torch.Tensor):
            _, prediction = y.max(dim=1)
            targets = torch.zeros(y.size()).scatter_(1, prediction.unsqueeze(1).data.cpu(), 1)
            targets = targets.cuda()
            target = targets * (1 - self.epsilon) + self.epsilon / self.num_classes
            return torch.sum(-target * shift_log(1. - F.softmax(y_adv, dim=1)), dim=1)

        super(ClfMDD, self).__init__(source_discrepancy, target_discrepancy, margin,
                                                                       **kwargs)
        self.ce = CrossEntropyLoss(number_class)
        self.epsilon = 0.1
        self.num_classes = number_class


def shift_log(x: torch.Tensor, offset: Optional[float] = 1e-6) -> torch.Tensor:
    r"""
    First shift, then calculate log, which can be described as:

    .. math::
        y = \max(\log(x+\text{offset}), 0)

    Used to avoid the gradient explosion problem in log(x) function when x=0.

    Args:
        x (torch.Tensor): input tensor
        offset (float, optional): offset size. Default: 1e-6

    .. note::
        Input tensor falls in [0., 1.] and the output tensor falls in [-log(offset), 0]
    """
    return torch.log(torch.clamp(x + offset, max=1.))
