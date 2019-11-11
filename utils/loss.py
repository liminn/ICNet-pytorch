"""Custom losses."""
import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.autograd import Variable

__all__ = ['ICNetLoss']

# TODO: optim function
class ICNetLoss(nn.CrossEntropyLoss):
    """Cross Entropy Loss for ICNet"""
    
    def __init__(self, aux_weight=0.4, ignore_index=-1):
        super(ICNetLoss, self).__init__(ignore_index=ignore_index)
        self.aux_weight = aux_weight

    def forward(self, *inputs):
        preds, target = tuple(inputs)
        inputs = tuple(list(preds) + [target])

        pred, pred_sub4, pred_sub8, pred_sub16, target = tuple(inputs)
        # [batch, H, W] -> [batch, 1, H, W]
        target = target.unsqueeze(1).float()
        target_sub4 = F.interpolate(target, pred_sub4.size()[2:], mode='bilinear', align_corners=True).squeeze(1).long()
        target_sub8 = F.interpolate(target, pred_sub8.size()[2:], mode='bilinear', align_corners=True).squeeze(1).long()
        target_sub16 = F.interpolate(target, pred_sub16.size()[2:], mode='bilinear', align_corners=True).squeeze(
            1).long()
        loss1 = super(ICNetLoss, self).forward(pred_sub4, target_sub4)
        loss2 = super(ICNetLoss, self).forward(pred_sub8, target_sub8)
        loss3 = super(ICNetLoss, self).forward(pred_sub16, target_sub16)
        #return dict(loss=loss1 + loss2 * self.aux_weight + loss3 * self.aux_weight)
        return loss1 + loss2 * self.aux_weight + loss3 * self.aux_weight

