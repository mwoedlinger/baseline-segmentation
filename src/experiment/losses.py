import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable


# Taken from: https://github.com/clcarwin/focal_loss_pytorch/blob/master/focalloss.py
class FocalLoss(nn.Module):
    def __init__(self, gamma=0, alpha=None, size_average=True):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        if isinstance(alpha, (float,int)): self.alpha = torch.Tensor([alpha,1-alpha])
        if isinstance(alpha, list): self.alpha = torch.Tensor(alpha)
        self.size_average = size_average

    def forward(self, input, target):
        if input.dim()>2:
            input = input.view(input.size(0),input.size(1),-1)  # N,C,H,W => N,C,H*W
            input = input.transpose(1,2)    # N,C,H*W => N,H*W,C
            input = input.contiguous().view(-1,input.size(2))   # N,H*W,C => N*H*W,C
        target = target.view(-1,1)

        logpt = F.log_softmax(input)
        logpt = logpt.gather(1,target)
        logpt = logpt.view(-1)
        pt = Variable(logpt.data.exp())

        if self.alpha is not None:
            if self.alpha.type()!=input.data.type():
                self.alpha = self.alpha.type_as(input.data)
            at = self.alpha.gather(0,target.data.view(-1))
            logpt = logpt * Variable(at)

        loss = -1 * (1-pt)**self.gamma * logpt
        if self.size_average: return loss.mean()
        else: return loss.sum()


class DiceLoss(nn.Module):
    def __init__(self, valid_classes: list):
        super(DiceLoss, self).__init__()
        self.valid_classes = valid_classes
        self.num_valid_classes = len(valid_classes)

    def forward(self, input, target):
        class_map = torch.argmax(input, dim=1)

        iou = 0
        for c in self.valid_classes:
            pred_c = (class_map == c)
            label_c = (target == c)

            intersection = (pred_c & label_c)
            union = (pred_c | label_c)

            iou += torch.sum(intersection).float() / torch.sum(union).float()

        return iou/self.num_valid_classes

class CEDiceLoss(nn.Module):
    """
    The sum of crossentropy loss and Dice loss.
    """
    def __init__(self):
        super(CEDiceLoss, self).__init__()
        self.ce_loss = nn.CrossEntropyLoss()

    def forward(self, input, target):
        class_map = torch.argmax(input, dim=1)
        unique_labels = target.unique()
        num_valid_classes = len(unique_labels)

        iou = 0
        for c in self.unique_labels:
            pred_c = (class_map == c)
            label_c = (target == c)

            intersection = (pred_c & label_c)
            union = (pred_c | label_c)

            if torch.sum(union) > 0:
                iou += torch.sum(intersection).float() / torch.sum(union).float()
            else:
                iou = 0

        loss = self.ce_loss(input, target) + iou/num_valid_classes

        return loss


