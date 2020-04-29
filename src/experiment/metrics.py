import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable


def miou(input_tensor: torch.Tensor, labels: torch.Tensor, valid_classes: list):
    """
    Expects a torch tensor with shape (n_classes, h, w) as input and a lable tensor with shape (h, w) containing the
    true classes for each pixel. valid_classes is a list containing the valid classes (for example [0,1,2,3,7])
    :param input: the model ouput without batch dimension
    :param labels: the correct classes
    :param valid_classes: which classes to compute the meanIoU for, the rest will be ignored
    :return: a dict containing the per class iou scores
    """
    class_map = torch.argmax(input_tensor, dim=0)

    unique_labels = labels.unique()

    iou_scores = {}
    for c in valid_classes:
        pred_c = (class_map == c)
        label_c = (labels == c)

        intersection = (pred_c & label_c)
        union = (pred_c | label_c)

        if c in unique_labels:#torch.sum(union):
            iou = torch.sum(intersection).float()/torch.sum(union).float()

            iou_scores.update({c: iou.item()})

    return iou_scores


def compute_scores(input_tensor: torch.Tensor, labels: torch.Tensor, valid_classes: list):
    """
    Computes accuracy, precision, recall and f1 score class-wise.
    :param input_tensor: The input
    :param labels: The GT labels
    :param valid_classes: Classes not contained in this list will be ignored
    :return: a tuple of dictionaries: accuracy_scores, precision_scores, recall_scores, f1score_scores
    """
    class_map = torch.argmax(input_tensor, dim=0)

    accuracy_scores = {}
    precision_scores = {}
    recall_scores = {}
    f1score_scores = {}

    for c in valid_classes:
        pred_c = (class_map == c)
        label_c = (labels == c)

        pred_c_inv = (pred_c & 0)
        label_c_inv = (label_c & 0)

        TP = torch.sum(pred_c & label_c).float()
        TN = torch.sum(pred_c_inv & label_c_inv).float()
        FP = torch.sum(pred_c).float() - TP
        FN = torch.sum(label_c).float() - TP

        accuracy = (TP + TN)/(TP + TN + FP + FN)
        precision = TP/(TP + FP)
        recall = TP/(TP + FN)
        f1score = 2*TP/(2*TP + FP + FN)

        accuracy_scores.update({c: accuracy})
        precision_scores.update({c: precision})
        recall_scores.update({c: recall})
        f1score_scores.update({c: f1score})

    return accuracy_scores, precision_scores, recall_scores, f1score_scores



