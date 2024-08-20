import torch
import torch.nn as nn
#######损失函数#######
def dice_coeff(y_true, y_pred):
    smooth = 1.
    y_true_f = y_true.view(-1)
    y_pred_f = y_pred.view(-1)
    intersection = torch.sum(y_true_f * y_pred_f)
    score = (2. * intersection + smooth) / (torch.sum(y_true_f) + torch.sum(y_pred_f) + smooth)
    return score


def dice_loss(y_true, y_pred):
    loss_dice = 1 - dice_coeff(y_true, y_pred)
    return loss_dice

def bce_dice_loss(y_true, y_pred):

    # y_true = y_true.data.cpu()
    y_pred = torch.sigmoid(y_pred)
    loss_bce = 0.4 * nn.BCELoss(reduction='mean')(y_pred, y_true) + 0.6*dice_loss(y_true, y_pred)

    return loss_bce
###################