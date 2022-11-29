import logging
import random
import sys
import time

import numpy as np
import torch
from pathlib import Path
from tensorboardX import SummaryWriter
from torch import nn as nn, optim as optim
from torch.nn import functional as F
from torch.utils.data import DataLoader

import utils.statistic_coranet as statistic

def to_cuda(tensors, device=None):
    res = []
    if isinstance(tensors, (list, tuple)):
        for t in tensors:
            res.append(to_cuda(t, device))
        return res
    elif isinstance(tensors, (dict,)):
        res = {}
        for k, v in tensors.items():
            res[k] = to_cuda(v, device)
        return res
    else:
        if isinstance(tensors, torch.Tensor):
            if device is None:
                return tensors.cuda()
            else:
                return tensors.to(device)
        else:
            return tensors


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
        return self

    def update(self, val, n=1):
        self.val = val
        self.sum += val
        self.count += n
        self.avg = self.sum / self.count
        return self


def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)


def get_current_consistency_weight(current, rampup_length):
    """Exponential rampup from https://arxiv.org/abs/1610.02242"""
    if rampup_length == 0:
        return 1.0
    else:
        current = np.clip(current, 0.0, rampup_length)
        phase = 1.0 - current / rampup_length
        return float(np.exp(-5.0 * phase * phase))


def save_net_opt(net, optimizer, path, epoch):
    state = {
        'net': net.state_dict(),
        'opt': optimizer.state_dict(),
        'epoch': epoch,
    }
    torch.save(state, str(path))


def load_net_opt(net, optimizer, path):
    state = torch.load(str(path))
    net.load_state_dict(state['net'])
    optimizer.load_state_dict(state['opt'])


def count_param(model):
    param_count = 0
    for param in model.parameters():
        param_count += param.view(-1).size()[0]
    return param_count


def get_mask(out, thres=0.5):
    probs = F.softmax(out, 1)
    masks = (probs >= thres).float()
    masks = masks[:, 1, :, :].contiguous()
    return masks


@torch.no_grad()
# def pred_unlabel(net, pred_loader, batch_size):
#     unimg, unlab, unmask, labs = [], [], [], []
#     plab_dice = 0
#     for (step, data) in enumerate(pred_loader):
#         img, lab = data
#         img, lab = img.cuda(), lab.cuda()
#         out = net(img)
#         plab0 = get_mask(out[0])
#         plab1 = get_mask(out[1])
#         plab2 = get_mask(out[2])
#
#         mask = (plab1 == plab2).long()
#         plab = plab0
#         unimg.append(img)
#         unlab.append(plab)
#         unmask.append(mask)
#         labs.append(lab)
#
#         plab_dice += statistic.dice_ratio(plab, lab)
#     plab_dice /= len(pred_loader)
#     new_loader = DataLoader(PancreasSTDataset(unimg, unlab, unmask, labs), batch_size=batch_size, shuffle=True, num_workers=0, drop_last=True)
#     return new_loader, plab_dice


def config_log(save_path, tensorboard=False):
    writer = SummaryWriter(str(save_path), filename_suffix=time.strftime('_%Y-%m-%d_%H-%M-%S')) if tensorboard else None

    save_path = str(Path(save_path) / 'log.txt')
    formatter = logging.Formatter('%(levelname)s [%(asctime)s] %(message)s')

    logger = logging.getLogger(save_path.split('/')[-2])
    logger.setLevel(logging.INFO)

    handler = logging.FileHandler(save_path)
    handler.setFormatter(formatter)
    logger.addHandler(handler)

    sh = logging.StreamHandler(sys.stdout)
    handler.setFormatter(formatter)
    logger.addHandler(sh)

    return logger, writer


class Measures():
    def __init__(self, keys, writer, logger):
        self.keys = keys
        self.measures = {k: AverageMeter() for k in self.keys}
        self.writer = writer
        self.logger = logger

    def reset(self):
        [v.reset() for v in self.measures.values()]


class PretrainMeasures(Measures):
    def __init__(self, writer, logger):
        keys = ['loss_ce', 'loss_dice', 'loss_con', 'loss_rad', 'loss_all', 'train_dice']
        super(PretrainMeasures, self).__init__(keys, writer, logger)

    def update(self, out, lab, *args):
        args = list(args)
        masks = get_mask(out[0])
        train_dice = statistic.dice_ratio(masks, lab)
        args.append(train_dice)

        dict_variables = dict(zip(self.keys, args))
        for k, v in dict_variables.items():
            self.measures[k].update(v)

    def log(self, epoch, step):
        # self.logger.info('epoch : %d, step : %d, train_loss: %.4f, train_dice: %.4f' % (
        #     epoch, step, self.measures['loss_all'].avg, self.measures['train_dice'].avg))

        log_string, params = 'Epoch : {}', []
        for k in self.keys:
            log_string += ', ' + k + ': {:.4f}'
            params.append(self.measures[k].val)
        self.logger.info(log_string.format(epoch, *params))

        for k, measure in self.measures.items():
            k = 'pretrain/' + k
            self.writer.add_scalar(k, measure.avg, step)
        self.writer.flush()


class STMeasures(Measures):
    def __init__(self, writer, logger):
        keys = ['train_loss', 'sup_all_loss', 'sup_ce_loss', 'sup_rad_loss', 'sup_con_loss', 'sup_dice_loss',
                'certain_all_loss', 'certain_ce_loss', 'certain_rad_loss', 'certain_con_loss', 'uncertain_loss',
                'train_dice', 'unlab_dice', 'unlab_rad_dice', 'unlab_con_dice', 'lab_con_dice', 'lab_rad_dice']
        super(STMeasures, self).__init__(keys, writer, logger)

    @torch.no_grad()
    def update(self, out1, out2, lab1, lab2, *args):
        mask1 = get_mask(out1[0])
        mask2 = get_mask(out2[0])
        dices = [statistic.dice_ratio(mask1, lab1), statistic.dice_ratio(mask2, lab2), statistic.dice_ratio(get_mask(out1[2]), lab1),
                 statistic.dice_ratio(get_mask(out2[2]), lab2), statistic.dice_ratio(get_mask(out1[1]), lab1),
                 statistic.dice_ratio(get_mask(out2[1]), lab2)]
        args = list(args)
        args.extend(dices)
        dict_variables = dict(zip(self.keys, args))

        for k, v in dict_variables.items():
            self.measures[k].update(v)

    def log(self, epoch):
        log_keys = ['train_loss', 'sup_all_loss', 'certain_all_loss', 'uncertain_loss',
                    'train_dice', 'unlab_dice', 'lab_rad_dice', 'lab_con_dice', 'unlab_rad_dice', 'unlab_con_dice']
        log_string = 'Epoch : {}'
        params = []
        for k in log_keys:
            log_string += ', ' + k + ': {:.4f}'
            params.append(self.measures[k].val)
        self.logger.info(log_string.format(epoch, *params))

    def write_tensorboard(self, epoch):
        for k, measure in self.measures.items():
            if 'sup' in k or 'train_loss' in k:
                k = 'supervised_loss/' + k
            elif 'certain' in k:
                k = 'upsupervised_loss/' + k
            else:
                k = 'dice/' + k
            self.writer.add_scalar(k, measure.avg, epoch)
        self.writer.flush()

############################################## loss ##############################################
def to_one_hot(tensor, nClasses):
    """ Input tensor : Nx1xHxW
    :param tensor:
    :param nClasses:
    :return:
    """
    assert tensor.max().item() < nClasses, 'one hot tensor.max() = {} < {}'.format(torch.max(tensor), nClasses)
    assert tensor.min().item() >= 0, 'one hot tensor.min() = {} < {}'.format(tensor.min(), 0)

    size = list(tensor.size())
    assert size[1] == 1
    size[1] = nClasses
    one_hot = torch.zeros(*size)
    if tensor.is_cuda:
        one_hot = one_hot.cuda(tensor.device)
    one_hot = one_hot.scatter_(1, tensor, 1)
    return one_hot


def get_probability(logits):
    """ Get probability from logits, if the channel of logits is 1 then use sigmoid else use softmax.
    :param logits: [N, C, H, W] or [N, C, D, H, W]
    :return: prediction and class num
    """
    size = logits.size()
    # N x 1 x H x W
    if size[1] > 1:
        pred = F.softmax(logits, dim=1)
        nclass = size[1]
    else:
        pred = F.sigmoid(logits)
        pred = torch.cat([1 - pred, pred], 1)
        nclass = 2
    return pred, nclass


class DiceLoss(nn.Module):
    def __init__(self, nclass, class_weights=None, smooth=1e-5):
        super(DiceLoss, self).__init__()
        self.smooth = smooth
        if class_weights is None:
            # default weight is all 1
            self.class_weights = nn.Parameter(torch.ones((1, nclass)).type(torch.float32), requires_grad=False)
        else:
            class_weights = np.array(class_weights)
            assert nclass == class_weights.shape[0]
            self.class_weights = nn.Parameter(torch.tensor(class_weights, dtype=torch.float32), requires_grad=False)

    def prob_forward(self, pred, target, mask=None):
        size = pred.size()
        N, nclass = size[0], size[1]
        # N x C x H x W
        pred_one_hot = pred.view(N, nclass, -1)
        target = target.view(N, 1, -1)
        target_one_hot = to_one_hot(target.type(torch.long), nclass).type(torch.float32)

        # N x C x H x W
        inter = pred_one_hot * target_one_hot
        union = pred_one_hot + target_one_hot

        if mask is not None:
            mask = mask.view(N, 1, -1)
            inter = (inter.view(N, nclass, -1) * mask).sum(2)
            union = (union.view(N, nclass, -1) * mask).sum(2)
        else:
            # N x C
            inter = inter.view(N, nclass, -1).sum(2)
            union = union.view(N, nclass, -1).sum(2)

        # smooth to prevent overfitting
        # [https://github.com/pytorch/pytorch/issues/1249]
        # NxC
        dice = (2 * inter + self.smooth) / (union + self.smooth)
        return 1 - dice.mean()

    def forward(self, logits, target, mask=None):
        size = logits.size()
        N, nclass = size[0], size[1]

        logits = logits.view(N, nclass, -1)
        target = target.view(N, 1, -1)

        pred, nclass = get_probability(logits)

        # N x C x H x W
        pred_one_hot = pred
        target_one_hot = to_one_hot(target.type(torch.long), nclass).type(torch.float32)

        # N x C x H x W
        inter = pred_one_hot * target_one_hot
        union = pred_one_hot + target_one_hot

        if mask is not None:
            mask = mask.view(N, 1, -1)
            inter = (inter.view(N, nclass, -1) * mask).sum(2)
            union = (union.view(N, nclass, -1) * mask).sum(2)
        else:
            # N x C
            inter = inter.view(N, nclass, -1).sum(2)
            union = union.view(N, nclass, -1).sum(2)

        # smooth to prevent overfitting
        # [https://github.com/pytorch/pytorch/issues/1249]
        # NxC
        dice = (2 * inter + self.smooth) / (union + self.smooth)
        return 1 - dice.mean()


def softmax_mse_loss(input_logits, target_logits):
    """Takes softmax on both sides and returns MSE loss
    Note:
    - Returns the sum over all examples. Divide by the batch size afterwards
      if you want the mean.
    - Sends gradients to inputs but not the targets.
    """
    assert input_logits.size() == target_logits.size()
    input_softmax = F.softmax(input_logits, dim=1)
    target_softmax = F.softmax(target_logits, dim=1)

    mse_loss = (input_softmax - target_softmax) ** 2
    return mse_loss
