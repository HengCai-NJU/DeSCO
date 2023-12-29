import os
import sys
from tqdm import tqdm
from tensorboardX import SummaryWriter
import shutil
import argparse
import logging
import time
import random
import numpy as np

import torch
import torch.optim as optim
from torchvision import transforms
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
from torchvision.utils import make_grid

from networks.vnet import VNet
from dataloaders import utils
from utils import ramps, losses
from dataloaders.la_heart import LAHeart1,LAHeart2, LARandomCrop1,LARandomCrop2, LARandomRotFlip, ToTensor, TwoStreamBatchSampler
from dataloaders.kits import KiTS1,KiTS2, KRandomCrop1,KRandomCrop2, KRandomRotFlip, ToTensor, TwoStreamBatchSampler
from dataloaders.lits import LiTS1,LiTS2,LRandomCrop1,LRandomCrop2, LRandomRotFlip, ToTensor, TwoStreamBatchSampler
parser = argparse.ArgumentParser()
parser.add_argument('--root_path', type=str, default='../data/2018LA_Seg_Training Set/', help='Name of Experiment')
parser.add_argument('--exp', type=str,  default='UAMT_unlabel', help='model_name')
parser.add_argument('--max_iterations', type=int,  default=6000, help='maximum epoch number to train')
parser.add_argument('--batch_size', type=int, default=2, help='batch_size per gpu')
parser.add_argument('--labeled_bs', type=int, default=1, help='labeled_batch_size per gpu')
parser.add_argument('--base_lr', type=float,  default=0.01, help='maximum epoch number to train')
parser.add_argument('--dataset',type=str,default="la",help='dataset to use')
parser.add_argument('--deterministic', type=int,  default=1, help='whether use deterministic training')
parser.add_argument('--seed', type=int,  default=1337, help='random seed')
parser.add_argument('--gpu', type=str,  default='0', help='GPU to use')
parser.add_argument('--slice_weight', type=float,  default=0.95, help='initial slice_weight')
parser.add_argument('--split', type=str,  default='train', help='datalist to use')
parser.add_argument('--slice_weight_step', type=int,  default=150, help='slice weight step')
parser.add_argument('--ema_decay', type=float,  default=0.99, help='ema_decay')
parser.add_argument('--consistency', type=float,  default=0.1, help='consistency')
parser.add_argument('--consistency_rampup', type=float,  default=40.0, help='consistency_rampup')
args = parser.parse_args()

train_data_path = args.root_path
snapshot_path = "../model/" + args.exp + "/"

os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
batch_size = args.batch_size * len(args.gpu.split(','))
max_iterations = args.max_iterations
base_lr = args.base_lr
labeled_bs = args.labeled_bs

if args.deterministic:
    cudnn.benchmark = False
    cudnn.deterministic = True
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

num_classes = 2
patch_size = (112, 112, 80)
if args.dataset=="la":
    patch_size = (112, 112, 80)
    train_data_path = "../data/LA"
elif args.dataset=='kits_kidney':
    patch_size=(128,128,64)
    train_data_path="../data/KiTS"
elif args.dataset=='lits_liver':
    patch_size=(176,176,64)
    train_data_path="../data/LiTS"


def get_current_slice_weight(epoch):
    return ramps.cosine_rampdown(epoch, args.consistency_rampup)

def get_current_consistency_weight(epoch):
    # Consistency ramp-up from https://arxiv.org/abs/1610.02242
    return args.consistency * ramps.sigmoid_rampup(epoch, args.consistency_rampup)
    #return args.consistency

def update_ema_variables(model, ema_model, alpha, global_step):
    # Use the true average until the exponential average is more correct
    alpha = min(1 - 1 / (global_step + 1), alpha)
    for ema_param, param in zip(ema_model.parameters(), model.parameters()):
        ema_param.data.mul_(alpha).add_(1 - alpha, param.data)

if __name__ == "__main__":
    ## make logger file
    if not os.path.exists(snapshot_path):
        os.makedirs(snapshot_path)
    if os.path.exists(snapshot_path + '/code'):
        shutil.rmtree(snapshot_path + '/code')
    shutil.copytree('.', snapshot_path + '/code', shutil.ignore_patterns(['.git', '__pycache__']))

    logging.basicConfig(filename=snapshot_path+"/log.txt", level=logging.INFO,
                        format='[%(asctime)s.%(msecs)03d] %(message)s', datefmt='%H:%M:%S')
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    logging.info(str(args))

    def create_model(ema=False):
        # Network definition
        net = VNet(n_channels=1, n_classes=num_classes, normalization='batchnorm', has_dropout=True)
        model = net.cuda()
        if ema:
            for param in model.parameters():
                param.detach_()
        return model

    m_seg1 = create_model()
    m_seg2 = create_model()
    if args.dataset == "la":
        db_train1 = LAHeart1(base_dir=train_data_path,
                           split=args.split,
                           transform = transforms.Compose([
                               LARandomCrop1(patch_size,args.slice_weight),
                               LARandomRotFlip(),
                               ToTensor(),
                           ]))
        db_train2 = LAHeart2(base_dir=train_data_path,
                             split=args.split,
                             transform=transforms.Compose([
                                 LARandomCrop2(patch_size,args.slice_weight),
                                 LARandomRotFlip(),
                                 ToTensor(),
                             ]))
        labeled_idxs = list(range(5))
        unlabeled_idxs = list(range(5, 80))
    elif args.dataset == 'kits_kidney':
        db_train1 = KiTS1(base_dir=train_data_path,
                        split=args.split,
                        transform=transforms.Compose([
                            KRandomCrop1(patch_size,args.slice_weight),
                            KRandomRotFlip(),
                            ToTensor(),
                        ]))
        db_train2 = KiTS2(base_dir=train_data_path,
                          split=args.split,
                          transform=transforms.Compose([
                              KRandomCrop2(patch_size,args.slice_weight),
                              KRandomRotFlip(),
                              ToTensor(),
                          ]))
        labeled_idxs = list(range(5))
        unlabeled_idxs = list(range(5, 190))
    elif args.dataset == 'lits_liver':
        db_train1 = LiTS1(base_dir=train_data_path,
                          split=args.split,
                          transform=transforms.Compose([
                              LRandomCrop1(patch_size,args.slice_weight),
                              LRandomRotFlip(),
                              ToTensor(),
                          ]))
        db_train2 = LiTS2(base_dir=train_data_path,
                          split=args.split,
                          transform=transforms.Compose([
                              LRandomCrop2(patch_size,args.slice_weight),
                              LRandomRotFlip(),
                              ToTensor(),
                          ]))
        labeled_idxs = list(range(5))
        unlabeled_idxs = list(range(5, 100))

    batch_sampler = TwoStreamBatchSampler(labeled_idxs, unlabeled_idxs, batch_size, batch_size - labeled_bs)


    def worker_init_fn(worker_id):
        random.seed(args.seed + worker_id)


    trainloader1 = DataLoader(db_train1, batch_sampler=batch_sampler, num_workers=4, pin_memory=True,
                              worker_init_fn=worker_init_fn)
    trainloader2 = DataLoader(db_train2, batch_sampler=batch_sampler, num_workers=4, pin_memory=True,
                              worker_init_fn=worker_init_fn)

    optimizer1 = optim.SGD(m_seg1.parameters(), lr=base_lr, momentum=0.9, weight_decay=0.0001)
    optimizer2 = optim.SGD(m_seg2.parameters(), lr=base_lr, momentum=0.9, weight_decay=0.0001)

    writer = SummaryWriter(snapshot_path + '/log')
    logging.info("{} itertations per epoch".format(len(trainloader1)))

    iter_num = 0
    max_epoch = max_iterations // len(trainloader1) + 1
    lr_ = base_lr
    contrast_model = None
    m_seg1.train()
    m_seg2.train()
    for epoch_num in tqdm(range(max_epoch), ncols=70):
        time1 = time.time()
        for i_batch, (sampled_batch1, sampled_batch2) in enumerate(zip(trainloader1, trainloader2)):
            time2 = time.time()
            volume_batch, label_batch, maskz = sampled_batch1['image'], sampled_batch1['label'], sampled_batch1[
                'weight']
            maskz = maskz[0:labeled_bs].cuda()
            maskzz = torch.unsqueeze(maskz, 1).cuda()
            maskzz = maskzz.repeat(1, 2, 1, 1, 1).cuda()
            volume_batch, label_batch = volume_batch.cuda(), label_batch.cuda()

            unlabeled_volume_batch = volume_batch[labeled_bs:]

            noise = torch.clamp(torch.randn_like(unlabeled_volume_batch) * 0.1, -0.2, 0.2)
            cs_inputs = unlabeled_volume_batch  # + noise
            outputs = m_seg1(volume_batch)
            with torch.no_grad():
                cs_output = m_seg2(cs_inputs)
            T = 8
            volume_batch_r = unlabeled_volume_batch.repeat(2, 1, 1, 1, 1)
            stride = volume_batch_r.shape[0] // 2
            preds = torch.zeros([stride * T, 2, patch_size[0], patch_size[0], patch_size[2]]).cuda()
            for i in range(T // 2):
                cs_inputs = volume_batch_r + torch.clamp(torch.randn_like(volume_batch_r) * 0.1, -0.2, 0.2)
                with torch.no_grad():
                    preds[2 * stride * i:2 * stride * (i + 1)] = m_seg2(cs_inputs)
            preds = F.softmax(preds, dim=1)
            preds = preds.reshape(T, stride, 2, patch_size[0], patch_size[0], patch_size[2])
            preds = torch.mean(preds, dim=0)
            uncertainty = -1.0 * torch.sum(preds * torch.log(preds + 1e-6), dim=1,
                                           keepdim=True)
            loss_seg = losses.wce(outputs[:labeled_bs], label_batch[:labeled_bs], maskzz, args.labeled_bs, patch_size[0], patch_size[0],
                                  patch_size[2])
            outputs_soft = F.softmax(outputs, dim=1)
            cs_output_soft=F.softmax(cs_output,dim=1)
            cs_pseudo_seg = torch.argmax(cs_output_soft.detach(), dim=1, keepdim=False)
            loss_seg_dice = losses.dice_loss_weight(outputs_soft[:labeled_bs, 1, :, :, :],
                                                    label_batch[:labeled_bs] == 1, maskz)
            supervised_loss = 0.5 * (loss_seg + loss_seg_dice)

            consistency_weight = get_current_consistency_weight(iter_num // 150)
            threshold = (0.75 + 0.25 * ramps.sigmoid_rampup(iter_num, max_iterations)) * np.log(2)
            mask = (uncertainty < threshold).float()
            mask=mask.repeat(1, 2, 1, 1, 1)

            consistency_dist = losses.wce(outputs[labeled_bs:], cs_pseudo_seg, mask, args.labeled_bs,
                                     patch_size[0], patch_size[0],
                                     patch_size[2])

            consistency_loss = consistency_weight * consistency_dist
            loss = (1 - consistency_weight) * supervised_loss + consistency_loss

            optimizer1.zero_grad()
            loss.backward()
            optimizer1.step()

            # model2

            volume_batch, label_batch, masky = sampled_batch2['image'], sampled_batch2['label'], sampled_batch2[
                'weight']
            volume_batch, label_batch = volume_batch.cuda(), label_batch.cuda()
            unlabeled_volume_batch = volume_batch[labeled_bs:]
            masky = masky[0:labeled_bs].cuda()
            maskyy = torch.unsqueeze(masky, 1).cuda()
            maskyy = maskyy.repeat(1, 2, 1, 1, 1).cuda()
            noise = torch.clamp(torch.randn_like(unlabeled_volume_batch) * 0.1, -0.2, 0.2)
            cs_inputs = unlabeled_volume_batch  # + noise
            outputs = m_seg2(volume_batch)
            with torch.no_grad():
                cs_output = m_seg1(cs_inputs)
            T = 8
            volume_batch_r = unlabeled_volume_batch.repeat(2, 1, 1, 1, 1)
            stride = volume_batch_r.shape[0] // 2
            preds = torch.zeros([stride * T, 2, patch_size[0], patch_size[0], patch_size[2]]).cuda()
            for i in range(T // 2):
                cs_inputs = volume_batch_r + torch.clamp(torch.randn_like(volume_batch_r) * 0.1, -0.2, 0.2)
                with torch.no_grad():
                    preds[2 * stride * i:2 * stride * (i + 1)] = m_seg1(cs_inputs)
            preds = F.softmax(preds, dim=1)
            preds = preds.reshape(T, stride, 2, patch_size[0], patch_size[0], patch_size[2])
            preds = torch.mean(preds, dim=0)
            uncertainty = -1.0 * torch.sum(preds * torch.log(preds + 1e-6), dim=1,
                                           keepdim=True)
            loss_seg = losses.wce(outputs[:labeled_bs], label_batch[:labeled_bs], maskyy, args.labeled_bs, patch_size[0], patch_size[0],
                                  patch_size[2])
            outputs_soft = F.softmax(outputs, dim=1)
            cs_output_soft = F.softmax(cs_output, dim=1)
            cs_pseudo_seg = torch.argmax(cs_output_soft.detach(), dim=1, keepdim=False)
            loss_seg_dice = losses.dice_loss_weight(outputs_soft[:labeled_bs, 1, :, :, :],
                                                    label_batch[:labeled_bs] == 1, masky)
            supervised_loss = 0.5 * (loss_seg + loss_seg_dice)

            consistency_weight = get_current_consistency_weight(iter_num // 150)

            threshold = (0.75 + 0.25 * ramps.sigmoid_rampup(iter_num, max_iterations)) * np.log(2)
            mask = (uncertainty < threshold).float()
            mask = mask.repeat(1, 2, 1, 1, 1)
            consistency_dist = losses.wce(outputs[labeled_bs:], cs_pseudo_seg, mask, args.labeled_bs,
                                          patch_size[0], patch_size[0],
                                          patch_size[2])
            consistency_loss = consistency_weight * consistency_dist
            loss2 = (1 - consistency_weight) * supervised_loss + consistency_loss

            optimizer2.zero_grad()
            loss2.backward()
            optimizer2.step()

            iter_num = iter_num + 1
            writer.add_scalar('uncertainty/mask_per', torch.sum(mask) / mask.numel(), iter_num)
            writer.add_scalar('uncertainty/threshold', threshold, iter_num)
            writer.add_scalar('lr', lr_, iter_num)
            writer.add_scalar('loss/loss', loss, iter_num)
            writer.add_scalar('loss/loss_seg', loss_seg, iter_num)
            writer.add_scalar('loss/loss_seg_dice', loss_seg_dice, iter_num)
            writer.add_scalar('train/consistency_loss', consistency_loss, iter_num)
            writer.add_scalar('train/consistency_weight', consistency_weight, iter_num)
            writer.add_scalar('train/consistency_dist', consistency_dist, iter_num)

            logging.info('iteration %d : loss : %f,%f cons_dist: %f, loss_weight: %f' %
                         (iter_num, loss.item(), loss2.item(), consistency_dist.item(), consistency_weight))
            if iter_num % 50 == 0:
                image = volume_batch[0, 0:1, :, :, 20:61:10].permute(3, 0, 1, 2).repeat(1, 3, 1, 1)
                grid_image = make_grid(image, 5, normalize=True)
                writer.add_image('train/Image', grid_image, iter_num)

                # image = outputs_soft[0, 3:4, :, :, 20:61:10].permute(3, 0, 1, 2).repeat(1, 3, 1, 1)
                image = torch.max(outputs_soft[0, :, :, :, 20:61:10], 0)[1].permute(2, 0, 1).data.cpu().numpy()
                image = utils.decode_seg_map_sequence(image)
                grid_image = make_grid(image, 5, normalize=False)
                writer.add_image('train/Predicted_label', grid_image, iter_num)

                image = label_batch[0, :, :, 20:61:10].permute(2, 0, 1)
                grid_image = make_grid(utils.decode_seg_map_sequence(image.data.cpu().numpy()), 5, normalize=False)
                writer.add_image('train/Groundtruth_label', grid_image, iter_num)

                image = volume_batch[-1, 0:1, :, :, 20:61:10].permute(3, 0, 1, 2).repeat(1, 3, 1, 1)
                grid_image = make_grid(image, 5, normalize=True)
                writer.add_image('unlabel/Image', grid_image, iter_num)

                # image = outputs_soft[-1, 3:4, :, :, 20:61:10].permute(3, 0, 1, 2).repeat(1, 3, 1, 1)
                image = torch.max(outputs_soft[-1, :, :, :, 20:61:10], 0)[1].permute(2, 0, 1).data.cpu().numpy()
                image = utils.decode_seg_map_sequence(image)
                grid_image = make_grid(image, 5, normalize=False)
                writer.add_image('unlabel/Predicted_label', grid_image, iter_num)

                image = label_batch[-1, :, :, 20:61:10].permute(2, 0, 1)
                grid_image = make_grid(utils.decode_seg_map_sequence(image.data.cpu().numpy()), 5, normalize=False)
                writer.add_image('unlabel/Groundtruth_label', grid_image, iter_num)


            ## change lr
            if iter_num % 2500 == 0:
                lr_ = base_lr * 0.1 ** (iter_num // 2500)
                for param_group in optimizer1.param_groups:
                    param_group['lr'] = lr_
            if iter_num % 2000 == 0:
                save_mode_path = os.path.join(snapshot_path, 'iter_' + str(iter_num) + '.pth')
                save_mode_path2 = os.path.join(snapshot_path, 'iter_' + str(iter_num) + '2.pth')
                torch.save(m_seg1.state_dict(), save_mode_path)
                torch.save(m_seg2.state_dict(), save_mode_path2)
                logging.info("save model to {}".format(save_mode_path))
            if iter_num >= max_iterations:
                break
            if iter_num % 1000 == 0:
                if args.dataset == "la":
                    db_train1 = LAHeart1(base_dir=train_data_path,
                                         split=args.split,
                                         transform=transforms.Compose([
                                             LARandomCrop1(patch_size, get_current_slice_weight(iter_num // args.slice_weight_step)),
                                             LARandomRotFlip(),
                                             ToTensor(),
                                         ]))
                    db_train2 = LAHeart2(base_dir=train_data_path,
                                         split=args.split,
                                         transform=transforms.Compose([
                                             LARandomCrop2(patch_size, get_current_slice_weight(iter_num // args.slice_weight_step)),
                                             LARandomRotFlip(),
                                             ToTensor(),
                                         ]))
                    labeled_idxs = list(range(5))
                    unlabeled_idxs = list(range(5, 80))
                elif args.dataset == 'kits_kidney':
                    db_train1 = KiTS1(base_dir=train_data_path,
                                      split=args.split,
                                      transform=transforms.Compose([
                                          KRandomCrop1(patch_size, get_current_slice_weight(iter_num // args.slice_weight_step)),
                                          KRandomRotFlip(),
                                          ToTensor(),
                                      ]))
                    db_train2 = KiTS2(base_dir=train_data_path,
                                      split=args.split,
                                      transform=transforms.Compose([
                                          KRandomCrop2(patch_size, get_current_slice_weight(iter_num // args.slice_weight_step)),
                                          KRandomRotFlip(),
                                          ToTensor(),
                                      ]))
                    labeled_idxs = list(range(5))
                    unlabeled_idxs = list(range(5, 190))
                elif args.dataset == 'lits_liver':
                    db_train1 = LiTS1(base_dir=train_data_path,
                                      split=args.split,
                                      transform=transforms.Compose([
                                          LRandomCrop1(patch_size, get_current_slice_weight(iter_num // args.slice_weight_step)),
                                          LRandomRotFlip(),
                                          ToTensor(),
                                      ]))
                    db_train2 = LiTS2(base_dir=train_data_path,
                                      split=args.split,
                                      transform=transforms.Compose([
                                          LRandomCrop2(patch_size, get_current_slice_weight(iter_num // args.slice_weight_step)),
                                          LRandomRotFlip(),
                                          ToTensor(),
                                      ]))

                    labeled_idxs = list(range(5))
                    unlabeled_idxs = list(range(5, 100))

                batch_sampler = TwoStreamBatchSampler(labeled_idxs, unlabeled_idxs, batch_size,
                                                      batch_size - labeled_bs)
                trainloader1 = DataLoader(db_train1, batch_sampler=batch_sampler, num_workers=4, pin_memory=True,
                                          worker_init_fn=worker_init_fn)
                trainloader2 = DataLoader(db_train2, batch_sampler=batch_sampler, num_workers=4, pin_memory=True,
                                          worker_init_fn=worker_init_fn)
                break
            time1 = time.time()
        if iter_num >= max_iterations:
            break
    save_mode_path = os.path.join(snapshot_path, 'iter_' + str(max_iterations) + '.pth')
    torch.save(m_seg1.state_dict(), save_mode_path)
    logging.info("save model to {}".format(save_mode_path))
    writer.close()