import datetime
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
import sys
sys.path.append('gma_core')

import torch
from torch import nn
from torch import optim
from torch.backends import cudnn
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm

import joint_transforms
from config import ViSha_training_root
from config import ViSha_validation_root
from dataset.VShadow_crosspairwise import CrossPairwiseImg, VSD_Dataset
from misc import AvgMeter, check_mkdir
from networks.TVSD import TVSD
from torch.optim.lr_scheduler import StepLR
import math
from losses import lovasz_hinge, binary_xloss
import random
import torch.nn.functional as F
import numpy as np
# from apex import amp
from torch.cuda import amp
from torch.cuda.amp import autocast as autocast
from torch.cuda.amp import GradScaler as GradScaler

import time
import pdb
from PIL import Image

from torch.utils.tensorboard import SummaryWriter
from RCRNet import VideoModel
from RCRNet import get_datasets
# from networks.OurModel5 import OurModel
from networks.STANet import OurModel
# from networks.TDAN import OurModel

# from flownet2.models import FlowNet2
from gma_core.gma_network import RAFTGMA
import argparse
from torch.autograd import Variable
# from flownet2.networks.resample2d_package.resample2d import Resample2d

cudnn.deterministic = True
cudnn.benchmark = True

ckpt_path = './models'
exp_name = 'BDRAR_plus_RCRNet_plus1'

parser = argparse.ArgumentParser()
# flownet2 parameters
parser.add_argument('--rgb_max', type=float, default=1.0)
parser.add_argument('--fp16', type=bool, default=False)
# gma parameters
parser.add_argument('--model', default="/mnt/data1/czpp/GMA-main/checkpoints/gma-sintel.pth", help="restore checkpoint")
parser.add_argument('--dataset', default="sintel", help="dataset for evaluation")
parser.add_argument('--iters', type=int, default=12)
parser.add_argument('--num_heads', default=1, type=int,
                    help='number of heads in attention and aggregation')
parser.add_argument('--position_only', default=False, action='store_true',
                    help='only use position-wise attention')
parser.add_argument('--position_and_content', default=False, action='store_true',
                    help='use position and content-wise attention')
parser.add_argument('--mixed_precision', default=True, help='use mixed precision')
parser.add_argument('--model_name')

# Ablations
parser.add_argument('--replace', default=False, action='store_true',
                    help='Replace local motion feature with aggregated motion features')
parser.add_argument('--no_alpha', default=False, action='store_true',
                    help='Remove learned alpha, set it to 1')
parser.add_argument('--no_residual', default=False, action='store_true',
                    help='Remove residual connection. Do not add local features with the aggregated features.')
opt = parser.parse_args()

args = {
    'max_epoch': 10,
    'train_batch_size': 7,
    'last_iter': 0,
    'finetune_lr': 5e-4,
    'scratch_lr': 5e-4,
    'weight_decay': 5e-4,
    'momentum': 0.9,
    'lr_decay': 0.9,
    'snapshot': '',
    'scale': 400,
    'multi-scale': None,
    'gpu': '1',
    'multi-GPUs': False,
    'fp16': True,
    'warm_up_epochs': 4,
    'seed': 2022
}
# fix random seed
np.random.seed(args['seed'])
torch.manual_seed(args['seed'])
torch.cuda.manual_seed(args['seed'])

writer = SummaryWriter(log_dir='./bdrar_plus_rcrnet_plus1')

# multi-GPUs training
if args['multi-GPUs']:
    os.environ['CUDA_VISIBLE_DEVICES'] = args['gpu']
    batch_size = args['train_batch_size'] * len(args['gpu'].split(','))
# single-GPU training
else:
    # torch.cuda.set_device(0)
    batch_size = args['train_batch_size']

joint_transform = joint_transforms.Compose([
    joint_transforms.RandomHorizontallyFlip(),
    joint_transforms.Resize((args['scale'], args['scale']))
])
val_joint_transform = joint_transforms.Compose([
    joint_transforms.Resize((args['scale'], args['scale']))
])
img_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])
test_transforms = transforms.Compose([
    transforms.Resize((args['scale'], args['scale'])),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])
target_transform = transforms.ToTensor()
to_pil = transforms.ToPILImage()

print('=====>Dataset loading<======')
training_root = [ViSha_training_root] # training_root should be a list form, like [datasetA, datasetB, datasetC], here we use only one dataset.
train_set = VSD_Dataset(training_root, joint_transform, img_transform, target_transform, )
train_loader = DataLoader(train_set, batch_size=batch_size, num_workers=8, shuffle=True)


print('=====>Validation Dataset loading<======')
validation_root = [ViSha_validation_root]
validation_set = VSD_Dataset(validation_root, val_joint_transform, img_transform, target_transform=target_transform, is_train=False)
val_loader = DataLoader(validation_set, batch_size=4, num_workers=8, shuffle=False)

print("max epoch:{}".format(args['max_epoch']))

# bce_loss = nn.BCELoss().cuda()
# bce_loss = binary_xloss
binary_xloss = nn.BCEWithLogitsLoss().cuda()

l1_loss = nn.L1Loss().cuda()

# flow_warping = Resample2d().cuda()

log_path = os.path.join(ckpt_path, exp_name, str(datetime.datetime.now()) + '.txt')


# val_dataset = get_datasets(
#     # name_list="CUHKshadow-Video",
#     name_list="ViSha-RCR",
#     split_list="test",
#     config_path='./RCRNet/datasets.yaml',
#     root='/mnt/data1/fxy/0Datasets',
#     # root='/mnt/data4/czpp/Video_datasets',
#     training=False,
#     transforms=test_transforms,
#     read_clip=True,
#     random_reverse_clip=False,
#     clip_len=3
# )
# val_dataloader = DataLoader(dataset=val_dataset, batch_size=batch_size, num_workers=8, shuffle=False, drop_last=True)


def main():
    print('=====>Prepare Network {}<======'.format(exp_name))
    # multi-GPUs training
    if args['multi-GPUs']:
        net = torch.nn.DataParallel(OurModel()).cuda().train()
        # flownet = torch.nn.DataParallel(FlowNet2(opt, requires_grad=False)).cuda()
        gma = torch.nn.DataParallel(RAFTGMA(opt)).cuda()
        # params = [
        #     {'params': net.module.pre_stage_network.parameters(), 'lr': 5e-5},
        #     {'params': net.module.first_conv.parameters(), 'lr': args['scratch_lr']},
        #     {'params': net.module.feature_extractor.parameters(), 'lr': args['finetune_lr']},
        #     {'params': net.module.aspp.parameters(), 'lr': args['scratch_lr']},
        #     {'params': net.module.bottlneck1.parameters(), 'lr': args['scratch_lr']},
        #     {'params': net.module.off2d_1.parameters(), 'lr': args['scratch_lr']},
        #     # {'params': net.module.off2d_11.parameters(), 'lr': args['scratch_lr']},
        #     {'params': net.module.deconv_1.parameters(), 'lr': args['scratch_lr']},
        #     {'params': net.module.off2d_2.parameters(), 'lr': args['scratch_lr']},
        #     # {'params': net.module.off2d_22.parameters(), 'lr': args['scratch_lr']},
        #     {'params': net.module.deconv_2.parameters(), 'lr': args['scratch_lr']},
        #     {'params': net.module.off2d_3.parameters(), 'lr': args['scratch_lr']},
        #     # {'params': net.module.off2d_33.parameters(), 'lr': args['scratch_lr']},
        #     {'params': net.module.deconv_3.parameters(), 'lr': args['scratch_lr']},
        #     {'params': net.module.off2d_4.parameters(), 'lr': args['scratch_lr']},
        #     {'params': net.module.deconv_4.parameters(), 'lr': args['scratch_lr']},
        #     {'params': net.module.skip1.parameters(), 'lr': args['scratch_lr']},
        #     {'params': net.module.deconv1.parameters(), 'lr': args['scratch_lr']},
        #     {'params': net.module.skip2.parameters(), 'lr': args['scratch_lr']},
        #     {'params': net.module.deconv2.parameters(), 'lr': args['scratch_lr']},
        #     {'params': net.module.skip3.parameters(), 'lr': args['scratch_lr']},
        #     {'params': net.module.deconv3.parameters(), 'lr': args['scratch_lr']},
        #     {'params': net.module.deconv4.parameters(), 'lr': args['scratch_lr']},
        #     {'params': net.module.pred.parameters(), 'lr': args['scratch_lr']}
        # ]
        params = [
                {'params': [param for name, param in net.module.named_parameters() if name[-4:] == 'bias'], 'lr': 2 * args['finetune_lr']},
                {'params': [param for name, param in net.module.named_parameters() if name[-4:] != 'bias'], 'lr': args['finetune_lr']}
        ]
    # single-GPU training
    else:
        net = OurModel().cuda().train()
        # flownet = FlowNet2(opt, requires_grad=False).cuda()
        gma = torch.nn.DataParallel(RAFTGMA(opt)).cuda()
        # params = [
        #     {'params': net.pre_stage_network.parameters(), 'lr': 5e-5},
        #     {'params': net.first_conv.parameters(), 'lr': args['scratch_lr']},
        #     {'params': net.feature_extractor.parameters(), 'lr': args['finetune_lr']},
        #     {'params': net.aspp.parameters(), 'lr': args['scratch_lr']},
        #     {'params': net.bottlneck1.parameters(), 'lr': args['scratch_lr']},
        #     {'params': net.off2d_1.parameters(), 'lr': args['scratch_lr']},
        #     # {'params': net.off2d_11.parameters(), 'lr': args['scratch_lr']},
        #     {'params': net.deconv_1.parameters(), 'lr': args['scratch_lr']},
        #     {'params': net.off2d_2.parameters(), 'lr': args['scratch_lr']},
        #     # {'params': net.off2d_22.parameters(), 'lr': args['scratch_lr']},
        #     {'params': net.deconv_2.parameters(), 'lr': args['scratch_lr']},
        #     {'params': net.off2d_3.parameters(), 'lr': args['scratch_lr']},
        #     # {'params': net.off2d_33.parameters(), 'lr': args['scratch_lr']},
        #     {'params': net.deconv_3.parameters(), 'lr': args['scratch_lr']},
        #     {'params': net.off2d_4.parameters(), 'lr': args['scratch_lr']},
        #     {'params': net.deconv_4.parameters(), 'lr': args['scratch_lr']},
        #     {'params': net.skip1.parameters(), 'lr': args['scratch_lr']},
        #     {'params': net.deconv1.parameters(), 'lr': args['scratch_lr']},
        #     {'params': net.skip2.parameters(), 'lr': args['scratch_lr']},
        #     {'params': net.deconv2.parameters(), 'lr': args['scratch_lr']},
        #     {'params': net.skip3.parameters(), 'lr': args['scratch_lr']},
        #     {'params': net.deconv3.parameters(), 'lr': args['scratch_lr']},
        #     {'params': net.deconv4.parameters(), 'lr': args['scratch_lr']},
        #     {'params': net.pred.parameters(), 'lr': args['scratch_lr']}
        # ]
        params = [
            {'params': [param for name, param in net.named_parameters() if name[-4:] == 'bias'],
             'lr': 2 * args['finetune_lr']},
            {'params': [param for name, param in net.named_parameters() if name[-4:] != 'bias'],
             'lr': args['finetune_lr']}
        ]

    for p in gma.module.parameters():
        p.requires_grad = False
    optimizer = optim.SGD(params, momentum=args['momentum'], weight_decay=args['weight_decay'])
    # optimizer = optim.Adam(params, betas=(0.9, 0.99), eps=6e-8, weight_decay=args['weight_decay'])
    # warm_up_with_cosine_lr = lambda epoch: epoch / (args['warm_up_epochs']) if epoch <= args['warm_up_epochs'] else 0.5 * \
    #                          (math.cos((epoch-args['warm_up_epochs'])/(args['max_epoch']-args['warm_up_epochs'])*math.pi)+1)
    # warm_up_with_cosine_lr = lambda epoch: 0.5 * (math.cos((epoch - args['warm_up_epochs']) / (
    #                                        args['max_epoch'] - args['warm_up_epochs']) * math.pi) + 1)
    # scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=warm_up_with_cosine_lr)
    # scheduler = StepLR(optimizer, step_size=10, gamma=0.1)  # change learning rate after 20000 iters

    check_mkdir(ckpt_path)
    check_mkdir(os.path.join(ckpt_path, exp_name))
    open(log_path, 'w').write(str(args) + '\n\n')
    # if args['fp16']:
    #     net, optimizer = amp.initialize(net, optimizer, opt_level="O1")
    if args['multi-GPUs']:
        net.module.pre_stage_network.load_state_dict(torch.load('/mnt/data1/czpp/BDRAR-master/ckpt/BDRAR/3001.pth'))
        # flownet.module.load_state_dict(
        #     torch.load('/mnt/data1/fxy/01RCRNet-Pytorch/models/FlowNet2_checkpoint.pth.tar')['state_dict'])
    else:
        net.pre_stage_network.load_state_dict(torch.load('/mnt/data1/czpp/BDRAR-master/ckpt/BDRAR/3001.pth'))
        # flownet.load_state_dict(
        #     torch.load('/mnt/data1/fxy/01RCRNet-Pytorch/models/FlowNet2_checkpoint.pth.tar')['state_dict'])
    gma.load_state_dict(torch.load(opt.model))
    gma.eval()
    # net.load_state_dict(torch.load('/mnt/data1/fxy/ViSha-main/models/BDRAR_plus_RCRNet4/3.pth'))
    # net.load_state_dict(torch.load('/mnt/data1/fxy/ViSha-main/models/BDRAR_plus_RCRNet13/3.pth')['model'])

    # train(net, optimizer, scheduler, gma)
    train(net, optimizer, gma)


def cal_BER(prediction, label, thr = 127.5):
    prediction = (prediction > thr)
    label = (label > thr)
    prediction_tmp = prediction.astype(np.float)
    label_tmp = label.astype(np.float)
    TP = np.sum(prediction_tmp * label_tmp)
    TN = np.sum((1 - prediction_tmp) * (1 - label_tmp))
    Np = np.sum(label_tmp)
    Nn = np.sum((1-label_tmp))
    BER = 0.5 * (2 - TP / Np - TN / Nn) * 100
    # shadow_BER = (1 - TP / Np) * 100
    # non_shadow_BER = (1 - TN / Nn) * 100

    return BER


def resize_flow(flow, size):
    origin_size = flow.shape[2:]
    flow = F.interpolate(flow, size=size, mode="nearest")
    flow[:, 0, :, :] /= origin_size[1] / size[1] # dx
    flow[:, 1, :, :] /= origin_size[0] / size[0] # dy
    return flow


# def train(net, optimizer, scheduler, gma):
def train(net, optimizer, gma):
    curr_epoch = 1
    curr_iter = 1
    start = 0
    scaler = GradScaler()
    print('=====>Start training<======')
    max_iteration = args['max_epoch'] * len(train_loader)
    while True:
        loss_record1, loss_record2, loss_record3 = AvgMeter(), AvgMeter(), AvgMeter()
        net.train()
        for i, sample in enumerate(tqdm(train_loader, desc=f'Epoch: {curr_epoch}', ncols=100, ascii=' =', bar_format='{l_bar}{bar}|')):

            # pdb.set_trace()
            optimizer.param_groups[0]['lr'] = 2 * args['finetune_lr'] * (1 - float(curr_iter) / max_iteration
                                                                ) ** args['lr_decay']
            optimizer.param_groups[1]['lr'] = args['finetune_lr'] * (1 - float(curr_iter) / max_iteration
                                                            ) ** args['lr_decay']

            exemplar, exemplar_gt, ref1 = sample['exemplar'].cuda(), sample['exemplar_gt'].cuda(), sample['ref1'].cuda()
            # ref2 = sample['ref2'].cuda()
            ref2, exemplar_resize, ref1_resize, ref2_resize = sample['ref2'].cuda(), sample['exemplar_resize'].cuda(), \
                                                              sample['ref1_resize'].cuda(), sample['ref2_resize'].cuda()
            exemplar_resize = exemplar_resize * 255.0
            ref1_resize = ref1_resize * 255.0
            ref2_resize = ref2_resize * 255.0

            # ref1_gt, ref2_gt = sample['ref1_gt'].cuda(), sample['ref2_gt'].cuda()

            # b, c, h_ori, w_ori = ref1.shape
            # ref1_resize = transforms.Resize([384, 384])(ref1.clone())
            # ref2_resize = ref2.clone().resize(b, c, 384, 384)
            # example_resize = exemplar.clone().resize(b, c, 384, 384)
            # pdb.set_trace()

            # flow_ref1_example = flownet(exemplar_resize, ref1_resize)
            # flow_ref2_example = flownet(exemplar_resize, ref2_resize)

            _, flow_ref1_example = gma(exemplar_resize, ref1_resize, iters=6, test_mode=True)
            _, flow_ref2_example = gma(exemplar_resize, ref2_resize, iters=6, test_mode=True)

            flow1_ref1 = resize_flow(flow_ref1_example, [25, 25])
            flow1_ref2 = resize_flow(flow_ref2_example, [25, 25])

            # flow1_ref1 = resize_flow(flow_ref1_example, [48, 48])
            # flow1_ref2 = resize_flow(flow_ref2_example, [48, 48])
            # #
            # flow2_ref1 = resize_flow(flow_ref1_example, [96, 96])
            # flow2_ref2 = resize_flow(flow_ref2_example, [96, 96])

            # flow_ref1_example2 = resize_flow(flow_ref1_example, [50, 50])
            # flow_ref2_example2 = resize_flow(flow_ref2_example, [50, 50])

            optimizer.zero_grad()

            with autocast():
                # pdb.set_trace()

                # prediction = net([exemplar, ref1, ref2], flow_ref1_example1, flow_ref2_example1)
                # prediction, pred_exemple = net([exemplar, ref1, ref2], flow_ref1_example1, flow_ref2_example1)
                # prediction, pred_3, pred_2, pred_1 = net([exemplar, ref1, ref2], flow1_ref1, flow1_ref2, flow2_ref1, flow2_ref2)
                pre_prediction = net([exemplar, ref1, ref2], flow1_ref1, flow1_ref2)
                # prediction = net([exemplar, ref1, ref2], flow_ref1_example, flow_ref2_example)
                # prediction = net([exemplar, ref1, ref2])
                # prediction = net([exemplar, ref1, ref2], flow_ref1_example1, flow_ref2_example1, flow_ref1_example2,
                #                  flow_ref2_example2)

                # bce_loss1 = binary_xloss(prediction, exemplar_gt)
                # bce_loss2 = binary_xloss(prediction2, exemplar_gt)
                # bce_loss2 = binary_xloss(pred_3, exemplar_gt)
                # bce_loss3 = binary_xloss(pred_2, exemplar_gt)
                # bce_loss4 = binary_xloss(pred_1, exemplar_gt)

                # bce_loss2 = binary_xloss(pred_exemple, exemplar_gt)
                # bce_loss3 = binary_xloss(pred_ref1, ref1_gt)
                # bce_loss4 = binary_xloss(pred_ref2, ref2_gt)
                # loss_hinge1 = lovasz_hinge(prediction, exemplar_gt)
                # loss_hinge2 = lovasz_hinge(prediction2, exemplar_gt)
                # loss_hinge2 = lovasz_hinge(pred_exemple, exemplar_gt)


                # loss_hinge1 = lovasz_hinge(prediction, exemplar_gt)

                # bce_loss1 = (binary_xloss(pre_prediction[0], exemplar_gt) + binary_xloss(pre_prediction[1], exemplar_gt)
                #  + binary_xloss(pre_prediction[2], exemplar_gt) + binary_xloss(pre_prediction[3], exemplar_gt)
                #  + binary_xloss(pre_prediction[4], exemplar_gt) + binary_xloss(pre_prediction[5], exemplar_gt)
                #  + binary_xloss(pre_prediction[6], exemplar_gt) + binary_xloss(pre_prediction[7], exemplar_gt)
                #  + binary_xloss(pre_prediction[8], exemplar_gt)) / 9.0
                #
                # # addition_loss = l1_loss(pre_prediction[0], exemplar_gt)
                #
                # #
                # loss_hinge1 = (lovasz_hinge(pre_prediction[0], exemplar_gt) + lovasz_hinge(pre_prediction[1], exemplar_gt) +
                #                lovasz_hinge(pre_prediction[2], exemplar_gt) + lovasz_hinge(pre_prediction[3], exemplar_gt) +
                #                lovasz_hinge(pre_prediction[4], exemplar_gt) + lovasz_hinge(pre_prediction[5], exemplar_gt) +
                #                lovasz_hinge(pre_prediction[6], exemplar_gt) + lovasz_hinge(pre_prediction[7], exemplar_gt) +
                #                lovasz_hinge(pre_prediction[8], exemplar_gt)) / 9.0

                # bce_loss1 = (binary_xloss(pre_prediction[0], exemplar_gt) + binary_xloss(pre_prediction[1], exemplar_gt)
                #              + binary_xloss(pre_prediction[2], exemplar_gt)) / 3.0
                bce_loss1 = (l1_loss(pre_prediction[0], exemplar_gt) + l1_loss(pre_prediction[1], exemplar_gt)
                             + l1_loss(pre_prediction[2], exemplar_gt)) / 3.0

                # addition_loss = l1_loss(pre_prediction[0], exemplar_gt)

                #
                # loss_hinge1 = (lovasz_hinge(pre_prediction[0], exemplar_gt) + lovasz_hinge(pre_prediction[1],
                #                                                                            exemplar_gt) +
                #                lovasz_hinge(pre_prediction[2], exemplar_gt)) / 3.0

                # loss_seg = bce_loss1 + bce_loss2 + loss_hinge1 + loss_hinge2
                loss_seg = bce_loss1
                # loss_seg = bce_loss1 * 9.0
                # loss_seg = bce_loss1 + bce_loss2 + loss_hinge1 + loss_hinge2

                loss = loss_seg

            scaler.scale(loss).backward()

            scaler.unscale_(optimizer)

            torch.nn.utils.clip_grad_norm_(net.parameters(), max_norm=12)

            scaler.step(optimizer)

            scaler.update()

            # optimizer.step()  # change gradient

            loss_record1.update(bce_loss1.item(), batch_size)
            # loss_record2.update(bce_loss2.item(), batch_size)
            # loss_record3.update(loss_hinge1.item(), batch_size)

            curr_iter += 1

            log = "iter: %d, bce1: %f5, bce2: %f5, hinge1: %f5, lr: %f8"%\
                  (curr_iter, loss_record1.avg, loss_record2.avg, loss_record3.avg, optimizer.param_groups[0]['lr'])

            if (curr_iter-1) % 20 == 0:
                elapsed = (time.clock() - start)
                start = time.clock()
                log_time = log + ' [time {}]'.format(elapsed)
                print(log_time)
            open(log_path, 'a').write(log + '\n')
        writer.add_scalar('training_bce_loss', loss_record1.avg, curr_epoch)
        # writer.add_scalar('learning_rate', scheduler.get_lr()[0], curr_epoch)
        writer.add_scalar('learning_rate', optimizer.param_groups[0]['lr'], curr_epoch)
        if curr_epoch % 1 == 0:
            if args['multi-GPUs']:
                # torch.save(net.module.state_dict(), os.path.join(ckpt_path, exp_name, '%d.pth' % curr_epoch))
                checkpoint = {
                    'model': net.module.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    # 'amp': amp.state_dict()
                }
                torch.save(checkpoint, os.path.join(ckpt_path, exp_name, f'{curr_epoch}.pth'))
            else:
                # torch.save(net.state_dict(), os.path.join(ckpt_path, exp_name, '%d.pth' % curr_epoch))
                checkpoint = {
                    'model': net.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    # 'amp': amp.state_dict()
                }
                torch.save(checkpoint, os.path.join(ckpt_path, exp_name, f'{curr_epoch}.pth'))

        # validate
        if curr_epoch % 2 == 1:
            net.eval()
            with torch.no_grad():
                running_ber = 0.0
                running_mae = 0.0
                sample_num = 0
                for data in tqdm(val_loader):
                    # images = []
                    exemplar, exemplar_gt, ref1 = data['exemplar'].cuda(), data['exemplar_gt'].cuda(), data['ref1'].cuda()
                    ref2 = data['ref2'].cuda()
                    exemplar_resize, ref1_resize, ref2_resize = data['exemplar_resize'].cuda(), data['ref1_resize'].cuda(), data[
                                                                    'ref2_resize'].cuda()
                    exemplar_resize = exemplar_resize * 255.0
                    ref1_resize = ref1_resize * 255.0
                    ref2_resize = ref2_resize * 255.0

                    # flow_ref1_example = flownet(ref1_resize, exemplar_resize)
                    # flow_ref2_example = flownet(ref2_resize, exemplar_resize)

                    # flow_ref1_example1 = resize_flow(flow_ref1_example, [24, 24])
                    # flow_ref2_example1 = resize_flow(flow_ref2_example, [24, 24])

                    # flow_ref1_example2 = resize_flow(flow_ref1_example, [50, 50])
                    # flow_ref2_example2 = resize_flow(flow_ref2_example, [50, 50])
                    _, flow_ref1_example = gma(exemplar_resize, ref1_resize, iters=6, test_mode=True)
                    _, flow_ref2_example = gma(exemplar_resize, ref2_resize, iters=6, test_mode=True)

                    flow1_ref1 = resize_flow(flow_ref1_example, [25, 25])
                    flow1_ref2 = resize_flow(flow_ref2_example, [25, 25])
                    #
                    # flow1_ref1 = resize_flow(flow_ref1_example, [48, 48])
                    # flow1_ref2 = resize_flow(flow_ref2_example, [48, 48])
                    # #
                    # flow2_ref1 = resize_flow(flow_ref1_example, [96, 96])
                    # flow2_ref2 = resize_flow(flow_ref2_example, [96, 96])

                    h, w = data['height'], data['width']
                    # for frame in data:
                    #     images.append(frame['image'].cuda())
                        # labels.append(frame['label'].cuda())
                    preds = net([exemplar, ref1, ref2], flow1_ref1, flow1_ref2, False)
                    # preds = net([exemplar, ref1, ref2], flow_ref1_example, flow_ref2_example)
                    # preds = net([exemplar, ref1, ref2])
                    # preds, _, _, _ = net([exemplar, ref1, ref2], flow1_ref1, flow1_ref2, flow2_ref1, flow2_ref2)
                    # preds = net([exemplar, ref1, ref2], flow_ref1_example1, flow_ref2_example1, flow_ref1_example2,
                    #             flow_ref2_example2)
                    # pdb.set_trace()
                    # for i, pred_ in enumerate(preds):
                    #     for j, pred in enumerate(pred_.detach().cpu()):
                    #         h = data[i]['height'][j]
                    #         w = data[i]['width'][j]
                    res = (preds.detach().cpu() > 0).to(torch.float32)
                    # res = (F.sigmoid(preds.detach()).cpu()).to(torch.float32)
                    # prediction = np.array(
                    #             transforms.Resize((h, w))(to_pil(res.squeeze(0).cpu()))).astype(np.float32)
                    #         # pdb.set_trace()
                    # # label = np.array(Image.open(data[i]['label_path'][j]).convert('L')).astype(np.float32)
                    # label = np.array(transforms.Resize((h, w))(to_pil(exemplar_gt.squeeze(0).cpu()))).astype(np.float32)
                    # running_ber += cal_BER(prediction, label, 127.5)
                    # sample_num += 1
                    for i in range(4):
                        prediction = np.array(
                                    transforms.Resize((h[0], w[0]))(to_pil(res[i, :, :, :].squeeze(0).cpu()))).astype(np.float32)
                                # pdb.set_trace()
                        # label = np.array(Image.open(data[i]['label_path'][j]).convert('L')).astype(np.float32)
                        label = np.array(transforms.Resize((h[0], w[0]))(to_pil(exemplar_gt[i, :, :, :].squeeze(0).cpu()))).astype(np.float32)
                        running_ber += cal_BER(prediction, label, 127.5)
                        running_mae += np.mean(abs(prediction - label))
                        sample_num += 1
                running_ber = running_ber / sample_num
                running_mae = running_mae / sample_num
                print(running_ber)
                print(running_mae)
                writer.add_scalar('test_ber', running_ber, curr_epoch)

        if curr_epoch > args['max_epoch']:
            # torch.save(net.state_dict(), os.path.join(ckpt_path, exp_name, '%d.pth' % curr_iter))
            return

        curr_epoch += 1
        # scheduler.step()  # change learning rate after epoch
        
        


if __name__ == '__main__':
    main()