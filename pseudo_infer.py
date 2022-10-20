import datetime
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "2"
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
from dataset.VShadow_crosspairwise import CrossPairwiseImg, RCRDataset
from misc import AvgMeter, check_mkdir, crf_refine
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
from networks.STANet import OurModel
# from networks.BDRAR_plus import OurModel
# from networks.TDAN import OurModel

from flownet2.models import FlowNet2
from gma_core.gma_network import RAFTGMA
import argparse
import cv2
import glob

cudnn.deterministic = True
cudnn.benchmark = True

ckpt_path = './models'

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
    'max_epoch': 15,
    'train_batch_size': 1,
    'last_iter': 0,
    'finetune_lr': 5e-5,
    'scratch_lr': 5e-4,
    'weight_decay': 5e-4,
    'momentum': 0.9,
    'snapshot': '',
    'scale': 400,
    'multi-scale': None,
    'gpu': '0',
    'multi-GPUs': False,
    'fp16': True,
    'warm_up_epochs': 4,
    'seed': 2022
}
# fix random seed
np.random.seed(args['seed'])
torch.manual_seed(args['seed'])
torch.cuda.manual_seed(args['seed'])

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


print("max epoch:{}".format(args['max_epoch']))

# bce_loss = nn.BCELoss().cuda()
bce_loss = binary_xloss


def main():
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
            {'params': [param for name, param in net.module.named_parameters() if name[-4:] == 'bias'],
             'lr': 2 * args['finetune_lr']},
            {'params': [param for name, param in net.module.named_parameters() if name[-4:] != 'bias'],
             'lr': args['finetune_lr']}
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

    # for p in gma.module.parameters():
    #     p.requires_grad = False
    # optimizer = optim.Adam(params, betas=(0.9, 0.99), eps=6e-8, weight_decay=args['weight_decay'])
    # if args['fp16']:
    #     net, optimizer = amp.initialize(net, optimizer, opt_level="O1")
        # flownet.load_state_dict(
        #     torch.load('/mnt/data1/fxy/01RCRNet-Pytorch/models/FlowNet2_checkpoint.pth.tar')['state_dict'])
    gma.load_state_dict(torch.load(opt.model))
    gma.eval()
    # net.load_state_dict(torch.load('/mnt/data1/fxy/ViSha-main/models/5.pth')['model'])
    # net.load_state_dict(torch.load('/mnt/data1/fxy/ViSha-main/models/BDRAR_plus_RCRNet14/9.pth')['model'])
    # net.load_state_dict(torch.load('/mnt/data1/fxy/ViSha-main/models/BDRAR_plus_RCRNet12/3.pth')['model'])
    net.load_state_dict(torch.load('')['model'])
    infer(net, gma)


def cal_BER(prediction, label, thr=127.5):
    prediction = (prediction > thr)
    label = (label > thr)
    prediction_tmp = prediction.astype(np.float)
    label_tmp = label.astype(np.float)
    TP = np.sum(prediction_tmp * label_tmp)
    TN = np.sum((1 - prediction_tmp) * (1 - label_tmp))
    Np = np.sum(label_tmp)
    Nn = np.sum((1 - label_tmp))
    BER = 0.5 * (2 - TP / Np - TN / Nn) * 100
    # shadow_BER = (1 - TP / Np) * 100
    # non_shadow_BER = (1 - TN / Nn) * 100

    return BER


def resize_flow(flow, size):
    origin_size = flow.shape[2:]
    flow = F.interpolate(flow, size=size, mode="nearest")
    flow[:, 0, :, :] /= origin_size[1] / size[1]  # dx
    flow[:, 1, :, :] /= origin_size[0] / size[0]  # dy
    return flow


def getAdjacentIndex(current_index, start_index, video_length):
    query_index_list = [current_index, current_index - 1, current_index + 1]
    if current_index == start_index:
        query_index_list = [current_index, current_index+1, current_index+2]
    if current_index == video_length-1:
        query_index_list = [current_index, current_index-1, current_index-2]
    return query_index_list


def infer(net, gma):
    data_root = '/mnt/data4/czpp/ourData2/'
    # data_root = '/mnt/data1/fxy/0Datasets/myDataPress/JPEGImages'
    result_root = '/mnt/data4/czpp/ourData2_res/MSANet14'
    # result_root = '/mnt/data4/czpp/ourData_res/MSANet10'

    check_mkdir(result_root)

    path1 = []
    path2 = []

    for s in os.listdir(data_root):
        path1.append(os.path.join(data_root, s))
        path2.append(os.path.join(result_root, s))

    curr_epoch = 1
    curr_iter = 1
    start = 0
    scaler = GradScaler()
    # res_path = '/mnt/data1/czpp/feature_visual/'
    # res_path = '/mnt/data4/czpp/ViSha_result/MPLCNet-original/'
    # check_mkdir(res_path)
    print('=====>Start training<======')
    # validate
    # net.train()
    net.eval()
    # for m in net.modules():
    #     if m.__class__.__name__.startswith('Dropout'):
    #         m.train()
    with torch.no_grad():
        running_ber = 0.0
        sample_num = 0
        for i in range(len(path1)):
            input_path = path1[i]
            inputs = []
            for img_path in sorted(glob.glob(os.path.join(input_path, '*.jpg'))):
                inputs.append(img_path)
            check_mkdir(path2[i])
            print(input_path)

            for idx, image_name in enumerate(inputs):
                query_idx = getAdjacentIndex(idx, 0, len(inputs))
                exemplar, ref1, ref2 = inputs[query_idx[0]], inputs[query_idx[1]], inputs[query_idx[2]]
                exemplar = Image.open(exemplar).convert('RGB')
                ref1 = Image.open(ref1).convert('RGB')
                ref2 = Image.open(ref2).convert('RGB')
                w, h = exemplar.size

                # H_o = int(math.ceil(float(h) / 4) * 4)
                # W_o = int(math.ceil(float(w) / 4) * 4)
                # H_f = int(math.ceil(float(h) / 64) * 64)
                # W_f = int(math.ceil(float(w) / 64) * 64)
                #
                # exemplar = transforms.Resize((W_o, H_o))(exemplar)
                # ref1 = transforms.Resize((W_o, H_o))(ref1)
                # ref2 = transforms.Resize((W_o, H_o))(ref2)
                # exemplar_resize = transforms.Resize([W_f, H_f])(exemplar)
                # ref1_resize = transforms.Resize([W_f, H_f])(ref1)
                # ref2_resize = transforms.Resize([W_f, H_f])(ref2)

                exemplar = transforms.Resize((args['scale'], args['scale']))(exemplar)
                ref1 = transforms.Resize((args['scale'], args['scale']))(ref1)
                ref2 = transforms.Resize((args['scale'], args['scale']))(ref2)
                exemplar_resize = transforms.Resize([384, 384])(exemplar)
                ref1_resize = transforms.Resize([384, 384])(ref1)
                ref2_resize = transforms.Resize([384, 384])(ref2)

                exemplar = img_transform(exemplar).unsqueeze(0).cuda()
                ref1 = img_transform(ref1).unsqueeze(0).cuda()
                ref2 = img_transform(ref2).unsqueeze(0).cuda()

                exemplar_resize = transforms.ToTensor()(exemplar_resize).unsqueeze(0).cuda() * 255.0
                ref1_resize = transforms.ToTensor()(ref1_resize).unsqueeze(0).cuda() * 255.0
                ref2_resize = transforms.ToTensor()(ref2_resize).unsqueeze(0).cuda() * 255.0

                _, flow_ref1_example = gma(exemplar_resize, ref1_resize, iters=6, test_mode=True)
                _, flow_ref2_example = gma(exemplar_resize, ref2_resize, iters=6, test_mode=True)

                flow1_ref1 = resize_flow(flow_ref1_example, [25, 25])
                flow1_ref2 = resize_flow(flow_ref2_example, [25, 25])

                net.eval()
                preds = net([exemplar, ref1, ref2], flow1_ref1, flow1_ref2, False)

                res = (preds.detach().cpu() > 0).to(torch.float32)
                prediction = np.array(transforms.Resize((h, w))(to_pil(res.squeeze(0).cpu()))).astype(np.uint8)

                Image.fromarray(prediction).save(path2[i] + '/' + os.path.basename(image_name)[:-4] + '.png')

                # uncertainty
                net.train()
                prediction_sequence = []
                for i in range(10):
                    preds = net([exemplar, ref1, ref2], flow1_ref1, flow1_ref2, False)
                    preds = F.sigmoid(preds)
                    prediction_sequence.append((preds.detach().cpu()).to(torch.float32))
                uncertainty = torch.stack(prediction_sequence, dim=0).numpy()
                uncertainty = np.std(uncertainty, axis=0) * 100
                print(uncertainty.mean())
                Image.fromarray((uncertainty.squeeze() * 255.0).astype(np.uint8)).save(path2[i] + '/' + os.path.basename(image_name)[:-4] + '_uncertain' + '.png')


if __name__ == '__main__':
    main()