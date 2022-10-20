import datetime
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "4"
import sys

sys.path.append('gma_core')
sys.path.append('Pytorch_grad_cam')

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
from networks.OurModel5 import OurModel
# from networks.TDAN import OurModel

from flownet2.models import FlowNet2
from gma_core.gma_network import RAFTGMA
import argparse
import cv2
from Pytorch_grad_cam.pytorch_grad_cam.utils.image import show_cam_on_image, preprocess_image
from Pytorch_grad_cam.pytorch_grad_cam import GradCAM

cudnn.deterministic = True
cudnn.benchmark = True

ckpt_path = './models'
exp_name = 'BDRAR_plus_RCRNet20'

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
    'gpu': '5',
    'multi-GPUs': False,
    'fp16': True,
    'warm_up_epochs': 4,
    'seed': 2022
}
# fix random seed
np.random.seed(args['seed'])
torch.manual_seed(args['seed'])
torch.cuda.manual_seed(args['seed'])

writer = SummaryWriter(log_dir='./bdrar_plus_rcrnet20')

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
training_root = [
    ViSha_training_root]  # training_root should be a list form, like [datasetA, datasetB, datasetC], here we use only one dataset.
train_set = RCRDataset(training_root, joint_transform, img_transform, target_transform, )
train_loader = DataLoader(train_set, batch_size=batch_size, num_workers=8, shuffle=True)

print('=====>Validation Dataset loading<======')
validation_root = [ViSha_validation_root]
# validation_root = [ViSha_training_root]
validation_set = RCRDataset(validation_root, val_joint_transform, img_transform, target_transform=target_transform,
                            is_train=False)
val_loader = DataLoader(validation_set, batch_size=1, num_workers=8, shuffle=False)

print("max epoch:{}".format(args['max_epoch']))

# bce_loss = nn.BCELoss().cuda()
bce_loss = binary_xloss

log_path = os.path.join(ckpt_path, exp_name, str(datetime.datetime.now()) + '.txt')


def main():
    print('=====>Prepare Network {}<======'.format(exp_name))
    # multi-GPUs training
    if args['multi-GPUs']:
        net = torch.nn.DataParallel(OurModel()).cuda().train()
        # flownet = torch.nn.DataParallel(FlowNet2(opt, requires_grad=False)).cuda()
        gma = torch.nn.DataParallel(RAFTGMA(opt)).cuda()

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

        params = [
            {'params': [param for name, param in net.named_parameters() if name[-4:] == 'bias'],
             'lr': 2 * args['finetune_lr']},
            {'params': [param for name, param in net.named_parameters() if name[-4:] != 'bias'],
             'lr': args['finetune_lr']}
        ]

    # for p in gma.module.parameters():
    #     p.requires_grad = False
    optimizer = optim.SGD(params, momentum=args['momentum'], weight_decay=args['weight_decay'])
    # optimizer = optim.Adam(params, betas=(0.9, 0.99), eps=6e-8, weight_decay=args['weight_decay'])
    warm_up_with_cosine_lr = lambda epoch: epoch / (args['warm_up_epochs']) if epoch <= args[
        'warm_up_epochs'] else 0.5 * \
                               (math.cos((epoch - args['warm_up_epochs']) / (
                                           args['max_epoch'] - args['warm_up_epochs']) * math.pi) + 1)
    # warm_up_with_cosine_lr = lambda epoch: 0.5 * (math.cos((epoch - args['warm_up_epochs']) / (
    #                                        args['max_epoch'] - args['warm_up_epochs']) * math.pi) + 1)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=warm_up_with_cosine_lr)
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
    net.load_state_dict(torch.load('/mnt/data1/fxy/ViSha-main/models/BDRAR_plus_RCRNet13/3.pth')['model'])
    # net.load_state_dict(torch.load('/mnt/data1/fxy/ViSha-main/models/BDRAR_plus_RCRNet12/3.pth')['model'])

    train(net, optimizer, scheduler, gma)


def resize_flow(flow, size):
    origin_size = flow.shape[2:]
    flow = F.interpolate(flow, size=size, mode="nearest")
    flow[:, 0, :, :] /= origin_size[1] / size[1]  # dx
    flow[:, 1, :, :] /= origin_size[0] / size[0]  # dy
    return flow


def train(net, optimizer, scheduler, gma):

    # Taken from the torchvision tutorial
    # https://pytorch.org/vision/stable/auto_examples/plot_visualization_utils.html
    # model = deeplabv3_resnet50(pretrained=True, progress=False)
    # model = model.eval()
    #
    # if torch.cuda.is_available():
    #     model = model.cuda()
    #     input_tensor = input_tensor.cuda()
    #
    # output = model(input_tensor)
    #
    # class SegmentationModelOutputWrapper(torch.nn.Module):
    #     def __init__(self, model):
    #         super(SegmentationModelOutputWrapper, self).__init__()
    #         self.model = model
    #
    #     def forward(self, x):
    #         return self.model(x)["out"]
    #
    # model = SegmentationModelOutputWrapper(model)
    # output = model(input_tensor)
    #
    # normalized_masks = torch.nn.functional.softmax(output, dim=1).cpu()
    # sem_classes = [
    #     '__background__', 'aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus',
    #     'car', 'cat', 'chair', 'cow', 'diningtable', 'dog', 'horse', 'motorbike',
    #     'person', 'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor'
    # ]
    # sem_class_to_idx = {cls: idx for (idx, cls) in enumerate(sem_classes)}
    #
    # car_category = sem_class_to_idx["car"]
    # car_mask = normalized_masks[0, :, :, :].argmax(axis=0).detach().cpu().numpy()
    # car_mask_uint8 = 255 * np.uint8(car_mask == car_category)
    # car_mask_float = np.float32(car_mask == car_category)
    #
    # both_images = np.hstack((image, np.repeat(car_mask_uint8[:, :, None], 3, axis=-1)))
    # Image.fromarray(both_images)
    #
    # class SemanticSegmentationTarget:
    #     def __init__(self, category, mask):
    #         self.category = category
    #         self.mask = torch.from_numpy(mask)
    #         if torch.cuda.is_available():
    #             self.mask = self.mask.cuda()
    #
    #     def __call__(self, model_output):
    #         return (model_output[self.category, :, :] * self.mask).sum()
    #
    # target_layers = [model.model.backbone.layer4]
    # targets = [SemanticSegmentationTarget(car_category, car_mask_float)]
    # with GradCAM(model=model,
    #              target_layers=target_layers,
    #              use_cuda=torch.cuda.is_available()) as cam:
    #     grayscale_cam = cam(input_tensor=input_tensor,
    #                         targets=targets)[0, :]
    #     cam_image = show_cam_on_image(rgb_img, grayscale_cam, use_rgb=True)
    #
    # Image.fromarray(cam_image)

    curr_epoch = 1
    curr_iter = 1
    start = 0
    scaler = GradScaler()
    # res_path = '/mnt/data1/czpp/uncertainty_visual/'
    res_path = '/mnt/data1/czpp/ViSha_result/MSANet13.76/'
    check_mkdir(res_path)
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
        for data in tqdm(val_loader):
            # images = []
            exemplar, exemplar_gt, ref1 = data['exemplar'].cuda(), data['exemplar_gt'].cuda(), data[
                'ref1'].cuda()
            path1, path2, path3 = data['exemplar_path'], data['ref1_path'], data['ref2_path']
            ref2 = data['ref2'].cuda()
            # if data['exemplar_path'][0].split('/')[-2] != 'Bikeshow_ce':
            #     continue
            exemplar_resize, ref1_resize, ref2_resize = data['exemplar_resize'].cuda(), data[
                'ref1_resize'].cuda(), data[
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

            flow1_ref1 = resize_flow(flow_ref1_example, [50, 50])
            flow1_ref2 = resize_flow(flow_ref2_example, [50, 50])
            #
            flow2_ref1 = resize_flow(flow_ref1_example, [100, 100])
            flow2_ref2 = resize_flow(flow_ref2_example, [100, 100])

            h, w = data['height'], data['width']
            # for frame in data:
            #     images.append(frame['image'].cuda())
            # labels.append(frame['label'].cuda())

            image = np.array(Image.open(path1[0]).convert('RGB'))
            rgb_img = np.float32(image) / 255

            inputsss = exemplar

            preds = net([exemplar, ref1, ref2], flow1_ref1, flow1_ref2, flow2_ref1, flow2_ref2)

            target_layers = [net.pre_stage_network.down3]

            class SemanticSegmentationTarget:
                def __init__(self):
                    self.mask = exemplar_gt
                    # if torch.cuda.is_available():
                    #     self.mask = self.mask.cuda()

                def __call__(self, model_output):
                    return (model_output * self.mask).sum()

            targets = [SemanticSegmentationTarget()]
            pdb.set_trace()
            with GradCAM(model=net.pre_stage_network,
                         target_layers=target_layers,
                         use_cuda=torch.cuda.is_available()) as cam:
                # grayscale_cam = cam(input_tensor=inputsss,
                #                     targets=targets)[0, :]
                grayscale_cam = cam(input_tensor=inputsss)[0, :]
                cam_image = show_cam_on_image(rgb_img, grayscale_cam, use_rgb=True)

            Image.fromarray(cam_image).save('./visual_test.png')

            # preds = net([exemplar, ref1, ref2], flow_ref1_example, flow_ref2_example)
            # preds = net([exemplar, ref1, ref2])

            # prediction_sequence = []
            # for i in range(10):
            #     preds = net([exemplar, ref1, ref2])
            #     preds = F.sigmoid(preds)
            #     prediction_sequence.append((preds.detach().cpu()).to(torch.float32))
            # # pdb.set_trace()
            # uncertainty = torch.stack(prediction_sequence, dim=0).numpy()
            # uncertainty = np.var(uncertainty, axis=0) * 100
            # print(uncertainty.mean())
            # current_name = data['exemplar_path'][0]
            # # uncertainty = crf_refine(np.array(transforms.Resize((400, 400))(Image.open(current_name).convert('RGB'))), (uncertainty.squeeze() * 255.0).astype(np.uint8))
            # uncertainty_map = cv2.applyColorMap((uncertainty.squeeze() * 255.0).astype(np.uint8), cv2.COLORMAP_JET)
            # video_name = current_name.split('/')[-2]
            # image_name = os.path.basename(current_name)
            # check_mkdir(os.path.join(res_path, video_name))
            # cv2.imwrite(os.path.join(res_path, video_name) + '/' + image_name[:-4] + '.png', uncertainty_map)



            # preds, _, _, _ = net([exemplar, ref1, ref2], flow1_ref1, flow1_ref2, flow2_ref1, flow2_ref2)
            # preds = net([exemplar, ref1, ref2], flow_ref1_example1, flow_ref2_example1, flow_ref1_example2,
            #             flow_ref2_example2)
            # pdb.set_trace()
            # for i, pred_ in enumerate(preds):
            #     for j, pred in enumerate(pred_.detach().cpu()):
            #         h = data[i]['height'][j]
            #         w = data[i]['width'][j]

            res = (preds.detach().cpu() > 0).to(torch.float32)
            prediction = np.array(
                        transforms.Resize((h, w))(to_pil(res.squeeze(0).cpu()))).astype(np.uint8)
            current_name = data['exemplar_path'][0]
            video_name = current_name.split('/')[-2]
            image_name = os.path.basename(current_name)
            # result = crf_refine(np.array(Image.open(current_name).convert('RGB')), prediction)
            check_mkdir(os.path.join(res_path, video_name))
            Image.fromarray(prediction).save(os.path.join(res_path, video_name) + '/' + image_name[:-4] + '.png')
                    # pdb.set_trace()
            # label = np.array(Image.open(data[i]['label_path'][j]).convert('L')).astype(np.float32)
            label = np.array(transforms.Resize((h, w))(to_pil(exemplar_gt.squeeze(0).cpu()))).astype(np.float32)
            running_ber += cal_BER(prediction.astype(np.float32), label, 0)
            sample_num += 1
            # for i in range(4):
            #     prediction = np.array(
            #         transforms.Resize((h[0], w[0]))(to_pil(res[i, :, :, :].squeeze(0).cpu()))).astype(
            #         np.float32)
            #     # pdb.set_trace()
            #     # label = np.array(Image.open(data[i]['label_path'][j]).convert('L')).astype(np.float32)
            #     label = np.array(
            #         transforms.Resize((h[0], w[0]))(to_pil(exemplar_gt[i, :, :, :].squeeze(0).cpu()))).astype(
            #         np.float32)
            #     running_ber += cal_BER(prediction, label, 0)
            #     sample_num += 1
        running_ber = running_ber / sample_num
        print(running_ber)


if __name__ == '__main__':
    main()