import torch
import torch.nn.functional as F
from torch import nn

from .resnext_modify import ResNeXt101
from networks.Deformable2.deform_conv import DeformConv, _DeformConv, DeformConvPack
from torch.cuda.amp import autocast
import pdb


class _AttentionModule(nn.Module):
    def __init__(self):
        super(_AttentionModule, self).__init__()
        self.block1 = nn.Sequential(
            nn.Conv2d(64, 64, 1, bias=False), nn.BatchNorm2d(64), nn.ReLU(),
            nn.Conv2d(64, 64, 3, dilation=2, padding=2, groups=32, bias=False), nn.BatchNorm2d(64), nn.ReLU(),
            nn.Conv2d(64, 64, 1, bias=False), nn.BatchNorm2d(64)
        )
        self.block2 = nn.Sequential(
            nn.Conv2d(64, 64, 1, bias=False), nn.BatchNorm2d(64), nn.ReLU(),
            nn.Conv2d(64, 64, 3, dilation=3, padding=3, groups=32, bias=False), nn.BatchNorm2d(64), nn.ReLU(),
            nn.Conv2d(64, 64, 1, bias=False), nn.BatchNorm2d(64)
        )
        self.block3 = nn.Sequential(
            nn.Conv2d(64, 64, 1, bias=False), nn.BatchNorm2d(64), nn.ReLU(),
            nn.Conv2d(64, 64, 3, dilation=4, padding=4, groups=32, bias=False), nn.BatchNorm2d(64), nn.ReLU(),
            nn.Conv2d(64, 32, 1, bias=False), nn.BatchNorm2d(32)
        )
        self.down = nn.Sequential(
            nn.Conv2d(64, 32, 1, bias=False), nn.BatchNorm2d(32)
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        block1 = self.relu(self.block1(x) + x)
        block2 = self.relu(self.block2(block1) + block1)
        block3 = F.sigmoid(self.block3(block2) + self.down(block2))
        return block3


class BDRAR_backbone(nn.Module):
    def __init__(self):
        super(BDRAR_backbone, self).__init__()
        resnext = ResNeXt101()
        self.layer0 = resnext.layer0
        self.layer1 = resnext.layer1
        self.layer2 = resnext.layer2
        self.layer3 = resnext.layer3
        self.layer4 = resnext.layer4

        self.down4 = nn.Sequential(
            nn.Conv2d(2048, 32, 1, bias=False), nn.BatchNorm2d(32), nn.ReLU()
        )
        self.down3 = nn.Sequential(
            nn.Conv2d(1024, 32, 1, bias=False), nn.BatchNorm2d(32), nn.ReLU()
        )
        self.down2 = nn.Sequential(
            nn.Conv2d(512, 32, 1, bias=False), nn.BatchNorm2d(32), nn.ReLU()
        )
        self.down1 = nn.Sequential(
            nn.Conv2d(256, 32, 1, bias=False), nn.BatchNorm2d(32), nn.ReLU()
        )

        self.refine3_hl = nn.Sequential(
            nn.Conv2d(64, 32, 1, bias=False), nn.BatchNorm2d(32), nn.ReLU(),
            nn.Conv2d(32, 32, 3, padding=1, groups=32, bias=False), nn.BatchNorm2d(32), nn.ReLU(),
            nn.Conv2d(32, 32, 1, bias=False), nn.BatchNorm2d(32)
        )
        self.refine2_hl = nn.Sequential(
            nn.Conv2d(64, 32, 1, bias=False), nn.BatchNorm2d(32), nn.ReLU(),
            nn.Conv2d(32, 32, 3, padding=1, groups=32, bias=False), nn.BatchNorm2d(32), nn.ReLU(),
            nn.Conv2d(32, 32, 1, bias=False), nn.BatchNorm2d(32)
        )
        self.refine1_hl = nn.Sequential(
            nn.Conv2d(64, 32, 1, bias=False), nn.BatchNorm2d(32), nn.ReLU(),
            nn.Conv2d(32, 32, 3, padding=1, groups=32, bias=False), nn.BatchNorm2d(32), nn.ReLU(),
            nn.Conv2d(32, 32, 1, bias=False), nn.BatchNorm2d(32)
        )
        self.attention3_hl = _AttentionModule()
        self.attention2_hl = _AttentionModule()
        self.attention1_hl = _AttentionModule()

        self.refine2_lh = nn.Sequential(
            nn.Conv2d(64, 32, 1, bias=False), nn.BatchNorm2d(32), nn.ReLU(),
            nn.Conv2d(32, 32, 3, padding=1, groups=32, bias=False), nn.BatchNorm2d(32), nn.ReLU(),
            nn.Conv2d(32, 32, 1, bias=False), nn.BatchNorm2d(32)
        )
        self.refine4_lh = nn.Sequential(
            nn.Conv2d(64, 32, 1, bias=False), nn.BatchNorm2d(32), nn.ReLU(),
            nn.Conv2d(32, 32, 3, padding=1, groups=32, bias=False), nn.BatchNorm2d(32), nn.ReLU(),
            nn.Conv2d(32, 32, 1, bias=False), nn.BatchNorm2d(32)
        )
        self.refine3_lh = nn.Sequential(
            nn.Conv2d(64, 32, 1, bias=False), nn.BatchNorm2d(32), nn.ReLU(),
            nn.Conv2d(32, 32, 3, padding=1, groups=32, bias=False), nn.BatchNorm2d(32), nn.ReLU(),
            nn.Conv2d(32, 32, 1, bias=False), nn.BatchNorm2d(32)
        )
        self.attention2_lh = _AttentionModule()
        self.attention3_lh = _AttentionModule()
        self.attention4_lh = _AttentionModule()

        self.fuse_attention = nn.Sequential(
            nn.Conv2d(64, 16, 3, padding=1, bias=False), nn.BatchNorm2d(16), nn.ReLU(),
            nn.Conv2d(16, 2, 1)
        )

        self.predict = nn.Sequential(
            nn.Conv2d(32, 8, 3, padding=1, bias=False), nn.BatchNorm2d(8), nn.ReLU(),
            nn.Dropout(0.1), nn.Conv2d(8, 1, 1)
        )

        for m in self.modules():
            if isinstance(m, nn.ReLU):
                m.inplace = True

        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        layer0 = self.layer0(x)
        layer1 = self.layer1(layer0)
        layer2 = self.layer2(layer1)
        layer3 = self.layer3(layer2)
        layer4 = self.layer4(layer3)

        down4 = self.down4(layer4)
        down3 = self.down3(layer3)
        down2 = self.down2(layer2)
        down1 = self.down1(layer1)

        down4 = F.upsample(down4, size=down3.size()[2:], mode='bilinear')
        refine3_hl_0 = self.relu(self.refine3_hl(torch.cat((down4, down3), 1)) + down4)
        refine3_hl_0 = (1 + self.attention3_hl(torch.cat((down4, down3), 1))) * refine3_hl_0
        refine3_hl_1 = self.relu(self.refine3_hl(torch.cat((refine3_hl_0, down3), 1)) + refine3_hl_0)
        refine3_hl_1 = (1 + self.attention3_hl(torch.cat((refine3_hl_0, down3), 1))) * refine3_hl_1

        refine3_hl_1 = F.upsample(refine3_hl_1, size=down2.size()[2:], mode='bilinear')
        refine2_hl_0 = self.relu(self.refine2_hl(torch.cat((refine3_hl_1, down2), 1)) + refine3_hl_1)
        refine2_hl_0 = (1 + self.attention2_hl(torch.cat((refine3_hl_1, down2), 1))) * refine2_hl_0
        refine2_hl_1 = self.relu(self.refine2_hl(torch.cat((refine2_hl_0, down2), 1)) + refine2_hl_0)
        refine2_hl_1 = (1 + self.attention2_hl(torch.cat((refine2_hl_0, down2), 1))) * refine2_hl_1

        refine2_hl_1 = F.upsample(refine2_hl_1, size=down1.size()[2:], mode='bilinear')
        refine1_hl_0 = self.relu(self.refine1_hl(torch.cat((refine2_hl_1, down1), 1)) + refine2_hl_1)
        refine1_hl_0 = (1 + self.attention1_hl(torch.cat((refine2_hl_1, down1), 1))) * refine1_hl_0
        refine1_hl_1 = self.relu(self.refine1_hl(torch.cat((refine1_hl_0, down1), 1)) + refine1_hl_0)
        refine1_hl_1 = (1 + self.attention1_hl(torch.cat((refine1_hl_0, down1), 1))) * refine1_hl_1

        down2 = F.upsample(down2, size=down1.size()[2:], mode='bilinear')
        refine2_lh_0 = self.relu(self.refine2_lh(torch.cat((down1, down2), 1)) + down1)
        refine2_lh_0 = (1 + self.attention2_lh(torch.cat((down1, down2), 1))) * refine2_lh_0
        refine2_lh_1 = self.relu(self.refine2_lh(torch.cat((refine2_lh_0, down2), 1)) + refine2_lh_0)
        refine2_lh_1 = (1 + self.attention2_lh(torch.cat((refine2_lh_0, down2), 1))) * refine2_lh_1

        down3 = F.upsample(down3, size=down1.size()[2:], mode='bilinear')
        refine3_lh_0 = self.relu(self.refine3_lh(torch.cat((refine2_lh_1, down3), 1)) + refine2_lh_1)
        refine3_lh_0 = (1 + self.attention3_lh(torch.cat((refine2_lh_1, down3), 1))) * refine3_lh_0
        refine3_lh_1 = self.relu(self.refine3_lh(torch.cat((refine3_lh_0, down3), 1)) + refine3_lh_0)
        refine3_lh_1 = (1 + self.attention3_lh(torch.cat((refine3_lh_0, down3), 1))) * refine3_lh_1

        down4 = F.upsample(down4, size=down1.size()[2:], mode='bilinear')
        refine4_lh_0 = self.relu(self.refine4_lh(torch.cat((refine3_lh_1, down4), 1)) + refine3_lh_1)
        refine4_lh_0 = (1 + self.attention4_lh(torch.cat((refine3_lh_1, down4), 1))) * refine4_lh_0
        refine4_lh_1 = self.relu(self.refine4_lh(torch.cat((refine4_lh_0, down4), 1)) + refine4_lh_0)
        refine4_lh_1 = (1 + self.attention4_lh(torch.cat((refine4_lh_0, down4), 1))) * refine4_lh_1

        predict1_hl = self.predict(refine1_hl_1)

        predict4_lh = self.predict(refine4_lh_1)

        fuse_attention = F.sigmoid(self.fuse_attention(torch.cat((refine1_hl_1, refine4_lh_1), 1)))
        fuse_predict = torch.sum(fuse_attention * torch.cat((predict1_hl, predict4_lh), 1), 1, True)

        fuse_predict = F.upsample(fuse_predict, size=x.size()[2:], mode='bilinear')

        return fuse_predict


class OurModel(nn.Module):
    def __init__(self):
        super(OurModel, self).__init__()

        self.pre_stage_network = BDRAR_backbone()

        self.pre_stage_network.load_state_dict(torch.load('/mnt/data1/czpp/BDRAR-master/ckpt/BDRAR/3001.pth'))

        self.off2d_4_1 = nn.Conv2d(64, 18 * 8, 3, padding=1, bias=False)
        self.off2d_4_2 = nn.Conv2d(64, 18 * 8, 3, padding=1, bias=False)
        self.off2d_4_3 = nn.Conv2d(64, 18 * 8, 3, padding=1, bias=False)
        self.deconv_4 = DeformConv(32, 256, kernel_size=(3, 3), stride=1, padding=1, dilation=1, groups=1,
                                   deformable_groups=8, im2col_step=1, bias=False)

        self.off2d_3 = nn.Conv2d(64, 18 * 8, 3, padding=1, bias=False)
        self.deconv_3 = DeformConv(32, 64, kernel_size=(3, 3), stride=1, padding=1, dilation=1, groups=1,
                                   deformable_groups=8, im2col_step=1, bias=False)

        self.off2d_21 = nn.Conv2d(64, 18 * 8, 3, padding=1, bias=False)
        self.off2d_22 = nn.Conv2d(18 * 8, 18 * 8, 1, padding=0, bias=False)
        self.deconv_2 = DeformConv(32, 32, kernel_size=(3, 3), stride=1, padding=1, dilation=1, groups=1,
                                   deformable_groups=8, im2col_step=1, bias=False)

        self.off2d_11 = nn.Conv2d(64, 18 * 8, 3, padding=1, bias=False)
        self.off2d_12 = nn.Conv2d(18 * 8, 18 * 8, 1, padding=0, bias=False)
        self.deconv_1 = DeformConv(32, 32, kernel_size=(3, 3), stride=1, padding=1, dilation=1, groups=1,
                                   deformable_groups=8, im2col_step=1, bias=False)

        self.fuse4 = nn.Sequential(
            nn.Conv2d(96, 32, 3, 1, 1, bias=False), nn.BatchNorm2d(32), nn.ReLU())

        self.fuse3 = nn.Sequential(
            nn.Conv2d(96, 32, 3, 1, 1, bias=False), nn.BatchNorm2d(32), nn.ReLU())

        self.fuse2 = nn.Sequential(
            nn.Conv2d(96, 32, 3, 1, 1, bias=False), nn.BatchNorm2d(32), nn.ReLU())

        self.fuse1 = nn.Sequential(
            nn.Conv2d(96, 32, 3, 1, 1, bias=False), nn.BatchNorm2d(32), nn.ReLU())

        for m in self.modules():
            if isinstance(m, nn.ReLU):
                m.inplace = True

        self.relu = nn.ReLU(inplace=True)

    def feature_extract(self, x):
        layer0 = self.pre_stage_network.layer0(x)
        layer1 = self.pre_stage_network.layer1(layer0)
        layer2 = self.pre_stage_network.layer2(layer1)
        layer3 = self.pre_stage_network.layer3(layer2)
        layer4 = self.pre_stage_network.layer4(layer3)

        down4 = self.pre_stage_network.down4(layer4)
        down3 = self.pre_stage_network.down3(layer3)
        down2 = self.pre_stage_network.down2(layer2)
        down1 = self.pre_stage_network.down1(layer1)
        return down1, down2, down3, down4

    def forward(self, clip, flow41, flow42, training = True):
        with autocast():
            f1, f2, f3, f4 = self.feature_extract(clip[0])
            ref1_f1, ref1_f2, ref1_f3, ref1_f4 = self.feature_extract(clip[1])
            ref2_f1, ref2_f2, ref2_f3, ref2_f4 = self.feature_extract(clip[2])

            fea_41 = torch.cat([f4, ref1_f4], dim=1)
            offset41 = (self.off2d_4_1(fea_41) + flow41.repeat(1, 72, 1, 1)) / 2
            fea4_1 = self.deconv_4(ref1_f4.float(), offset41.float())
            fea_41 = torch.cat([f4, fea4_1], dim=1)
            offset42 = self.off2d_4_2(fea_41) + offset41
            fea4_1 = self.deconv_4(ref1_f4.float(), offset42.float())
            fea_41 = torch.cat([f4, fea4_1], dim=1)
            offset43 = self.off2d_4_3(fea_41) + offset42
            fea4_1 = self.deconv_4(ref1_f4.float(), offset43.float())

            fea_41 = torch.cat([f4, ref2_f4], dim=1)
            offset41 = (self.off2d_4_1(fea_41) + flow42.repeat(1, 72, 1, 1)) / 2
            fea4_2 = self.deconv_4(ref2_f4.float(), offset41.float())
            fea_41 = torch.cat([f4, fea4_2], dim=1)
            offset42 = self.off2d_4_2(fea_41) + offset41
            fea4_2 = self.deconv_4(ref2_f4.float(), offset42.float())
            fea_41 = torch.cat([f4, fea4_2], dim=1)
            offset43 = self.off2d_4_3(fea_41) + offset42
            fea4_2 = self.deconv_4(ref2_f4.float(), offset43.float())

            fea_31 = torch.cat([f3, ref1_f3], dim=1)
            offset31 = self.off2d_3(fea_31)
            fea3_1 = self.deconv_3(ref1_f3.float(), offset31.float())
            fea_32 = torch.cat([f3, ref2_f3], dim=1)
            offset32 = self.off2d_3(fea_32)
            fea3_2 = self.deconv_3(ref2_f3.float(), offset32.float())

            fea_21 = torch.cat([f2, ref1_f2], dim=1)
            offset21 = self.off2d_21(fea_21)
            fea2_1 = self.deconv_2(ref1_f2.float(), offset21.float())
            fea_22 = torch.cat([f2, ref2_f2], dim=1)
            offset22 = self.off2d_21(fea_22)
            fea2_2 = self.deconv_2(ref2_f2.float(), offset22.float())

            fea_11 = torch.cat([f1, ref1_f1], dim=1)
            offset11 = self.off2d_11(fea_11)
            fea1_1 = self.deconv_1(ref1_f1.float(), offset11.float())
            fea_12 = torch.cat([f1, ref2_f1], dim=1)
            offset12 = self.off2d_11(fea_12)
            fea1_2 = self.deconv_1(ref2_f1.float(), offset12.float())

            f_fuse4 = self.fuse4(torch.cat([f4, fea4_1, fea4_2], dim=1))
            f_fuse3 = self.fuse3(torch.cat([f3, fea3_1, fea3_2], dim=1))
            f_fuse2 = self.fuse2(torch.cat([f2, fea2_1, fea2_2], dim=1))
            f_fuse1 = self.fuse1(torch.cat([f1, fea1_1, fea1_2], dim=1))

            f4 = F.upsample(f4, size=f3.size()[2:], mode='bilinear')
            refine3_hl_0 = self.relu(self.pre_stage_network.refine3_hl(torch.cat((f4, f_fuse3), 1)) + f4)
            refine3_hl_0 = (1 + self.pre_stage_network.attention3_hl(torch.cat((f4, f_fuse3), 1))) * refine3_hl_0
            refine3_hl_1 = self.relu(self.pre_stage_network.refine3_hl(torch.cat((refine3_hl_0, f_fuse3), 1)) + refine3_hl_0)
            refine3_hl_1 = (1 + self.pre_stage_network.attention3_hl(torch.cat((refine3_hl_0, f_fuse3), 1))) * refine3_hl_1

            refine3_hl_1 = F.upsample(refine3_hl_1, size=f2.size()[2:], mode='bilinear')
            refine2_hl_0 = self.relu(self.pre_stage_network.refine2_hl(torch.cat((refine3_hl_1, f_fuse2), 1)) + refine3_hl_1)
            refine2_hl_0 = (1 + self.pre_stage_network.attention2_hl(torch.cat((refine3_hl_1, f_fuse2), 1))) * refine2_hl_0
            refine2_hl_1 = self.relu(self.pre_stage_network.refine2_hl(torch.cat((refine2_hl_0, f_fuse2), 1)) + refine2_hl_0)
            refine2_hl_1 = (1 + self.pre_stage_network.attention2_hl(torch.cat((refine2_hl_0, f_fuse2), 1))) * refine2_hl_1

            refine2_hl_1 = F.upsample(refine2_hl_1, size=f1.size()[2:], mode='bilinear')
            refine1_hl_0 = self.relu(self.pre_stage_network.refine1_hl(torch.cat((refine2_hl_1, f_fuse1), 1)) + refine2_hl_1)
            refine1_hl_0 = (1 + self.pre_stage_network.attention1_hl(torch.cat((refine2_hl_1, f_fuse1), 1))) * refine1_hl_0
            refine1_hl_1 = self.relu(self.pre_stage_network.refine1_hl(torch.cat((refine1_hl_0, f_fuse1), 1)) + refine1_hl_0)
            refine1_hl_1 = (1 + self.pre_stage_network.attention1_hl(torch.cat((refine1_hl_0, f_fuse1), 1))) * refine1_hl_1

            down2 = F.upsample(f_fuse2, size=f1.size()[2:], mode='bilinear')
            refine2_lh_0 = self.relu(self.pre_stage_network.refine2_lh(torch.cat((f1, down2), 1)) + f1)
            refine2_lh_0 = (1 + self.pre_stage_network.attention2_lh(torch.cat((f1, down2), 1))) * refine2_lh_0
            refine2_lh_1 = self.relu(self.pre_stage_network.refine2_lh(torch.cat((refine2_lh_0, down2), 1)) + refine2_lh_0)
            refine2_lh_1 = (1 + self.pre_stage_network.attention2_lh(torch.cat((refine2_lh_0, down2), 1))) * refine2_lh_1

            down3 = F.upsample(f_fuse3, size=f1.size()[2:], mode='bilinear')
            refine3_lh_0 = self.relu(self.pre_stage_network.refine3_lh(torch.cat((refine2_lh_1, down3), 1)) + refine2_lh_1)
            refine3_lh_0 = (1 + self.pre_stage_network.attention3_lh(torch.cat((refine2_lh_1, down3), 1))) * refine3_lh_0
            refine3_lh_1 = self.relu(self.pre_stage_network.refine3_lh(torch.cat((refine3_lh_0, down3), 1)) + refine3_lh_0)
            refine3_lh_1 = (1 + self.pre_stage_network.attention3_lh(torch.cat((refine3_lh_0, down3), 1))) * refine3_lh_1

            down4 = F.upsample(f_fuse4, size=f1.size()[2:], mode='bilinear')
            refine4_lh_0 = self.relu(self.pre_stage_network.refine4_lh(torch.cat((refine3_lh_1, down4), 1)) + refine3_lh_1)
            refine4_lh_0 = (1 + self.pre_stage_network.attention4_lh(torch.cat((refine3_lh_1, down4), 1))) * refine4_lh_0
            refine4_lh_1 = self.relu(self.pre_stage_network.refine4_lh(torch.cat((refine4_lh_0, down4), 1)) + refine4_lh_0)
            refine4_lh_1 = (1 + self.pre_stage_network.attention4_lh(torch.cat((refine4_lh_0, down4), 1))) * refine4_lh_1

            predict1_hl = self.pre_stage_network.predict(refine1_hl_1)

            predict4_lh = self.pre_stage_network.predict(refine4_lh_1)

            fuse_attention = F.sigmoid(self.pre_stage_network.fuse_attention(torch.cat((refine1_hl_1, refine4_lh_1), 1)))
            fuse_predict = torch.sum(fuse_attention * torch.cat((predict1_hl, predict4_lh), 1), 1, True)

            predict1_hl = F.upsample(predict1_hl, size=clip[0].size()[2:], mode='bilinear')
            predict4_lh = F.upsample(predict4_lh, size=clip[0].size()[2:], mode='bilinear')
            fuse_predict = F.upsample(fuse_predict, size=clip[0].size()[2:], mode='bilinear')

            if training:
                return fuse_predict, predict1_hl, predict4_lh

            return fuse_predict
