import torch
import torch.nn.functional as F
from torch import nn

from .resnext_modify import ResNeXt101
from networks.Deformable2.deform_conv import DeformConv, _DeformConv, DeformConvPack
from torch.cuda.amp import autocast
from .BDRAR_model import BDRAR
import pdb


class UpsampleConvLayer(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size, stride, upsample=None, norm=None, bias=True):
        super(UpsampleConvLayer, self).__init__()

        self.upsample = upsample
        if upsample:
            self.upsample_layer = nn.Upsample(scale_factor=upsample, mode='nearest')

        reflection_padding = kernel_size // 2
        self.reflection_pad = nn.ReflectionPad2d(reflection_padding)
        self.conv2d = nn.Conv2d(in_channels, out_channels, kernel_size, stride, bias=bias)

        self.norm = norm
        if norm == "BN":
            self.norm_layer = nn.BatchNorm2d(out_channels)
        elif norm == "IN":
            self.norm_layer = nn.InstanceNorm2d(out_channels, track_running_stats=True)

    def forward(self, x):

        x_in = x
        if self.upsample:
            x_in = self.upsample_layer(x_in)
        out = self.reflection_pad(x_in)
        out = self.conv2d(out)

        if self.norm in ["BN" or "IN"]:
            out = self.norm_layer(out)

        return out


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


class MPLNet(nn.Module):
    def __init__(self):
        super(MPLNet, self).__init__()

        resnext = ResNeXt101()
        self.layer0 = resnext.layer0
        self.layer1 = resnext.layer1
        self.layer2 = resnext.layer2
        self.layer3 = resnext.layer3
        self.layer4 = resnext.layer4

        self.bottel4 = nn.Sequential(
            nn.Conv2d(2048, 256, 1, bias=False), nn.BatchNorm2d(256), nn.LeakyReLU(negative_slope=0.2, inplace=True)
        )
        self.bottel3 = nn.Sequential(
            nn.Conv2d(1024, 256, 1, bias=False), nn.BatchNorm2d(256), nn.LeakyReLU(negative_slope=0.2, inplace=True)
        )
        self.bottel2 = nn.Sequential(
            nn.Conv2d(512, 256, 1, bias=False), nn.BatchNorm2d(256), nn.LeakyReLU(negative_slope=0.2, inplace=True)
        )

        self.off2d_4_1 = nn.Conv2d(512, 18 * 8, 3, padding=1, bias=False)
        self.off2d_4_2 = nn.Conv2d(512, 18 * 8, 3, padding=1, bias=False)
        self.off2d_4_3 = nn.Conv2d(512, 18 * 8, 3, padding=1, bias=False)
        self.deconv_4 = DeformConv(256, 256, kernel_size=(3, 3), stride=1, padding=1, dilation=1, groups=1,
                                   deformable_groups=8, im2col_step=1, bias=False)

        self.off2d_3_1 = nn.Conv2d(512, 18 * 8, 3, padding=1, bias=False)
        self.off2d_3_2 = nn.Conv2d(512, 18 * 8, 3, padding=1, bias=False)
        self.off2d_3_3 = nn.Conv2d(512, 18 * 8, 3, padding=1, bias=False)
        self.deconv_3 = DeformConv(256, 256, kernel_size=(3, 3), stride=1, padding=1, dilation=1, groups=1,
                                   deformable_groups=8, im2col_step=1, bias=False)

        self.deconv_4_2 = DeformConv(256, 256, kernel_size=(3, 3), stride=1, padding=1, dilation=1, groups=1,
                                   deformable_groups=8, im2col_step=1, bias=False)
        self.sigmoid = nn.Sigmoid()

        self.gate = nn.Conv2d(512, 256, 3, padding=1, bias=False)
        # self.gate2 = nn.Conv2d(512, 256, 3, padding=1, bias=False)

        # f4
        self.refine4_1 = nn.Sequential(
            nn.Conv2d(256+256+256, 256, 1, bias=False), nn.BatchNorm2d(256), nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.Conv2d(256, 256, 3, padding=1, groups=32, bias=False), nn.BatchNorm2d(256), nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.Conv2d(256, 256, 1, bias=False), nn.BatchNorm2d(256)
        )
        # f4 + f3
        self.refine4_2 = nn.Sequential(
            nn.Conv2d(256 + 256, 256, 1, bias=False), nn.BatchNorm2d(256), nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.Conv2d(256, 256, 3, padding=1, groups=32, bias=False), nn.BatchNorm2d(256), nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.Conv2d(256, 256, 1, bias=False), nn.BatchNorm2d(256)
        )
        # f3
        self.refine3_1 = nn.Sequential(
            nn.Conv2d(256 + 256, 256, 1, bias=False), nn.BatchNorm2d(256), nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.Conv2d(256, 256, 3, padding=1, groups=32, bias=False), nn.BatchNorm2d(256), nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.Conv2d(256, 256, 1, bias=False), nn.BatchNorm2d(256)
        )
        # f3 + f2
        self.deconv4 = UpsampleConvLayer(256, 256, kernel_size=3, stride=1, upsample=2, bias=False, norm='BN')
        self.refine3_2 = nn.Sequential(
            nn.Conv2d(256 + 256, 256, 1, bias=False), nn.BatchNorm2d(256), nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.Conv2d(256, 256, 3, padding=1, groups=32, bias=False), nn.BatchNorm2d(256), nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.Conv2d(256, 256, 1, bias=False), nn.BatchNorm2d(256)
        )
        # f2 + f1
        self.deconv3 = UpsampleConvLayer(256, 256, kernel_size=3, stride=1, upsample=2, bias=False, norm='BN')
        self.refine2 = nn.Sequential(
            nn.Conv2d(256 + 256, 256, 1, bias=False), nn.BatchNorm2d(256), nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.Conv2d(256, 256, 3, padding=1, groups=32, bias=False), nn.BatchNorm2d(256), nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.Conv2d(256, 256, 1, bias=False), nn.BatchNorm2d(256)
        )

        self.deconv2 = UpsampleConvLayer(256, 256, kernel_size=3, stride=1, upsample=2, bias=False,
                                         norm='BN')
        self.deconv1 = UpsampleConvLayer(256, 256, kernel_size=3, stride=1, upsample=2, bias=False,
                                         norm='BN')
        self.relu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

        self.pred = nn.Sequential(
            nn.Conv2d(256, 32, 3, padding=1, bias=False), nn.BatchNorm2d(32), nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.Conv2d(32, 8, 3, padding=1, bias=False), nn.BatchNorm2d(8), nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.Dropout(0.1), nn.Conv2d(8, 1, 1)
        )
        self.fuse = nn.Conv2d(5, 1, 1)

    def feature_extract(self, x):
        layer0 = self.layer0(x)
        layer1 = self.layer1(layer0)
        layer2 = self.layer2(layer1)
        layer3 = self.layer3(layer2)
        layer4 = self.layer4(layer3)

        layer4 = self.bottle4(layer4)
        layer3 = self.bottle3(layer3)
        layer2 = self.bottle2(layer2)

        return layer1, layer2, layer3, layer4

    def forward(self, clip, flow4):
        with autocast():
            res = []
            last_frame_feature = None
            memory = None
            for i in range(len(clip)):
                f1, f2, f3, f4 = self.feature_extract(clip[i])
                if last_frame_feature is None:
                    last_frame_feature = [f1, f2, f3, f4]
                if memory is None:
                    memory = f4
                last_f1, last_f2, last_f3, last_f4 = last_frame_feature
                last_frame_feature = [f1, f2, f3, f4]

                fea_4 = torch.cat([f4, last_f4], dim=1)
                offset41 = (self.off2d_4_1(fea_4) + flow4[i].repeat(1, 72, 1, 1)) / 2
                fea_4 = torch.cat([f4, self.deconv_4(last_f4.float(), offset41.float())], dim=1)
                offset42 = self.off2d_4_2(fea_4) + offset41
                fea_4 = torch.cat([f4, self.deconv_4(last_f4.float(), offset42.float())], dim=1)
                offset43 = self.off2d_4_3(fea_4) + offset42
                last_f4 = self.deconv_4(last_f4.float(), offset43.float())

                fea_3 = torch.cat([f3, last_f3], dim=1)
                offset31 = (self.off2d_3_1(fea_3) + flow4[i].repeat(1, 72, 1, 1)) / 2
                fea_3 = torch.cat([f3, self.deconv_3(last_f3.float(), offset31.float())], dim=1)
                offset32 = self.off2d_3_2(fea_3) + offset31
                fea_3 = torch.cat([f3, self.deconv_3(last_f3.float(), offset32.float())], dim=1)
                offset33 = self.off2d_3_3(fea_3) + offset32
                last_f3 = self.deconv_3(last_f3.float(), offset33.float())
                
                memory = self.deconv_4_2(memory.float(), offset43.float())
                memory_fea = torch.cat([f4, memory], dim=1)
                alpha = self.sigmoid(self.gate(memory_fea))
                memory = memory * alpha + f4 * (1 - alpha)

                f4 = self.relu(self.refine4_1(torch.cat([f4, last_f4, memory], dim=1)) + f4)
                f3 = self.relu(self.refine4_2(torch.cat([f4, f3], dim=1)) + f4)

                f3 = self.relu(self.refine3_1(torch.cat([f3, last_f3], dim=1)) + f3)
                f3 = self.deconv4(f3)
                f2 = self.relu(self.refine3_2(torch.cat([f3, f2], dim=1)) + f3)

                f2 = self.deconv3(f2)
                f1 = self.relu(self.refine2(torch.cat([f2, f1], dim=1)) + f2)

                f_final = self.deconv2(f1)

                f_final = self.deconv1(f_final)

                pre_4 = F.upsample(self.pred(f4), size=clip[0].size()[2:], mode='bilinear')
                pre_3 = F.upsample(self.pred(f3), size=clip[0].size()[2:], mode='bilinear')
                pre_2 = F.upsample(self.pred(f2), size=clip[0].size()[2:], mode='bilinear')
                pre_1 = F.upsample(self.pred(f1), size=clip[0].size()[2:], mode='bilinear')
                pre_global = self.pred(f_final)
                pre_final = self.fuse(torch.cat([pre_4, pre_3, pre_2, pre_1, pre_global], dim=1))

                res.append([pre_final, pre_global, pre_1, pre_2, pre_3, pre_4])

            return res

    def recurrent_forward(self, current_input, last_frame_feature, memory, flow4):
        with autocast():
            f1, f2, f3, f4 = self.feature_extract(current_input)
            if last_frame_feature is None:
                last_frame_feature = [f1, f2, f3, f4]
            if memory is None:
                memory = f4
            last_f1, last_f2, last_f3, last_f4 = last_frame_feature
            last_frame_feature = [f1, f2, f3, f4]
            fea_4 = torch.cat([f4, last_f4], dim=1)
            offset41 = (self.off2d_4_1(fea_4) + flow4.repeat(1, 72, 1, 1)) / 2
            fea_4 = torch.cat([f4, self.deconv_4(last_f4.float(), offset41.float())], dim=1)
            offset42 = self.off2d_4_2(fea_4) + offset41
            fea_4 = torch.cat([f4, self.deconv_4(last_f4.float(), offset42.float())], dim=1)
            offset43 = self.off2d_4_3(fea_4) + offset42
            last_f4 = self.deconv_4(last_f4.float(), offset43.float())

            fea_3 = torch.cat([f3, last_f3], dim=1)
            offset31 = (self.off2d_3_1(fea_3) + flow4.repeat(1, 72, 1, 1)) / 2
            fea_3 = torch.cat([f3, self.deconv_3(last_f3.float(), offset31.float())], dim=1)
            offset32 = self.off2d_3_2(fea_3) + offset31
            fea_3 = torch.cat([f3, self.deconv_3(last_f3.float(), offset32.float())], dim=1)
            offset33 = self.off2d_3_3(fea_3) + offset32
            last_f3 = self.deconv_3(last_f3.float(), offset33.float())

            memory = self.deconv_4_2(memory.float(), offset43.float())
            memory_fea = torch.cat([f4, memory], dim=1)
            alpha = self.sigmoid(self.gate(memory_fea))
            memory = memory * alpha + f4 * (1 - alpha)

            f4 = self.relu(self.refine4_1(torch.cat([f4, last_f4, memory], dim=1)) + f4)
            f3 = self.relu(self.refine4_2(torch.cat([f4, f3], dim=1)) + f4)

            f3 = self.relu(self.refine3_1(torch.cat([f3, last_f3], dim=1)) + f3)
            f3 = self.deconv4(f3)
            f2 = self.relu(self.refine3_2(torch.cat([f3, f2], dim=1)) + f3)

            f2 = self.deconv3(f2)
            f1 = self.relu(self.refine2(torch.cat([f2, f1], dim=1)) + f2)

            f_final = self.deconv2(f1)

            f_final = self.deconv1(f_final)

            pre_4 = F.upsample(self.pred(f4), size=current_input.size()[2:], mode='bilinear')
            pre_3 = F.upsample(self.pred(f3), size=current_input.size()[2:], mode='bilinear')
            pre_2 = F.upsample(self.pred(f2), size=current_input.size()[2:], mode='bilinear')
            pre_1 = F.upsample(self.pred(f1), size=current_input.size()[2:], mode='bilinear')
            pre_global = self.pred(f_final)
            pre_final = self.fuse(torch.cat([pre_4, pre_3, pre_2, pre_1, pre_global], dim=1))

        return last_frame_feature, memory, pre_final



