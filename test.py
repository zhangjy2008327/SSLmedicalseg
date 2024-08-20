import torch
from torch import nn
import torch.nn.functional as F

class CNN1(nn.Module):  # (32, 3, 1)
    def __init__(self, input_channels, map_size, pad):
        super(CNN1, self).__init__()
        self.weight = nn.Parameter(torch.ones(input_channels, input_channels, map_size, map_size, map_size),
                                   requires_grad=False)
        self.bias = nn.Parameter(torch.zeros(input_channels), requires_grad=False)
        self.pad = pad
        self.stride = 1
        self.norm = nn.BatchNorm3d(input_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        out = F.conv3d(x, self.weight, self.bias, stride=self.stride, padding=self.pad)
        out = self.norm(out)
        out = self.relu(out)
        return out


class M2SNet(nn.Module):
    def __init__(self, ):
        super(M2SNet, self).__init__()
        self.conv2 = nn.Conv3d(32, 64, 3, 2, 1)
        self.conv3 = nn.Conv3d(64, 128, 3, 2, 1)
        self.conv4 = nn.Conv3d(128, 256, 3, 2, 1)
        self.conv5 = nn.Conv3d(256, 320, 3, 2, 1)
        self.conv6 = nn.Conv3d(320, 320, 3, (1, 2, 2), 1)

        self.layer2 = nn.Sequential(
            nn.Conv3d(32, 32, 3, 1, 1, groups=32),
            nn.Conv3d(32, 64, 1),
            nn.BatchNorm3d(64),
            nn.ReLU(inplace=True),
        )
        self.layer3 = nn.Sequential(
            nn.Conv3d(64, 64, 3, 1, 1, groups=64),
            nn.Conv3d(64, 128, 1),
            nn.BatchNorm3d(128),
            nn.ReLU(inplace=True),
        )
        self.layer4 = nn.Sequential(
            nn.Conv3d(128, 128, 3, 1, 1, groups=128),
            nn.Conv3d(128, 256, 1),
            nn.BatchNorm3d(256),
            nn.ReLU(inplace=True),
        )
        self.layer5 = nn.Sequential(
            nn.Conv3d(256, 256, 3, 1, 1, groups=256),
            nn.Conv3d(256, 320, 1),
            nn.BatchNorm3d(320),
            nn.ReLU(inplace=True),
        )
        self.layer6 = nn.Sequential(
            nn.Conv3d(320, 320, 3, 1, 1, groups=320),
            nn.Conv3d(320, 320, 1),
            nn.BatchNorm3d(320),
            nn.ReLU(inplace=True),
        )
        # self.layer1 = nn.Conv3d(32, 64, 3, 2, 1, bias=False)
        # self.layer2 = nn.Conv3d(64, 128, 3, 2, 1, bias=False)
        # self.layer3 = nn.Conv3d(128, 256, 3, 2, 1, bias=False)

        self.conv_3 = CNN1(32, 3, 1)
        self.conv_5 = CNN1(32, 5, 2)

        self.x6_dem_1 = nn.Sequential(nn.Conv3d(320, 32, kernel_size=3, stride=1, padding=1),
                                      nn.BatchNorm3d(32),
                                      nn.ReLU(inplace=True)
                                      )
        self.x5_dem_1 = nn.Sequential(nn.Conv3d(320, 32, kernel_size=3, stride=1, padding=1),
                                      nn.BatchNorm3d(32),
                                      nn.ReLU(inplace=True)
                                      )
        self.x4_dem_1 = nn.Sequential(nn.Conv3d(256, 32, kernel_size=3, stride=1, padding=1),
                                      nn.BatchNorm3d(32),
                                      nn.ReLU(inplace=True)
                                      )
        self.x3_dem_1 = nn.Sequential(nn.Conv3d(128, 32, kernel_size=3, stride=1, padding=1),
                                      nn.BatchNorm3d(32),
                                      nn.ReLU(inplace=True)
                                      )
        self.x2_dem_1 = nn.Sequential(nn.Conv3d(64, 32, kernel_size=3, stride=1, padding=1),
                                      nn.BatchNorm3d(32),
                                      nn.ReLU(inplace=True)
                                      )
        self.x6_x5 = nn.Sequential(nn.Conv3d(32, 32, kernel_size=3, padding=1),
                                   nn.BatchNorm3d(32),
                                   nn.ReLU(inplace=True))
        self.x5_x4 = nn.Sequential(nn.Conv3d(32, 32, kernel_size=3, padding=1),
                                   nn.BatchNorm3d(32),
                                   nn.ReLU(inplace=True))
        self.x4_x3 = nn.Sequential(nn.Conv3d(32, 32, kernel_size=3, padding=1),
                                   nn.BatchNorm3d(32),
                                   nn.ReLU(inplace=True))
        self.x3_x2 = nn.Sequential(nn.Conv3d(32, 32, kernel_size=3, padding=1),
                                   nn.BatchNorm3d(32),
                                   nn.ReLU(inplace=True))
        self.x2_x1 = nn.Sequential(nn.Conv3d(32, 32, kernel_size=3, padding=1),
                                   nn.BatchNorm3d(32),
                                   nn.ReLU(inplace=True))

        self.x6_x5_x4 = nn.Sequential(nn.Conv3d(32, 32, kernel_size=3, padding=1),
                                      nn.BatchNorm3d(32),
                                      nn.ReLU(inplace=True))
        self.x5_x4_x3 = nn.Sequential(nn.Conv3d(32, 32, kernel_size=3, padding=1),
                                      nn.BatchNorm3d(32),
                                      nn.ReLU(inplace=True))
        self.x4_x3_x2 = nn.Sequential(nn.Conv3d(32, 32, kernel_size=3, padding=1),
                                      nn.BatchNorm3d(32),
                                      nn.ReLU(inplace=True))
        self.x3_x2_x1 = nn.Sequential(nn.Conv3d(32, 32, kernel_size=3, padding=1),
                                      nn.BatchNorm3d(32),
                                      nn.ReLU(inplace=True))

        self.x6_x5_x4_x3 = nn.Sequential(nn.Conv3d(32, 32, kernel_size=3, padding=1),
                                         nn.BatchNorm3d(32),
                                         nn.ReLU(inplace=True))
        self.x5_x4_x3_x2 = nn.Sequential(nn.Conv3d(32, 32, kernel_size=3, padding=1),
                                         nn.BatchNorm3d(32),
                                         nn.ReLU(inplace=True))
        self.x4_x3_x2_x1 = nn.Sequential(nn.Conv3d(32, 32, kernel_size=3, padding=1),
                                         nn.BatchNorm3d(32),
                                         nn.ReLU(inplace=True))

        self.x6_x5_x4_x3_x2 = nn.Sequential(nn.Conv3d(32, 32, kernel_size=3, padding=1),
                                            nn.BatchNorm3d(32),
                                            nn.ReLU(inplace=True))
        self.x5_x4_x3_x2_x1 = nn.Sequential(nn.Conv3d(32, 32, kernel_size=3, padding=1),
                                            nn.BatchNorm3d(32),
                                            nn.ReLU(inplace=True))

        self.x6_x5_x4_x3_x2_x1 = nn.Sequential(nn.Conv3d(32, 32, kernel_size=3, padding=1),
                                               nn.BatchNorm3d(32),
                                               nn.ReLU(inplace=True))

        self.level6 = nn.Sequential(nn.Conv3d(32, 320, kernel_size=3, padding=1),
                                    nn.BatchNorm3d(320),
                                    nn.ReLU(inplace=True))
        self.level5 = nn.Sequential(nn.Conv3d(32, 320, kernel_size=3, padding=1),
                                    nn.BatchNorm3d(320),
                                    nn.ReLU(inplace=True))
        self.level4 = nn.Sequential(nn.Conv3d(32, 256, kernel_size=3, padding=1),
                                    nn.BatchNorm3d(256),
                                    nn.ReLU(inplace=True))
        self.level3 = nn.Sequential(nn.Conv3d(32, 128, kernel_size=3, padding=1),
                                    nn.BatchNorm3d(128),
                                    nn.ReLU(inplace=True))
        self.level2 = nn.Sequential(nn.Conv3d(32, 64, kernel_size=3, padding=1),
                                    nn.BatchNorm3d(64),
                                    nn.ReLU(inplace=True))
        self.level1 = nn.Sequential(nn.Conv3d(32, 32, kernel_size=3, padding=1),
                                    nn.BatchNorm3d(32),
                                    nn.ReLU(inplace=True))

    def forward(self, x):
        #x1 = skips[0]
        #x2 = skips[1]
        #x3 = skips[2]
        #x4 = skips[3]
        #x5 = skips[4]
        #x6 = skips[5]

        x1 = x
        x2 = self.conv2(x1)
        x3 = self.conv3(x2)
        x4 = self.conv4(x3)
        x5 = self.conv5(x4)
        x6 = self.conv6(x5)

        skips = []

        # torch.Size([1, 32, 32, 32, 32])
        # torch.Size([1, 256, 32, 32, 32])
        # torch.Size([1, 128, 16, 16, 16])
        # torch.Size([1, 256, 8, 8, 8])
        # torch.Size([1, 2048, 4, 4, 4])

        # '''
        # MS1 因为第一层通道数为32， 所以第一层不需要借助卷积，不改变图像大小只改变通道数大小
        x6_dem_1 = self.x6_dem_1(x6)
        x5_dem_1 = self.x5_dem_1(x5)
        x4_dem_1 = self.x4_dem_1(x4)
        x3_dem_1 = self.x3_dem_1(x3)
        x2_dem_1 = self.x2_dem_1(x2)

        x6_dem_1_up = F.interpolate(x6_dem_1, size=x5.size()[2:], mode='trilinear', align_corners=False)
        x6_dem_1_up_map1 = self.conv_3(x6_dem_1_up)
        x5_dem_1_map1 = self.conv_3(x5_dem_1)
        x6_dem_1_up_map2 = self.conv_5(x6_dem_1_up)
        x5_dem_1_map2 = self.conv_5(x5_dem_1)
        x6_5 = self.x6_x5(
            abs(x6_dem_1_up - x5_dem_1) + abs(x6_dem_1_up_map1 - x5_dem_1_map1) + abs(x6_dem_1_up_map2 - x5_dem_1_map2))

        x5_dem_1_up = F.interpolate(x5_dem_1, size=x4.size()[2:], mode='trilinear', align_corners=False)
        x5_dem_1_up_map1 = self.conv_3(x5_dem_1_up)
        x4_dem_1_map1 = self.conv_3(x4_dem_1)
        x5_dem_1_up_map2 = self.conv_5(x5_dem_1_up)
        x4_dem_1_map2 = self.conv_5(x4_dem_1)
        x5_4 = self.x5_x4(
            abs(x5_dem_1_up - x4_dem_1) + abs(x5_dem_1_up_map1 - x4_dem_1_map1) + abs(x5_dem_1_up_map2 - x4_dem_1_map2))

        # Multi-scale Subtraction Unit 1
        x4_dem_1_up = F.interpolate(x4_dem_1, size=x3.size()[2:], mode='trilinear', align_corners=False)
        x4_dem_1_up_map1 = self.conv_3(x4_dem_1_up)
        x3_dem_1_map1 = self.conv_3(x3_dem_1)
        x4_dem_1_up_map2 = self.conv_5(x4_dem_1_up)
        x3_dem_1_map2 = self.conv_5(x3_dem_1)
        x4_3 = self.x4_x3(
            abs(x4_dem_1_up - x3_dem_1) + abs(x4_dem_1_up_map1 - x3_dem_1_map1) + abs(x4_dem_1_up_map2 - x3_dem_1_map2))
        # torch.Size([1, 32, 16, 16, 16])

        x3_dem_1_up = F.interpolate(x3_dem_1, size=x2.size()[2:], mode='trilinear', align_corners=False)
        x3_dem_1_up_map1 = self.conv_3(x3_dem_1_up)
        x2_dem_1_map1 = self.conv_3(x2_dem_1)
        x3_dem_1_up_map2 = self.conv_5(x3_dem_1_up)
        x2_dem_1_map2 = self.conv_5(x2_dem_1)
        x3_2 = self.x3_x2(
            abs(x3_dem_1_up - x2_dem_1) + abs(x3_dem_1_up_map1 - x2_dem_1_map1) + abs(x3_dem_1_up_map2 - x2_dem_1_map2))
        # torch.Size([1, 32, 32, 32, 32])

        x2_dem_1_up = F.interpolate(x2_dem_1, size=x1.size()[2:], mode='trilinear', align_corners=False)
        x2_dem_1_up_map1 = self.conv_3(x2_dem_1_up)
        x1_map1 = self.conv_3(x1)
        x2_dem_1_up_map2 = self.conv_5(x2_dem_1_up)
        x1_map2 = self.conv_5(x1)
        x2_1 = self.x2_x1(abs(x2_dem_1_up - x1) + abs(x2_dem_1_up_map1 - x1_map1) + abs(x2_dem_1_up_map2 - x1_map2))
        # torch.Size([1, 32, 32, 32, 32])

        # Multi-scale Subtraction Unit 2
        x6_5_up = F.interpolate(x6_5, size=x5_4.size()[2:], mode='trilinear', align_corners=False)
        x6_5_up_map1 = self.conv_3(x6_5_up)
        x5_4_map1 = self.conv_3(x5_4)
        x6_5_up_map2 = self.conv_5(x6_5_up)
        x5_4_map2 = self.conv_5(x5_4)
        x6_5_4 = self.x6_x5_x4(abs(x6_5_up - x5_4) + abs(x6_5_up_map1 - x5_4_map1) + abs(x6_5_up_map2 - x5_4_map2))

        x5_4_up = F.interpolate(x5_4, size=x4_3.size()[2:], mode='trilinear', align_corners=False)
        x5_4_up_map1 = self.conv_3(x5_4_up)
        x4_3_map1 = self.conv_3(x4_3)
        x5_4_up_map2 = self.conv_5(x5_4_up)
        x4_3_map2 = self.conv_5(x4_3)
        x5_4_3 = self.x5_x4_x3(abs(x5_4_up - x4_3) + abs(x5_4_up_map1 - x4_3_map1) + abs(x5_4_up_map2 - x4_3_map2))

        x4_3_up = F.interpolate(x4_3, size=x3_2.size()[2:], mode='trilinear', align_corners=False)
        x4_3_up_map1 = self.conv_3(x4_3_up)
        x3_2_map1 = self.conv_3(x3_2)
        x4_3_up_map2 = self.conv_5(x4_3_up)
        x3_2_map2 = self.conv_5(x3_2)
        x4_3_2 = self.x4_x3_x2(abs(x4_3_up - x3_2) + abs(x4_3_up_map1 - x3_2_map1) + abs(x4_3_up_map2 - x3_2_map2))
        # torch.Size([1, 32, 32, 32, 32])

        x3_2_up = F.interpolate(x3_2, size=x2_1.size()[2:], mode='trilinear', align_corners=False)
        x3_2_up_map1 = self.conv_3(x3_2_up)
        x2_1_map1 = self.conv_3(x2_1)
        x3_2_up_map2 = self.conv_5(x3_2_up)
        x2_1_map2 = self.conv_5(x2_1)
        x3_2_1 = self.x3_x2_x1(abs(x3_2_up - x2_1) + abs(x3_2_up_map1 - x2_1_map1) + abs(x3_2_up_map2 - x2_1_map2))
        # torch.Size([1, 32, 32, 32, 32])

        x6_5_4_up = F.interpolate(x6_5_4, size=x5_4_3.size()[2:], mode='trilinear', align_corners=False)
        x6_5_4_up_map1 = self.conv_3(x6_5_4_up)
        x5_4_3_map1 = self.conv_3(x5_4_3)
        x6_5_4_up_map2 = self.conv_5(x6_5_4_up)
        x5_4_3_map2 = self.conv_5(x5_4_3)
        x6_5_4_3 = self.x6_x5_x4_x3(
            abs(x6_5_4_up - x5_4_3) + abs(x6_5_4_up_map1 - x5_4_3_map1) + abs(x6_5_4_up_map2 - x5_4_3_map2))

        x5_4_3_up = F.interpolate(x5_4_3, size=x4_3_2.size()[2:], mode='trilinear', align_corners=False)
        x5_4_3_up_map1 = self.conv_3(x5_4_3_up)
        x4_3_2_map1 = self.conv_3(x4_3_2)
        x5_4_3_up_map2 = self.conv_5(x5_4_3_up)
        x4_3_2_map2 = self.conv_5(x4_3_2)
        x5_4_3_2 = self.x5_x4_x3_x2(
            abs(x5_4_3_up - x4_3_2) + abs(x5_4_3_up_map1 - x4_3_2_map1) + abs(x5_4_3_up_map2 - x4_3_2_map2))

        x4_3_2_up = F.interpolate(x4_3_2, size=x3_2_1.size()[2:], mode='trilinear', align_corners=False)
        x4_3_2_up_map1 = self.conv_3(x4_3_2_up)
        x3_2_1_map1 = self.conv_3(x3_2_1)
        x4_3_2_up_map2 = self.conv_5(x4_3_2_up)
        x3_2_1_map2 = self.conv_5(x3_2_1)
        x4_3_2_1 = self.x4_x3_x2_x1(
            abs(x4_3_2_up - x3_2_1) + abs(x4_3_2_up_map1 - x3_2_1_map1) + abs(x4_3_2_up_map2 - x3_2_1_map2))
        # torch.Size([1, 32, 32, 32, 32])

        x6_5_4_3_up = F.interpolate(x6_5_4_3, size=x5_4_3_2.size()[2:], mode='trilinear', align_corners=False)
        x6_5_4_3_up_map1 = self.conv_3(x6_5_4_3_up)
        x5_4_3_2_map1 = self.conv_3(x5_4_3_2)
        x6_5_4_3_up_map2 = self.conv_5(x6_5_4_3_up)
        x5_4_3_2_map2 = self.conv_5(x5_4_3_2)
        x6_5_4_3_2 = self.x6_x5_x4_x3_x2(
            abs(x6_5_4_3_up - x5_4_3_2) + abs(x6_5_4_3_up_map1 - x5_4_3_2_map1) + abs(x6_5_4_3_up_map2 - x5_4_3_2_map2))

        x5_4_3_2_up = F.interpolate(x5_4_3_2, size=x4_3_2_1.size()[2:], mode='trilinear', align_corners=False)
        x5_4_3_2_up_map1 = self.conv_3(x5_4_3_2_up)
        x4_3_2_1_map1 = self.conv_3(x4_3_2_1)
        x5_4_3_2_up_map2 = self.conv_5(x5_4_3_2_up)
        x4_3_2_1_map2 = self.conv_5(x4_3_2_1)
        x5_4_3_2_1 = self.x5_x4_x3_x2_x1(
            abs(x5_4_3_2_up - x4_3_2_1) + abs(x5_4_3_2_up_map1 - x4_3_2_1_map1) + abs(x5_4_3_2_up_map2 - x4_3_2_1_map2))

        x6_5_4_3_2_up = F.interpolate(x6_5_4_3_2, size=x5_4_3_2_1.size()[2:], mode='trilinear', align_corners=False)
        x6_5_4_3_2_up_map1 = self.conv_3(x6_5_4_3_2_up)
        x5_4_3_2_1_map1 = self.conv_3(x5_4_3_2_1)
        x6_5_4_3_2_up_map2 = self.conv_5(x6_5_4_3_2_up)
        x5_4_3_2_1_map2 = self.conv_5(x5_4_3_2_1)
        x6_5_4_3_2_1 = self.x6_x5_x4_x3_x2_x1(
            abs(x6_5_4_3_2_up - x5_4_3_2_1) + abs(x6_5_4_3_2_up_map1 - x5_4_3_2_1_map1) + abs(
                x6_5_4_3_2_up_map2 - x5_4_3_2_1_map2))

        level6 = self.level6(x6_dem_1)
        level5 = self.level5(x6_5)
        level4 = self.level4(x6_5_4 + x5_4)
        level3 = self.level3(x6_5_4_3 + x5_4_3 + x4_3)
        level2 = self.level2(x6_5_4_3_2 + x5_4_3_2 + x4_3_2 + x3_2)
        level1 = self.level1(x6_5_4_3_2_1 + x5_4_3_2_1 + x4_3_2_1 + x3_2_1 + x2_1)

        skips.append(level1)
        skips.append(level2)
        skips.append(level3)
        skips.append(level4)
        skips.append(level5)
        skips.append(level6)

        return skips

if torch.cuda.is_available():
    print('cuda')

#skips = [torch.randn(2, 32, 80, 160, 160).cuda( ), torch.randn(2, 64, 40, 80, 80).cuda( ), torch.randn(2, 128, 20, 40, 40).cuda( ), torch.randn(2, 256, 10, 20, 20).cuda( ), torch.randn(2, 320, 5, 10, 10)#.cuda( ), torch.randn(2, 320, 5, 5, 5).cuda( )]
x = torch.randn(2, 32, 80, 160, 160).cuda()
model = M2SNet().cuda( )
y = model(x)

for i in y:
    print(i.shape)
    print('--------')