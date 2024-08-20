import torch.nn as nn
import torch

skip_connections_one = []
skip_connections_two = []


class Conv(nn.Module):
    def __init__(self, nin, nout):
        super(Conv, self).__init__()
        self.depthwise = nn.Conv2d(nin, nout, kernel_size=3, padding=1)
        self.norm2 = nn.BatchNorm2d(nout)
        self.relu = nn.ReLU()

    def forward(self, x):
        out = self.depthwise(x)
        out = self.norm2(out)
        out_relu = self.relu(out)
        out = out_relu
        return out


class DoubleConv(nn.Module):
    def __init__(self, nin, nout):
        super(DoubleConv, self).__init__()
        self.conv = nn.Sequential(
            Conv(nin, nout),
            Conv(nout, nout),

        )

    def forward(self, x):
        out = self.conv(x)
        return out



class Attention_block(nn.Module):
    def __init__(self,F_g, F_l, F_int):
        super(Attention_block, self).__init__()
        self.W_g = nn.Sequential(
            nn.Conv2d(F_g, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int)
        )

        self.W_x = nn.Sequential(
            nn.Conv2d(F_l, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int)
        )

        self.psi = nn.Sequential(
            nn.Conv2d(F_int, 1, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )

        self.relu = nn.ReLU(inplace=True)

    def forward(self, g, x):
        x1 = self.W_x(x)
        # g1 = self.W_g(g)  # 1x512x64x64->conv(512，256)/B.N.->1x256x64x64
        g1 = nn.functional.interpolate(self.W_g(g), x1.shape[2:], mode = 'bilinear', align_corners = False)
        # print('g1--',g1.shape)

        # print('x1---',x1.shape)
        psi = self.relu(g1 + x1)
        psi = self.psi(psi)  # 得到权重矩阵  1x256x64x64 -> 1x1x64x64 ->sigmoid 结果到（0，1）

        return x * psi  # 与low-level feature相乘，将权重矩阵赋值进去

class Unet_encoder(nn.Module):
    def __init__(self):
        super(Unet_encoder, self).__init__()
        self.down1 = DoubleConv(3, 16)
        self.down2 = DoubleConv(16, 32)
        self.down3 = DoubleConv(32, 64)
        self.down4 = DoubleConv(64, 128)
        self.down5 = DoubleConv(128, 256)
        self.MaxPool = nn.MaxPool2d(kernel_size=2, stride=2)
        # self.down2_1 = Conv3d_w(1, 16)#普通二维卷积
        # self.down2_2 = Conv3d_w(16, 32)
        # self.down2_3 = Conv3d_w(32, 64)
        # self.down2_4 = Conv3d_w(64, 128)
        # self.down2_5 = Conv3d_w(128, 256)
        # self.bn = nn.BatchNorm3d()

    def forward(self, x):
        # print('x---',x.shape)
        down1 = self.down1(x)
        skip_connections_two.append(down1)
        # print('down1----',down1.shape)
        MaxPool1= self.MaxPool(down1)
        # print('MaxPool1----',MaxPool1.shape)


        down2 = self.down2(MaxPool1)
        skip_connections_two.append(down2)
        MaxPool2 = self.MaxPool(down2)


        down3 = self.down3(MaxPool2)
        skip_connections_two.append(down3)
        MaxPool3 = self.MaxPool(down3)


        down4 = self.down4(MaxPool3)
        skip_connections_two.append(down4)
        MaxPool4 = self.MaxPool(down4)

        down5 = self.down5(MaxPool4)
        return down5


class Unet_decoder(nn.Module):
    def __init__(self):
        super(Unet_decoder, self).__init__()

        self.up1 = DoubleConv(256, 128)
        self.up2 = DoubleConv(128, 64)
        self.up3 = DoubleConv(64, 32)
        self.up4 = DoubleConv(32, 16)
        self.up5 = nn.Conv2d(16, 1, kernel_size=1)  # last feature

        # self.MaxPool = nn.MaxPool3d(kernel_size=2, stride=2)
        self.UpConv1 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.UpConv2 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.UpConv3 = nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2)
        self.UpConv4 = nn.ConvTranspose2d(32, 16, kernel_size=2, stride=2)
        self.Att1 = Attention_block( F_g=256, F_l=128, F_int=128)
        self.Att2 = Attention_block( F_g=128, F_l=64, F_int=64)
        self.Att3 = Attention_block( F_g=64, F_l=32, F_int=32)
        self.Att4 = Attention_block(F_g=32, F_l=16, F_int=16)


    def forward(self, x):
        # print('x---',x.shape)
        x1 = self.Att1(g=x, x=skip_connections_two[3])
        # print('x1----', x1)
        UpC1 = self.UpConv1(x)
        # print('UpC1--', UpC1.shape)

        cat1 = torch.cat((UpC1, x1), dim=1)
        up1 = self.up1(cat1)
        # print("up1------", up1.shape)

        x2 = self.Att2(g=up1, x=skip_connections_two[2])
        UpC2 = self.UpConv2(up1)
        cat2 = torch.cat((UpC2, x2), dim=1)
        up2 = self.up2(cat2)
        # print("up2------", up2.shape)

        x3 = self.Att3(g=up2, x=skip_connections_two[1])
        UpC3 = self.UpConv3(up2)
        cat3 = torch.cat((UpC3, x3), dim=1)
        up3 = self.up3(cat3)
        # print("up3------", up3.shape)

        x4 = self.Att4(g=up3, x=skip_connections_two[0])
        UpC4 = self.UpConv4(up3)
        cat4 = torch.cat((UpC4, x4), dim=1)
        up4 = self.up4(cat4)
        # print("up4------", up4.shape)
        up5 = self.up5(up4)
        # print("up5------", up5.shape)
        skip_connections_two.clear()#这个每次储存的会有上一次的参数  梯度回传会有问题 （这个问题真的找了好长时间）所以要每次都清空一下子
        return up5

class Unet(nn.Module):
    def __init__(self):
        super(Unet, self).__init__()

        self.encoder = Unet_encoder()
        self.decoder = Unet_decoder()

    def forward(self, x):
        output = self.encoder(x)
        output = self.decoder(output)
        return output

