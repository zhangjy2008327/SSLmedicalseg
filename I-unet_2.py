import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.fft as fft

skip_connections = []


class Hybrid_pooling(nn.Module):
    def __init__(self, nin, original_size ,flag, kernel_size, stride):
        super(Hybrid_pooling, self).__init__()
        self.max_pooling = nn.MaxPool2d(kernel_size=kernel_size, stride=stride)
        # self.avg_pooling = nn.AvgPool2d(kernel_size=kernel_size, stride=stride)
        self.conv1x1 = nn.Conv2d(nin*2, nin, kernel_size=1)
        self.upsample = nn.Upsample(size=original_size, mode='bilinear', align_corners=False)
        self.flag = flag

    def forward(self, x):
        max_pooled = self.max_pooling(x)
        # avg_pooled = self.avg_pooling(x)
        hartley_transform = fft.fftn(x)#应用Hartley变换
        hartley_magnitude = torch.abs(hartley_transform)#变换的幅度
        hartley_pooled = self.max_pooling(hartley_magnitude)#使用最大池化池化Hartley变换后的幅度

        pooled_features = torch.cat([max_pooled, hartley_pooled], dim=1)
        out = self.conv1x1(pooled_features)
        # print("out---", out.shape)
        if self.flag:
            out = self.upsample(out)
        # print("out2---", out.shape)

        return out


class Block(nn.Module):
    def __init__(self, in_channels, out_channels, size, pooling):
        super(Block, self).__init__()
        self.pooling = pooling
        self.conva = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        )
        self.convb = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=5, padding=2),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        )
        self.convc = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        )
        self.convd = nn.Sequential(
            Hybrid_pooling(in_channels, size, flag=True, kernel_size=2, stride=2),
            nn.BatchNorm2d(in_channels),
            nn.ReLU()
        )
        self.conv_2d = nn.Conv2d(3*out_channels+in_channels, out_channels, kernel_size=1)
        self.dropout = nn.Dropout(p=0.2)
        self.pool = pooling
        self.pooling = Hybrid_pooling(out_channels, size, flag=False, kernel_size=2, stride=2)#flag控制池化后是否还原图像大小
        self.out_channel = out_channels

    def forward(self, x):
        # print("x---", x.shape)
        conva = self.conva(x)
        # print("xa-----", conva.shape)
        convb = self.convb(x)
        convc = self.convc(x)
        convd = self.convd(x)
        cat = torch.cat([conva, convb, convc, convd], dim=1)
        if self.out_channel == 256:
            return cat
        conv = self.conv_2d(cat)
        out = self.dropout(conv)

        if self.pool:
            out = self.pooling(out)
            skip_connections.append(conv)
            # print("conv",conv.shape)

        return out

class Iunet_encoder(nn.Module):
    def __init__(self):
        super(Iunet_encoder, self).__init__()
        self.conv1 = Block(3, 16, 128, pooling=True)#第三个参数是当前图像大小
        self.conv2 = Block(16, 32, 64, pooling=True)
        self.conv3 = Block(32, 64, 32, pooling=True)
        self.conv4 = Block(64, 128, 16, pooling=True)
        self.conv5 = Block(128, 256, 8, pooling=True)
        self.pool = Hybrid_pooling(3, 128, flag=False, kernel_size=16, stride=16)

    def forward(self, x):
        input = x

        # print("x----", x.shape)
        conv1 = self.conv1(x)
        # print("conv1---", conv1.shape)
        conv2 = self.conv2(conv1)
        # print("conv2---", conv2.shape)
        conv3 = self.conv3(conv2)
        # print("conv3---", conv3.shape)
        conv4 = self.conv4(conv3)
        # print("conv4---", conv4.shape)
        conv5 = self.conv5(conv4)
        # print("conv5---", conv5.shape)
        input = self.pool(input)
        # print("out---", input.shape)

        out = torch.cat([input, conv5], dim=1)#通道数是899
        # exit(0)
        print("encoder---", out.shape)
        return out

class Iunet_decoder(nn.Module):
    def __init__(self):
        super(Iunet_decoder, self).__init__()
        self.conv1 = Block(256, 128, 16, pooling=False)
        self.conv2 = Block(128, 64, 32, pooling=False)
        self.conv3 = Block(64, 32, 64, pooling=False)
        self.conv4 = Block(32, 16, 128, pooling=False)
        self.relu = nn.ReLU()
        self.sig = nn.Sigmoid()
        self.c1 = nn.Conv2d(899, 256, kernel_size=1)
        self.end = nn.Conv2d(16, 1, kernel_size=1)
        self.upconv1 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.upconv2 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.upconv3 = nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2)
        self.upconv4 = nn.ConvTranspose2d(32, 16, kernel_size=2, stride=2)
        self.upconv5 = nn.ConvTranspose2d(64, 16, kernel_size=2, stride=2)

    def forward(self, x):

        conv1 = self.c1(x)
        up1 = self.upconv1(conv1)
        # print("up1---", up1.shape)
        x1 = torch.cat((up1, skip_connections[3]), dim=1)
        x1_1 = self.conv1(x1)
        # print("x1_1----", x1_1.shape)
        # print("-------------", len(skip_connections))

        up2 = self.upconv2(x1_1)
        # print("up2---------", up2.shape)
        x2 = torch.cat((up2, skip_connections[2]), dim=1)
        x2_1 = self.conv2(x2)

        # print("x2_1---", x2_1.shape)
        up3 = self.upconv3(x2_1)
        x3 = torch.cat((up3, skip_connections[1]), dim=1)
        x3_1 = self.conv3(x3)

        # print("x3_1--",x3_1.shape)
        up4 = self.upconv4(x3_1)
        x4 = torch.cat((up4, skip_connections[0]), dim=1)
        x4_1 = self.conv4(x4)

        xx = self.sig(self.end(x4_1))

        skip_connections.clear()
        return xx


class Iunet(nn.Module):
    def __init__(self):
        super(Iunet, self).__init__()

        self.encoder = Iunet_encoder()
        self.decoder = Iunet_decoder()

    def forward(self, x):
        output = self.encoder(x)
        output = self.decoder(output)
        return output

img = torch.randn(3, 3, 128, 128)
model = Iunet()
out = model(img)
print('out-------', out.shape)
