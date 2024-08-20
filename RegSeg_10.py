# ERFNET full network definition for Pytorch
# Sept 2017
# Eduardo Romera
#######################

import torch
import torch.nn as nn
# import torch.nn.init as init
import torch.nn.functional as F

skip_connnection=[]

class DownsamplerBlock(nn.Module):
    def __init__(self, ninput, noutput):
        super().__init__()

        self.conv = nn.Conv2d(ninput, noutput - ninput, (3, 3), stride=2, padding=1, bias=True)
        self.pool = nn.MaxPool2d(2, stride=2)
        self.bn = nn.BatchNorm2d(noutput, eps=1e-3)

    def forward(self, input):
        output = torch.cat([self.conv(input), self.pool(input)], 1)
        output = self.bn(output)
        return F.relu(output)

class DilatedConv(nn.Module):
    def __init__(self,w,dilations,group_width,stride,bias):
        super().__init__()
        num_splits=len(dilations)
        assert(w%num_splits==0)
        temp=w//num_splits
        assert(temp%group_width==0)
        groups=temp//group_width
        convs=[]
        for d in dilations:
            convs.append(nn.Conv2d(temp,temp,3,padding=d,dilation=d,stride=stride,bias=bias,groups=groups))
        self.convs=nn.ModuleList(convs)
        self.num_splits=num_splits
    def forward(self,x):
        x=torch.tensor_split(x,self.num_splits,dim=1)#
        res=[]
        for i in range(self.num_splits):#
            res.append(self.convs[i](x[i]))
        return torch.cat(res, dim=1)

#
class SEModule(nn.Module):
    """Squeeze-and-Excitation (SE) block: AvgPool, FC, Act, FC, Sigmoid."""
    def __init__(self, w_in, w_se):
        super().__init__()
        #print("w_in--------w_se----", w_in, w_se)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv1=nn.Conv2d(w_in, w_se, 1, bias=True)#1x1卷积    
        self.act1=nn.ReLU(inplace=True)#
        self.conv2=nn.Conv2d(w_se, w_in, 1, bias=True)#
        self.act2=nn.Sigmoid()

    def forward(self, x):
        y=self.avg_pool(x)
        #print("y----------", y.shape)
        y=self.act1(self.conv1(y))
        y=self.act2(self.conv2(y))
        #print("y--------", y.shape)

        return x * y


class CBMA(nn.Module):
    def __init__(self, channel, reduction=16, spatial_kernel=7):
        super(CBMA, self).__init__()
        #channel attention
        self.max_pool=nn.AdaptiveMaxPool2d(1)
        self.avg_pool=nn.AdaptiveAvgPool2d(1)
        #shared mlp
        self.mlp = nn.Sequential(
            nn.Conv2d(channel, channel//reduction, 1, bias=False),
            nn.ReLU(),
            nn.Conv2d(channel//reduction, channel,1 ,bias=False)
        )
        #spatial attention
        self.conv = nn.Conv2d(2, 1, kernel_size=spatial_kernel, padding=spatial_kernel//2, bias=False)
        self.sig = nn.Sigmoid()


    def forward(self, x):
        max_out = self.mlp(self.max_pool(x))
        avg_out = self.mlp(self.avg_pool(x))
        channel_out = self.sig(max_out+avg_out)
        x = x*channel_out

        # max_out, _ = torch.max(x, dim=1, keepdim=True)
        # avg_out = torch.mean(x, dim=1, keepdim=True)
        # spatial_out = self.sig(self.conv(torch.cat([max_out,avg_out], dim=1)))
        # x = x*spatial_out
        return x




class Shortcut(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, avg_downsample=False):
        super(Shortcut, self).__init__()
        if avg_downsample and stride != 1:
            self.avg=nn.AvgPool2d(2, 2, ceil_mode=True)
            self.conv=nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
            self.bn=nn.BatchNorm2d(out_channels)
        else:
            self.avg=None
            self.conv=nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False)
            self.bn=nn.BatchNorm2d(out_channels)

    def forward(self, x):
        if self.avg is not None:
            x=self.avg(x)
        x = self.conv(x)
        x = self.bn(x)
        return x


class DBlock(nn.Module):
    def __init__(self, in_channels, out_channels, dilations,group_width, stride,attention):#g
        super().__init__()
        avg_downsample = True
        groups = out_channels//group_width
        self.conv1 = nn.Conv2d(in_channels, out_channels,kernel_size=1,bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.act1 = nn.ReLU()
        if len(dilations) == 1:
            dilation = dilations[0]
            self.conv2 = nn.Conv2d(out_channels, out_channels,kernel_size=3,stride=stride,groups=groups, padding=dilation,dilation=dilation,bias=False)
        else:
            self.conv2=DilatedConv(out_channels,dilations,group_width=group_width,stride=stride,bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.act2 = nn.ReLU()
        self.conv3 = nn.Conv2d(out_channels, out_channels,kernel_size=1,bias=False)
        self.bn3 = nn.BatchNorm2d(out_channels)
        self.act3 = nn.ReLU()
        if attention=="se":
            self.se = SEModule(out_channels,out_channels//4)
        elif attention=="cbam":
            self.se = CBMA(out_channels)
        else :
            self.se = None
        if stride != 1 or in_channels != out_channels:#这

            self.shortcut=Shortcut(in_channels,out_channels,stride,avg_downsample)
        else:
            self.shortcut = None

    def forward(self, x):
        shortcut=self.shortcut(x) if self.shortcut else x
        x=self.conv1(x)
        x=self.bn1(x)
        x=self.act1(x)
        x=self.conv2(x)
        x=self.bn2(x)
        x=self.act2(x)
        if self.se is not None:
            #print("x---------", x.shape)
            x=self.se(x)
        x=self.conv3(x)
        x=self.bn3(x)
        x = self.act3(x + shortcut)
        return x


def generate_stage2(ds,block_fun):
    blocks=[]
    for d in ds:
        blocks.append(block_fun(d))
    return blocks


class ConvBnAct(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=1, stride=1, padding=0, dilation=1, groups=1,
                 bias=False, apply_act=True):
        super(ConvBnAct, self).__init__()
        self.conv=nn.Conv2d(in_channels,out_channels,kernel_size,stride,padding,dilation,groups,bias)
        self.bn=nn.BatchNorm2d(out_channels)
        if apply_act:
            self.act=nn.ReLU()
        else:
            self.act=None
    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        if self.act is not None:
            x=self.act(x)
        return x


class UpsamplerBlock (nn.Module):
    def __init__(self, ninput, noutput):
        super().__init__()
        self.conv = nn.ConvTranspose2d(ninput, noutput, 3, stride=2, padding=1, output_padding=1, bias=True)
        self.bn = nn.BatchNorm2d(noutput, eps=1e-3)

    def forward(self, input):
        output = self.conv(input)
        output = self.bn(output)
        return F.relu(output)


class Encoder(nn.Module):
    def __init__(self):
        super().__init__()
        gw = 8
        attention = "cbam"
        ds = [[1],[1,2],[1,4],[1,6],[1,8],[1,10]]+7*[[1,3,6,12]]
        self.stage1 = DBlock(3, 16, [1], gw, 2, attention)
        self.stage2 = nn.Sequential(
            DBlock(16, 64, [1], gw, 2, attention),
            DBlock(64, 64, [1], gw, 1, attention),
            DBlock(64, 64, [1], gw, 1, attention)
        )
        self.stage3 = nn.Sequential(

            DBlock(64, 128, [1], gw, 2, attention),
            DBlock(128, 256, [1], gw, 1, attention),
            *generate_stage2(ds[:-1], lambda d: DBlock(256, 256, d, gw, 1, attention='se'))

        )

        # only for encoder mode:
        # self.output_conv = nn.Conv2d(128, num_classes, 1, stride=1, padding=0, bias=True)

    def forward(self, input, predict=False):

        x1 = self.stage1(input)
        skip_connnection.append(x1)
        x2 = self.stage2(x1)
        skip_connnection.append(x2)
        x3 = self.stage3(x2)
        #print("x1------", x1.shape)
        #print("x2-------", x2.shape)
        #print("x3--------", x3.shape)
        #print(("ecoder----", output.shape))
        return x3
        # return {"4":x1,"8":x2,"16":x3}


class Decoder (nn.Module):
    def __init__(self):
        super().__init__()

        self.head16 = ConvBnAct(256, 64, 1)
        self.head8 = ConvBnAct(64, 64, 1)
        self.head4 = ConvBnAct(16, 8, 1)
        self.conv8 = ConvBnAct(64, 32, 3, 1, 1)
        self.conv4 = ConvBnAct(32 + 8, 16, 3, 1, 1)
        self.up = UpsamplerBlock(16, 16)
        self.classifier = nn.Conv2d(16, 1, 1)

    def forward(self, x):
        # x4, x8, x16 = x["4"], x["8"], x["16"]
        x16 = x
        x8 = skip_connnection[1]
        x4 = skip_connnection[0]
        x16 = self.head16(x16)
        x8 = self.head8(x8)
        x4 = self.head4(x4)
        x16 = F.interpolate(x16, size=x8.shape[-2:], mode='bilinear', align_corners=False)
        x8 = x8 + x16
        x8 = self.conv8(x8)
        x8 = F.interpolate(x8, size=x4.shape[-2:], mode='bilinear', align_corners=False)
        x4 = torch.cat((x8, x4), dim=1)

        x4 = self.conv4(x4)
        # print("x4------", x4.shape)
        x4 = self.up(x4)
        x4 = self.classifier(x4)
        skip_connnection.clear()
        return x4

class RegSeg(nn.Module):
    def __init__(self):  # use encoder to pass pretrained encoder
        super().__init__()
        self.encoder = Encoder()

        self.decoder = Decoder()

    def forward(self, input):
        '''if only_encode:
            return self.encoder.forward(input, predict=True)
        else:'''
        output = self.encoder(input)  # predict=False by default
        # print('encoder------', output.shape)
        return self.decoder.forward(output)



from thop import profile
import torch

# model = ERFNet(1)

# randn_input = torch.randn(1, 3, 256, 256)
"""
FLOPs = 3.607953657G
Params = 2.019027M

"""

#
#randn_input = torch.randn(1, 3, 512, 512)
# """
# FLOPs = 5.383742608G
# Params = 0.832533M
#
# """
#

#img = torch.randn(2, 3, 256, 256)
#model = RegSeg()

#out = model(img)
#print('out-------', out.shape)
#print("out-----------type", type(out))
#flops, params = profile(model, inputs=(randn_input, ))
#print('FLOPs = ' + str(flops/1000**3) + 'G')
#print('Params = ' + str(params/1000**2) + 'M')

