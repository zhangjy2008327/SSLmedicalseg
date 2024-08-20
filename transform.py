
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from torch.autograd import Variable
from einops import rearrange, repeat
from einops.layers.torch import Rearrange


def relative_pos_dis(height=32, weight=32, sita=0.9):
    coords_h = torch.arange(height)
    coords_w = torch.arange(weight)
    coords = torch.stack(torch.meshgrid([coords_h, coords_w]))  # 2, Wh, Ww # 0 is 32 * 32 for h, 1 is 32 * 32 for w
    coords_flatten = torch.flatten(coords, 1)  # 2, Wh*Ww
    relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]  # 2, Wh*Ww, Wh*Ww
    relative_coords = relative_coords.permute(1, 2, 0).contiguous()  # Wh*Ww, Wh*Ww, 2
    dis = (relative_coords[:, :, 0].float()/height) ** 2 + (relative_coords[:, :, 1].float()/weight) ** 2
    return dis


class CNNAttention(nn.Module):
    def __init__(self, dim, heads=8, dim_head=64, dropout=0., num_patches=256):
        super().__init__()
        #在多头自注意力机制中，输入特征首先会被分成多个头，然后每个头会独立计算自注意力，并产生一个输出。这些输出在每个头内部具有相同的维度，由dim_head决定，而inner_dim表示每个头输出的总维度
        inner_dim = dim_head * heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head ** -0.5
        self.num_patches = num_patches

        #self.to_qkv = nn.Conv2d(dim, inner_dim * 3, kernel_size=1, padding=0, bias=False)
        self.to_qkv = nn.Conv2d(dim, inner_dim * 3, kernel_size=3, padding=1, bias=False)
        self.dis = relative_pos_dis(math.sqrt(num_patches), math.sqrt(num_patches), sita=0.9).cuda()  #这个后面本来加了.cuda
        #print("!!!!!!!!", self.dis)
        #print("@@@@@@@@@@@@@@", type(self.dis))
        self.headsita = nn.Parameter(torch.randn(heads), requires_grad=True)#创建一个可训练的参数，并初始化其值为服从标准正态分布的随机数，requires_grad=True表示在反向传播过程中会计算并更新与该参数相关的梯度，用于优化模型
        self.sig = nn.Sigmoid()

        self.to_out = nn.Sequential(
            nn.Conv2d(inner_dim, dim, kernel_size=1, padding=0, bias=False),
            nn.BatchNorm2d(dim), # inner_dim
            nn.ReLU(inplace=True),
        ) if project_out else nn.Identity()

    def forward(self, x, mode="train", smooth=1e-4):
        # print("x----------", x.shape)
        qkv = self.to_qkv(x).chunk(3, dim=1)#将输入数据x转换成三个不同的张量q,k,v      .

        q, k, v = map(lambda t: rearrange(t, 'b (g d) h w -> b g (h w) d', g=self.heads), qkv)
        attn = torch.matmul(q, k.transpose(-1, -2)) # b g n n   k.transpose(-1,-2)是将键张量k在最后两个维度上进行转置，从(b,g,n,d)转换成(b,g,d,n)  g表示头数，n表示序列长度，d表示通道数
        qk_norm = torch.sqrt(torch.sum(q ** 2, dim=-1)+smooth)[:, :, :, None] * torch.sqrt(torch.sum(k ** 2, dim=-1)+smooth)[:, :, None, :] + smooth
        attn = attn/qk_norm
        #attentionheatmap_visual2(attn, self.sig(self.headsita), out_dir='./Visualization/ACDC/SETR_plane2', value=1)
        #factor = 1/(2*(self.sig(self.headsita)+0.01)**2) # h
        factor = 1/(2*(self.sig(self.headsita)*(0.4-0.003)+0.003)**2) # af3 + limited setting this, or using the above line code
        # print("factor------", factor.shape)
        #factor = factor.to(self.dis.device)
        dis = factor[:, None, None]*self.dis[None, :, :] # g n n
        # print("dis00000000", self.dis[None, :, :].shape)
        # print("dis1111-----", dis.shape)
        dis = torch.exp(-dis)#将张量dis中的每个元素取自然指数的负数幂，即e的-dis次方
        # print("dis2222-----", dis.shape)
        dis = dis/torch.sum(dis, dim=-1)[:, :, None]
        #attentionheatmap_visual2(dis[None, :, :, :], self.sig(self.headsita), out_dir='./Visualization/ACDC/dis', value=0.003)

        # print("attn----", attn.shape)
        # print("dis------", dis.shape)
        #attn = attn.to(self.dis.device)
        attn = attn * dis[None, :, :, :]
        #attn = attn.to(torch.double)
        #attentionheatmap_visual2(attn, self.sig(self.headsita), out_dir='./Visualization/ACDC/after', value=0.003)
        #attentionheatmap_visual(attn, out_dir='./Visualization/attention_af3/')
        #print("v-----------", type(v))
        #v = v.to(self.dis.device)
        #print("attn------------", type(attn))
        #v = v.to(torch.double)
        out = torch.matmul(attn, v)
        out = rearrange(out, 'b g (h w) d -> b (g d) h w', h=x.shape[2])
        if mode=="train":
            return self.to_out(out)
        else:
            return self.to_out(out), attn


class CNNFeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout=0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(dim, hidden_dim, kernel_size=1, padding=0, bias=False),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden_dim, dim, kernel_size=1, padding=0, bias=False),
            nn.BatchNorm2d(dim),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.net(x)


class CNNTransformer_record(nn.Module):
    def __init__(self, dim=256, depth=12, heads=8, dim_head=64, mlp_dim=512, dropout=0.1, num_patches=256):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                CNNAttention(dim, heads=heads, dim_head=dim_head, dropout=dropout, num_patches=num_patches),
                CNNFeedForward(dim, mlp_dim, dropout=dropout)
            ]))

    def forward(self, x):
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x
        return x

    def infere(self, x):
        ftokens, attmaps = [], []
        for attn, ff in self.layers:
            ax, amap = attn(x, mode="record")
            x = ax + x
            x = ff(x) + x
            ftokens.append(rearrange(x, 'b c h w -> b (h w) c'))
            attmaps.append(amap)
        return x, ftokens, attmaps