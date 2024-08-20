import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.fft as fft
import torch.nn.init as init

skip_connections = []


class SwitchNorm2d(nn.Module):
    def __init__(self, num_features, eps=1e-5, momentum=0.9, using_moving_average=True, using_bn=True,
                 last_gamma=False):
        super(SwitchNorm2d, self).__init__()
        self.eps = eps
        self.momentum = momentum
        self.using_moving_average = using_moving_average
        self.using_bn = using_bn
        self.last_gamma = last_gamma
        self.weight = nn.Parameter(torch.ones(1, num_features, 1, 1))
        self.bias = nn.Parameter(torch.zeros(1, num_features, 1, 1))
        if self.using_bn:
            self.mean_weight = nn.Parameter(torch.ones(3))
            self.var_weight = nn.Parameter(torch.ones(3))
        else:
            self.mean_weight = nn.Parameter(torch.ones(2))
            self.var_weight = nn.Parameter(torch.ones(2))
        if self.using_bn:
            self.register_buffer('running_mean', torch.zeros(1, num_features, 1))
            self.register_buffer('running_var', torch.zeros(1, num_features, 1))

        self.reset_parameters()

    def reset_parameters(self):
        if self.using_bn:
            self.running_mean.zero_()
            self.running_var.zero_()
        if self.last_gamma:
            self.weight.data.fill_(0)
        else:
            self.weight.data.fill_(1)
        self.bias.data.zero_()

    def _check_input_dim(self, input):
        if input.dim() != 4:
            raise ValueError('expected 4D input (got {}D input)'
                             .format(input.dim()))

    def forward(self, x):
        self._check_input_dim(x)
        N, C, H, W = x.size()
        x = x.view(N, C, -1)
        mean_in = x.mean(-1, keepdim=True)
        var_in = x.var(-1, keepdim=True)

        mean_ln = mean_in.mean(1, keepdim=True)
        temp = var_in + mean_in ** 2
        var_ln = temp.mean(1, keepdim=True) - mean_ln ** 2

        if self.using_bn:
            if self.training:
                mean_bn = mean_in.mean(0, keepdim=True)
                #print("self.momentum------", self.momentum)
                #print("mean_bn.data---------", mean_bn.data.shape)
                var_bn = temp.mean(0, keepdim=True) - mean_bn ** 2
                if self.using_moving_average:
                    self.running_mean.mul_(self.momentum)
                    self.running_mean.add_((1 - self.momentum) * mean_bn.data)
                    self.running_var.mul_(self.momentum)
                    self.running_var.add_((1 - self.momentum) * var_bn.data)
                else:
                    self.running_mean.add_(mean_bn.data)
                    self.running_var.add_(mean_bn.data ** 2 + var_bn.data)
            else:
                mean_bn = torch.autograd.Variable(self.running_mean)
                var_bn = torch.autograd.Variable(self.running_var)

        softmax = nn.Softmax(0)
        mean_weight = softmax(self.mean_weight)
        var_weight = softmax(self.var_weight)

        if self.using_bn:
            mean = mean_weight[0] * mean_in + mean_weight[1] * mean_ln + mean_weight[2] * mean_bn
            var = var_weight[0] * var_in + var_weight[1] * var_ln + var_weight[2] * var_bn
        else:
            mean = mean_weight[0] * mean_in + mean_weight[1] * mean_ln
            var = var_weight[0] * var_in + var_weight[1] * var_ln

        x = (x-mean) / (var+self.eps).sqrt()
        x = x.view(N, C, H, W)
        return x * self.weight + self.bias

def Common_spectral_pool(images, filter_size):
    assert len(images.shape) == 4
    assert filter_size>=3

    if filter_size%2 == 1:
        n = (filter_size - 1)//2
        top_left = images[:, :, :n+1, :n+1]
        top_right = images[:, :, :n+1, -n:]
        bottom_left = images[:, :, -n:, :n+1]
        bottom_right = images[:, :, -n:, -n:]
        top_combined = torch.cat([top_left, top_right], dim=3)
        bottom_combined = torch.cat([bottom_left, bottom_right], dim=3)
        all_together = torch.cat([top_combined, bottom_combined], dim=2)
    else:
        n = filter_size//2
        top_left = images[:, :, :n, :n]
        top_middle = torch.unsqueeze(0.5**0.5 * (images[:, :, :n, n] + images[:, :, :n, -n]), dim=-1)
        top_right = images[:, :, :n, -(n - 1):]
        middle_left = torch.unsqueeze(0.5 ** 0.5 * (images[:, :, n, :n] + images[:, :, -n, :n]), dim=2)
        middle_middle = torch.unsqueeze(torch.unsqueeze(0.5 * (images[:, :, n, n] + images[:, :, n, -n] + images[:, :, -n, n] + images[:, :, -n, -n]), dim=-1), -1)
        middle_right = torch.unsqueeze(0.5 ** 0.5 * (images[:, :, n, -(n - 1):] + images[:, :, -n, -(n - 1):]), dim=2)
        bottom_left = images[:, :, -(n - 1):, :n]
        bottom_middle = torch.unsqueeze(0.5 ** 0.5 * (images[:, :, -(n - 1):, n] + images[:, :, -(n - 1):, -n]), dim=-1)
        bottom_right = images[:, :, -(n - 1):, -(n - 1):]

        #print("top_left------------", top_left.shape)
        #print("top_middle------------", top_middle.shape)
        #print("top_right--------------", top_right.shape)
        top_combined = torch.cat([top_left, top_middle, top_right], dim=3)
        middle_combined = torch.cat([middle_left, middle_middle, middle_right], dim=3)
        bottom_combined = torch.cat([bottom_left, bottom_middle, bottom_right], dim=3)
        all_together = torch.cat([top_combined, middle_combined, bottom_combined], dim=2)

    return all_together

def _frequency_dropout_mask(filter_size, threshold):
    # 创建一个零填充的掩码
    mask = torch.zeros((filter_size, filter_size), dtype=torch.float32)

    # 设置要保留的频率分量为1
    mask[:threshold, :threshold] = 1.0

    return mask
class Spectral_pooling(nn.Module):
    def __init__(self, filter_size=3, freq_dropout_lower_bound=None, freq_dropout_upper_bound=None):
        super(Spectral_pooling, self).__init__()
        self.filter_size = filter_size
        self.freq_dropout_lower_bound = freq_dropout_lower_bound
        self.freq_dropout_upper_bound = freq_dropout_upper_bound
        self.relu = nn.ReLU()

    def forward(self, x, train_phase=True):
        im_fft = torch.fft.fft2(x.to(torch.complex64))
        im_transformed = Common_spectral_pool(im_fft, self.filter_size)
        if self.freq_dropout_upper_bound is not None and self.freq_dropout_upper_bound is not None:
            def true_fn():
                torch_random_cutoff = torch.rand(1)*(self.freq_dropout_upper_bound-self.freq_dropout_lower_bound)+self.freq_dropout_lower_bound
                dropout_mask = _frequency_dropout_mask(self.filter_size, torch_random_cutoff)
                return im_transformed * dropout_mask

            def false_fn():
                return im_transformed

            im_downsampled = torch.cond(train_phase, true_fn, false_fn)
            im_out = torch.real(torch.fft.ifft2(im_downsampled))

        else:
            im_out = torch.real(torch.fft.ifft2(im_transformed))
        cell_out = self.relu(im_out)#这里没有判断激活函数是否为none
        return cell_out




class Get_spectral_pool_same(nn.Module):
    def __init__(self, size, nin):
        super(Get_spectral_pool_same, self).__init__()
        self.spl =Spectral_pooling(filter_size=size, freq_dropout_lower_bound=None,freq_dropout_upper_bound=None)


    def forward(self, x):
        return self.spl(x)


class Hybrid_pooling_same(nn.Module):#这个可以跟下面那个same_pool合并
    def __init__(self, nin, original_size ,flag, kernel_size, stride):
        super(Hybrid_pooling_same, self).__init__()
        self.max_pooling = nn.MaxPool2d(kernel_size=kernel_size, stride=stride)
        self.conv1x1 = nn.Conv2d(nin*2, nin, kernel_size=1)
        self.upsample = nn.Upsample(size=original_size, mode="bilinear", align_corners=False)
        self.har = Get_spectral_pool_same(original_size, nin)

    def forward(self, x):
        max_pooled = self.upsample(self.max_pooling(x))
        hartley_pooled = self.har(x)
        #print("max_pool---------",max_pooled.shape)
        #print("hartley_pooled-------------", hartley_pooled.shape)
        pooled_features = torch.cat([max_pooled, hartley_pooled], dim=1)
        out = self.conv1x1(pooled_features)

        return out

class Hybrid_pooling_valid(nn.Module):
    def __init__(self, nin, original_size, flag, kernel_size, stride):
        super(Hybrid_pooling_valid, self).__init__()
        self.max_pooling = nn.MaxPool2d(kernel_size=kernel_size, stride=stride)
        self.conv1x1 = nn.Conv2d(nin*2, nin, kernel_size=1)
        if original_size==128 and kernel_size==16 :
            original_size = 16

        self.har = Get_spectral_pool_same(original_size//2, nin)


    def forward(self, x):
        max_pooled = self.max_pooling(x)
        hartley_pooled = self.har(x)
        #print("pooling_valid-------max_pool----", max_pooled.shape)
        #print("pooling_valid-------hartley_pool--", hartley_pooled.shape)
        pooled_features = torch.cat([max_pooled, hartley_pooled], dim=1)
        out = self.conv1x1(pooled_features)
        # print("out---", out.shape)

        return out



class Block(nn.Module):#把bn和relu又换了下位置
    def __init__(self, in_channels, out_channels, size, pooling):
        super(Block, self).__init__()
        self.pooling = pooling
        self.conva = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(out_channels)

        )
        self.convb = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.BatchNorm2d(out_channels)

        )
        self.convc = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1),
            nn.ReLU(),
            nn.BatchNorm2d(out_channels)

        )
        self.convd = nn.Sequential(
            Hybrid_pooling_same(in_channels, size, flag=True, kernel_size=2, stride=2),
            nn.ReLU(),
            nn.BatchNorm2d(in_channels)

        )
        self.conv_2d = nn.Conv2d(3*out_channels+in_channels, out_channels, kernel_size=1)

        self.bn = nn.BatchNorm2d(out_channels)#因为预训练出来的的loss是NAN  所以后来又加了个BN测试
        self.dropout = nn.Dropout(p=0.2)
        self.pool = pooling
        self.pooling = Hybrid_pooling_valid(out_channels, size, flag=False, kernel_size=2, stride=2)#flag控制池化后是否还原图像大小
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
        conv = self.bn(self.conv_2d(cat))
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
        self.pool = Hybrid_pooling_valid(3, 128, flag=False, kernel_size=16, stride=16)
        self.norm = SwitchNorm2d(3)

    def forward(self, x):

        x = self.norm(x)
        input = x
        #print("x----", x.shape)
        conv1 = self.conv1(x)
        #print("conv1---", conv1.shape)
        conv2 = self.conv2(conv1)
        #print("conv2---", conv2.shape)
        conv3 = self.conv3(conv2)
        #print("conv3---", conv3.shape)
        conv4 = self.conv4(conv3)
        #print("conv4---", conv4.shape)
        conv5 = self.conv5(conv4)
        #print("conv5---", conv5.shape)
        input = self.pool(input)
        # print("out---", input.shape)

        out = torch.cat([input, conv5], dim=1)#通道数是899
        # exit(0)
        #print("encoder---", out.shape)
        return out

class Iunet_decoder(nn.Module):
    def __init__(self):
        super(Iunet_decoder, self).__init__()
        self.conv1 = Block(256, 128, 16, pooling=False)
        self.conv2 = Block(128, 64, 32, pooling=False)
        self.conv3 = Block(64, 32, 64, pooling=False)
        self.conv4 = Block(32, 16, 128, pooling=False)
        self.sig = nn.Sigmoid()
        self.c1 = nn.Conv2d(899, 256, kernel_size=1)
        self.end = nn.Conv2d(16, 1, kernel_size=1)
        self.upconv1 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.upconv2 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.upconv3 = nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2)
        self.upconv4 = nn.ConvTranspose2d(32, 16, kernel_size=2, stride=2)
        #self.upconv5 = nn.ConvTranspose2d(64, 16, kernel_size=2, stride=2)



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
    def initialize_weights(self):
        # 迭代模型的所有参数
        for module in self.modules():
            if isinstance(module, nn.Conv2d) or isinstance(module, nn.Linear) or isinstance(module, nn.ConvTranspose2d):
                nn.init.kaiming_normal_(module.weight, mode='fan_in', nonlinearity='relu')
                # 如果是卷积层或全连接层，执行高斯初始化
                #init.normal_(module.weight, mean=0, std=0.01)  # 根据需要调整均值和标准差
                #if module.bias is not None:
                #    init.constant_(module.bias, 0)  # 初始化

#img = torch.randn(3, 3, 128, 128)
#model = Iunet()
#out = model(img)
#print('out-------', out.shape)
