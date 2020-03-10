# Partly taken from here: https://github.com/ycszen/pytorch-segmentation
import torch.nn as nn

class UpscalingConv2d(nn.Module):
    """
    Deconvolution layer.
    """
    def __init__(self, inplanes, n_classes, ks=3, stride=2, padding=1, output_padding=1):
        super(UpscalingConv2d, self).__init__()
        self.tconv2d = nn.ConvTranspose2d(inplanes, n_classes, kernel_size=ks, stride=stride,
                                          padding=padding, output_padding=output_padding)

    def forward(self, x):
        return self.tconv2d(x)

class UpscalingNN(nn.Module):
    """
    Upscaling with nearest neighbour and convolution.
    """
    def __init__(self, in_classes, out_classes):
        super(UpscalingNN, self).__init__()
        self.conv = nn.Conv2d(in_classes, out_classes, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        out = nn.functional.interpolate(x, scale_factor=2, mode='nearest')
        out = self.conv(out)

        return out

class GCN_module(nn.Module):
    """
    Global convolutional network module.
    """
    def __init__(self, inplanes, planes, ks):
        super(GCN_module, self).__init__()
        padding_size = round((ks - 1)/2)
        self.conv_l1 = nn.Conv2d(inplanes, planes, kernel_size=(ks, 1),
                                 padding=(padding_size, 0))
        self.conv_l2 = nn.Conv2d(planes, planes, kernel_size=(1, ks),
                                 padding=(0, padding_size))
        self.conv_r1 = nn.Conv2d(inplanes, planes, kernel_size=(1, ks),
                                 padding=(0, padding_size))
        self.conv_r2 = nn.Conv2d(planes, planes, kernel_size=(ks, 1),
                                 padding=(padding_size, 0))

    def forward(self, x):
        x_l = self.conv_l1(x)
        x_l = self.conv_l2(x_l)

        x_r = self.conv_r1(x)
        x_r = self.conv_r2(x_r)

        x = x_l + x_r

        return x


class Refine(nn.Module):
    def __init__(self, planes):
        super(Refine, self).__init__()
        self.bn = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(planes, planes, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, padding=1)

    def forward(self, x):
        residual = x
        #x = self.bn(x)
        x = self.conv1(x)
        x = self.relu(x)
        #x = self.bn(x)
        x = self.conv2(x)
        #x = self.relu(x)

        out = residual + x
        return out