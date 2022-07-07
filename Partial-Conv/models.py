import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='leaky_relu')
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)
    elif classname.find('BatchNorm') != -1:
        nn.init.constant_(m.weight, 1)
        nn.init.constant_(m.bias, 0)


class PartialConv2d(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, stride: int, padding: int,
                 norm: str = None, activation: str = None):
        super().__init__()
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, bias=False)
        self.bias = nn.Parameter(torch.zeros((out_channels, )))
        self.mask_conv_weight = torch.ones(out_channels, in_channels, kernel_size, kernel_size)
        if norm == 'bn':
            self.norm = nn.BatchNorm2d(out_channels)
        else:
            self.norm = None
        if activation == 'relu':
            self.activation = nn.ReLU()
        elif activation == 'leakyrelu':
            self.activation = nn.LeakyReLU(0.2)
        elif activation == 'tanh':
            self.activation = nn.Tanh()
        elif activation == 'sigmoid':
            self.activation = nn.Sigmoid()
        else:
            self.activation = None
        self.conv.apply(weights_init)

    def forward(self, X: torch.Tensor, mask: torch.Tensor):
        """ Note that 1 in mask denote invalid pixels.

        Args:
            X: Tensor[bs, C, H, W]
            mask: Tensor[bs, 1, H, W]

        """
        mask = 1. - mask  # now 1 is valid pixel and 0 is invalid pixel
        self.mask_conv_weight = self.mask_conv_weight.to(device=mask.device)
        with torch.no_grad():
            mask_conv = F.conv2d(mask, self.mask_conv_weight, stride=self.stride, padding=self.padding)
        invalid_pos = mask_conv == 0

        scale = self.kernel_size * self.kernel_size / (mask_conv + 1e-8)
        scale.masked_fill_(invalid_pos, 0.)

        X = self.conv(X * mask)
        X = X * scale + self.bias.view(1, -1, 1, 1)
        X.masked_fill_(invalid_pos, 0.)
        if self.norm:
            X = self.norm(X)
        if self.activation:
            X = self.activation(X)

        new_mask = torch.ones_like(mask_conv)
        new_mask.masked_fill_(invalid_pos, 0.)
        new_mask = 1 - new_mask  # 1 is invalid pixel and 0 is valid pixel

        return X, new_mask


class TransposePartialConv2d(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, stride: int, padding: int,
                 scale_factor: float = 2, norm: str = None, activation: str = None):
        super().__init__()
        self.partial_conv = PartialConv2d(in_channels, out_channels, kernel_size, stride, padding, norm, activation)
        self.scale_factor = scale_factor

    def forward(self, X: torch.Tensor, mask: torch.Tensor, X_lateral: torch.Tensor, mask_lateral: torch.Tensor):
        X = F.interpolate(X, scale_factor=self.scale_factor, mode='nearest')
        mask = F.interpolate(mask, scale_factor=self.scale_factor, mode='nearest')
        X, mask = self.partial_conv(torch.cat([X, X_lateral], dim=1), torch.cat([mask, mask_lateral], dim=1))
        return X, mask


class Generator(nn.Module):
    def __init__(self, img_channels: int, n_layer: int = 7) -> None:
        super().__init__()
        self.n_layer = n_layer

        self.encoder1 = PartialConv2d(img_channels, 64, kernel_size=7, stride=2, padding=3, activation='relu')     # 1/2
        self.encoder2 = PartialConv2d(64, 128, kernel_size=5, stride=2, padding=2, norm='bn', activation='relu')   # 1/4
        self.encoder3 = PartialConv2d(128, 256, kernel_size=5, stride=2, padding=2, norm='bn', activation='relu')  # 1/8
        self.encoder4 = PartialConv2d(256, 512, kernel_size=3, stride=2, padding=1, norm='bn', activation='relu')  # 1/16
        for i in range(5, n_layer+1):
            setattr(self, f'encoder{i}', PartialConv2d(512, 512, kernel_size=3, stride=2, padding=1, norm='bn', activation='relu'))
            setattr(self, f'decoder{i}', TransposePartialConv2d(512 + 512, 512, kernel_size=3, stride=1, padding=1, scale_factor=2, norm='bn', activation='leakyrelu'))
        self.decoder4 = TransposePartialConv2d(512 + 256, 256, kernel_size=3, stride=1, padding=1, scale_factor=2, norm='bn', activation='leakyrelu')   # 1/8
        self.decoder3 = TransposePartialConv2d(256 + 128, 128, kernel_size=3, stride=1, padding=1, scale_factor=2, norm='bn', activation='leakyrelu')   # 1/4
        self.decoder2 = TransposePartialConv2d(128 + 64, 64, kernel_size=3, stride=1, padding=1, scale_factor=2, norm='bn', activation='leakyrelu')     # 1/2
        self.decoder1 = TransposePartialConv2d(64 + img_channels, img_channels, kernel_size=3, stride=1, padding=1, scale_factor=2, activation='tanh')  # 1/1

    def forward(self, X: torch.Tensor, mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        X_lateral, mask_lateral = [X], [mask]
        for i in range(1, self.n_layer+1):
            layer = getattr(self, f'encoder{i}')
            X, mask = layer(X, mask)
            X_lateral.append(X)
            mask_lateral.append(mask)
        for i in range(self.n_layer, 0, -1):
            layer = getattr(self, f'decoder{i}')
            X, mask = layer(X, mask, X_lateral[i-1], mask_lateral[i-1])
        return X


class VGG16FeatureExtractor(nn.Module):
    def __init__(self):
        super().__init__()
        vgg16 = models.vgg16(pretrained=True)
        self.pool1 = nn.Sequential(*vgg16.features[:5])
        self.pool2 = nn.Sequential(*vgg16.features[5:10])
        self.pool3 = nn.Sequential(*vgg16.features[10:17])
        for i in range(1, 4):
            for param in getattr(self, f'pool{i}').parameters():
                param.requires_grad_(False)

    def forward(self, X: torch.Tensor):
        pool1 = self.pool1(X)
        pool2 = self.pool2(pool1)
        pool3 = self.pool3(pool2)
        return pool1, pool2, pool3


def _test():
    G = Generator(img_channels=3, n_layer=7)
    x = torch.randn(10, 3, 128, 128)
    mask = torch.randint(0, 2, size=(10, 3, 128, 128))
    fakeX = G(x, mask)
    print(fakeX.shape)

    VGG16FeatureExtractor()


if __name__ == '__main__':
    _test()
