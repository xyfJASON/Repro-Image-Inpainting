import torch
import torch.nn as nn


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight, mean=0, std=0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.constant_(m.weight, 1)
        nn.init.constant_(m.bias, 0)


class Discriminator(nn.Module):
    def __init__(self, img_channels: int) -> None:
        super().__init__()
        self.disc = nn.Sequential(
            nn.Conv2d(img_channels, 64, (4, 4), stride=(2, 2), padding=1, bias=False),  # 32x32
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, 128, (4, 4), stride=(2, 2), padding=1, bias=False),  # 16x16
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(128, 256, (4, 4), stride=(2, 2), padding=1, bias=False),  # 8x8
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(256, 512, (4, 4), stride=(2, 2), padding=1, bias=False),  # 4x4
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(512, 1, (4, 4), stride=(1, 1), padding=0),  # 1x1
            nn.Sigmoid(),
        )
        self.apply(weights_init)

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        return self.disc(X).squeeze()


class Generator(nn.Module):
    def __init__(self, img_channels: int) -> None:
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(img_channels, 64, (4, 4), stride=(2, 2), padding=1, bias=False),  # 64x64
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, 64, (4, 4), stride=(2, 2), padding=1, bias=False),  # 32x32
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, 128, (4, 4), stride=(2, 2), padding=1, bias=False),  # 16x16
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(128, 256, (4, 4), stride=(2, 2), padding=1, bias=False),  # 8x8
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(256, 512, (4, 4), stride=(2, 2), padding=1, bias=False),  # 4x4
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(512, 4000, (4, 4), stride=(1, 1), padding=0, bias=False),  # 1x1
            nn.BatchNorm2d(4000),
            nn.LeakyReLU(0.2, inplace=True),
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(4000, 512, (4, 4), stride=(1, 1), padding=(0, 0)),  # 4x4
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(512, 256, (4, 4), stride=(2, 2), padding=(1, 1)),  # 8x8
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(256, 128, (4, 4), stride=(2, 2), padding=(1, 1)),  # 16x16
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(128, 64, (4, 4), stride=(2, 2), padding=(1, 1)),  # 32x32
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(64, img_channels, (4, 4), stride=(2, 2), padding=(1, 1)),  # 64x64
            nn.Tanh(),
        )
        self.apply(weights_init)

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        X = self.encoder(X)
        X = self.decoder(X)
        return X


if __name__ == '__main__':
    G = Generator(3)
    D = Discriminator(3)
    z = torch.randn(10, 3, 128, 128)
    fakeX = G(z)
    score = D(fakeX)
    print(fakeX.shape)
    print(score.shape)
