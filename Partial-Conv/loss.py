import torch
import torch.nn as nn


class ReconstructLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.mae = nn.L1Loss()

    def forward(self, fake_img: torch.Tensor, real_img: torch.Tensor, mask: torch.Tensor):
        L_hole = self.mae(mask * fake_img, mask * real_img)
        L_valid = self.mae((1 - mask) * fake_img, (1 - mask) * real_img)
        return L_hole, L_valid


class PerceptualLoss(nn.Module):
    def __init__(self, feature_extractor: nn.Module, n_features: int = 3):
        super().__init__()
        self.mae = nn.L1Loss()
        self.feature_extractor = feature_extractor
        self.n_features = n_features

    def forward(self, fake_img: torch.Tensor, real_img: torch.Tensor, mask: torch.Tensor):
        comp_img = fake_img * mask + real_img * (1 - mask)
        fake_features = self.feature_extractor(fake_img)
        real_features = self.feature_extractor(real_img)
        comp_features = self.feature_extractor(comp_img)
        L_perceptual = 0.
        for i in range(self.n_features):
            L_perceptual += self.mae(fake_features[i], real_features[i]) + self.mae(comp_features[i], real_features[i])
        return L_perceptual


class StyleLoss(nn.Module):
    def __init__(self, feature_extractor: nn.Module, n_features: int = 3):
        super().__init__()
        self.mae = nn.L1Loss()
        self.feature_extractor = feature_extractor
        self.n_features = n_features

    def forward(self, fake_img: torch.Tensor, real_img: torch.Tensor, mask: torch.Tensor):
        comp_img = fake_img * mask + real_img * (1 - mask)
        fake_features = self.feature_extractor(fake_img)
        real_features = self.feature_extractor(real_img)
        comp_features = self.feature_extractor(comp_img)

        L_style = 0.
        for i in range(self.n_features):
            bs, C, H, W = fake_features[i].shape
            fake_feature = fake_features[i].flatten(start_dim=-2)
            real_feature = real_features[i].flatten(start_dim=-2)
            comp_feature = comp_features[i].flatten(start_dim=-2)
            assert fake_feature.shape == (bs, C, H * W)
            fake_gram = torch.bmm(fake_feature, fake_feature.transpose(1, 2)) / (C * H * W)
            real_gram = torch.bmm(real_feature, real_feature.transpose(1, 2)) / (C * H * W)
            comp_gram = torch.bmm(comp_feature, comp_feature.transpose(1, 2)) / (C * H * W)
            assert fake_gram.shape == (bs, C, C)
            L_style += self.mae(fake_gram, real_gram) + self.mae(comp_gram, real_gram)
        return L_style


class TVLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.mae = nn.L1Loss()

    def forward(self, img: torch.Tensor):
        L_tv = self.mae(img[:, :, 1:, :], img[:, :, :-1, :]) + self.mae(img[:, :, :, 1:], img[:, :, :, :-1])
        return L_tv


def _test():
    from models import VGG16FeatureExtractor
    fake_img = torch.rand((10, 3, 128, 128))
    real_img = torch.rand((10, 3, 128, 128))
    mask = torch.randint(0, 2, (10, 3, 128, 128))
    L_reconstruct = ReconstructLoss()(fake_img, real_img, mask)
    L_perceptual = PerceptualLoss(VGG16FeatureExtractor())(fake_img, real_img, mask)
    L_style = StyleLoss(VGG16FeatureExtractor())(fake_img, real_img, mask)
    L_tv = TVLoss()(fake_img)
    print(L_reconstruct, L_perceptual, L_style, L_tv)


if __name__ == '__main__':
    _test()
