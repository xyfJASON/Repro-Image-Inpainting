from PIL import Image
import os
import random
import torch
from torch.utils.data import Dataset
import torchvision.transforms as T


class DatasetWithMask(Dataset):
    def __init__(self, dataset, mask_fill, mask_root):
        """

        Args:
            dataset: an instance of torch.utils.data.Dataset
            mask_fill: float, or a str of {'random', 'natural'}
            mask_root: path to mask images

        """
        self.dataset = dataset
        self.mask_fill = mask_fill
        self.mask_root = mask_root
        self.mask_paths = os.listdir(self.mask_root)
        self.mask_paths = [os.path.join(os.path.abspath(self.mask_root), p)
                           for p in self.mask_paths if os.path.splitext(p)[1] == '.png']

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, item):
        """

        Returns:
            deteriorated image, ground-truth image, noise, mask

        The deteriorated image can be calculated as: (1 - mask) * img + mask * noise
        """
        img = self.dataset[item]
        if isinstance(img, tuple):
            img = img[0]
        assert (-1 <= img).all() and (img <= 1).all()
        noise = self._get_noise(img)
        mask = self._get_mask(img)
        X = (1 - mask) * img + mask * noise
        return X, img, noise, mask

    def _get_noise(self, img):
        if isinstance(self.mask_fill, float):
            noise = torch.ones_like(img) * self.mask_fill
        elif self.mask_fill == 'random':
            noise = torch.rand_like(img) * 2 - 1
        elif self.mask_fill == 'natural':
            rnd = random.randint(0, len(self.dataset)-1)
            noise = self.dataset[rnd]
            if len(noise) >= 2:
                noise = noise[0]
        else:
            raise ValueError
        return noise

    def _get_mask(self, img):
        rnd = random.randint(0, len(self.mask_paths)-1)
        mask = Image.open(self.mask_paths[rnd])
        mask = T.Resize(img.shape[-2:])(mask)
        mask = T.ToTensor()(mask)  # [0, 1]
        mask[mask < 0.5] = 0
        mask[mask >= 0.5] = 1
        return mask


def _test():
    import torchvision.datasets
    import matplotlib.pyplot as plt

    t = T.Compose([T.Resize((256, 256)), T.ToTensor(), T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])])
    # base_dataset = torchvision.datasets.CIFAR10(root='../data', train=True, transform=t)
    base_dataset = torchvision.datasets.CelebA(root='../data', split='train', transform=t)
    # base_dataset = torchvision.datasets.Places365(root='../data', split='train-standard', download=True)
    mask_dataset = DatasetWithMask(dataset=base_dataset, mask_fill=0., mask_root='../data/Masks/irregular_mask')
    X, img, noise, mask = mask_dataset[1234]
    print(mask)
    plt.imshow(T.ToPILImage()((X + 1) / 2))
    plt.show()


if __name__ == '__main__':
    _test()
