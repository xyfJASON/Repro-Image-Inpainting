import os
import math
import random
from PIL import Image, ImageDraw, ImageFilter

import torch
from torch.utils.data import Dataset
import torchvision.transforms as T


class DatasetWithMask(Dataset):
    def __init__(self, dataset, mask_type, mask_fill, smooth_radius: float = 0,
                 n_rectangles: int = 5,
                 n_brushes: int = 15, max_angle: int = 4, max_len: int = 40, max_width: int = 20, max_turns: int = 8):
        """

        Args:
            dataset: an instance of torch.utils.data.Dataset
            mask_type:
              - 'center': a rectangle in the center
              - 'rectangles': rectangles at random position with random height and weight
              - 'brushes': brushes mask
              - a directory path: directory containing mask images
            mask_fill:
              - a float in [-1, 1]
              - 'random'
              - an instance of Dataset
            smooth_radius: gaussian smooth at mask boundary

            n_rectangles: number of rectangles, only for mask_type == 'rectangles'

            n_brushes: max number of brushed, only for mask_type == 'brushes'
            max_angle: max angle to turn, only for mask_type == 'brushes'
            max_len: max length of a brush segment, only for mask_type == 'brushes'
            max_width: max width of a brush stroke, only for mask_type == 'brushes'
            max_turns: max number of turns in a brush stroke, only for mask_type == 'brushes'

        """
        self.dataset = dataset
        self.mask_type = mask_type
        self.mask_fill = mask_fill
        self.smooth_radius = smooth_radius
        self.n_rectangles = n_rectangles
        self.n_brushes = n_brushes
        self.max_angle = max_angle
        self.max_len = max_len
        self.max_width = max_width
        self.max_turns = max_turns

        if os.path.exists(mask_type):
            img_format = ['.png', '.jpg', '.jpeg']
            self.mask_paths = os.listdir(self.mask_type)
            self.mask_paths = [os.path.join(os.path.realpath(self.mask_type), p)
                               for p in self.mask_paths if os.path.splitext(p)[1] in img_format]

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, item):
        """

        Returns:
            deteriorated image, ground-truth image, noise, mask

        The deteriorated image can be calculated as: (1 - mask) * img + mask * noise
        """
        img = self.dataset[item]
        if isinstance(img, tuple) or isinstance(img, list):
            img = img[0]
        assert (-1 <= img).all() and (img <= 1).all()
        mask = self._get_mask(img.shape[-2:])
        noise = self._get_noise(img.shape[-2:])
        X = (1 - mask) * img + mask * noise
        return X, img, noise, mask

    def _get_noise(self, shape: tuple[int, int]):
        if isinstance(self.mask_fill, float):
            noise = torch.ones(shape) * self.mask_fill
        elif self.mask_fill == 'random':
            noise = torch.rand(shape) * 2 - 1
        elif isinstance(self.mask_fill, Dataset):
            rnd = random.randint(0, len(self.dataset) - 1)
            noise = self.dataset[rnd]
            if isinstance(noise, tuple) or isinstance(noise, list):
                noise = noise[0]
            noise = T.Resize(shape)(noise)
        else:
            raise ValueError
        return noise

    def _get_mask(self, shape: tuple[int, int]):
        if self.mask_type == 'center':
            mask = self._get_mask_center(shape)
        elif self.mask_type == 'rectangles':
            mask = self._get_mask_rectangles(shape)
        elif self.mask_type == 'brushes':
            mask = self._get_mask_brushes(shape)
        elif os.path.exists(self.mask_type):
            mask = self._get_mask_directory(shape)
        else:
            raise ValueError
        assert torch.bitwise_or(mask == 0, mask == 1).all()  # check if mask is binary
        if self.smooth_radius:
            mask = T.ToPILImage()(mask)
            mask = mask.filter(ImageFilter.GaussianBlur(radius=self.smooth_radius))
            mask = T.ToTensor()(mask)
        return mask

    @staticmethod
    def _get_mask_center(shape: tuple[int, int]):
        mask = torch.zeros((1, *shape))
        mask[:, shape[0]//4:shape[0]*3//4, shape[1]//4:shape[1]*3//4] = 1
        return mask

    def _get_mask_rectangles(self, shape: tuple[int, int]):
        mask = torch.zeros((1, *shape))
        n_rec = random.randint(1, self.n_rectangles)
        area = shape[0] * shape[1] // (4 + 2 * n_rec)
        for n in range(n_rec):
            if area == 0:
                break
            if random.randint(0, 1) == 0:
                h = random.randint(int(0.05 * shape[0]), int(0.6 * shape[0]))
                w = random.randint(int(0.02 * shape[1]), min(int(0.001 * shape[0] * shape[1]), area//h))
            else:
                w = random.randint(int(0.05 * shape[1]), int(0.6 * shape[1]))
                h = random.randint(int(0.02 * shape[0]), min(int(0.001 * shape[0] * shape[1]), area//w))
            mini = random.randint(0, shape[0] - h)
            minj = random.randint(0, shape[1] - w)
            mask[:, mini:mini+h, minj:minj+w] = 1
            area = max(0, area - h * w // 2)
        return mask

    def _get_mask_brushes(self, shape: tuple[int, int]):
        mask = Image.new('L', shape, 0)
        n_brushes = random.randint(1, self.n_brushes)
        for i in range(n_brushes):
            xy = []
            width = 5 + random.randint(0, self.max_width - 5)
            xy.append((random.randint(0, shape[1]-1), random.randint(0, shape[0]-1)))
            for j in range(random.randint(1, self.max_turns + 1)):
                angle = random.random() * self.max_angle
                length = 10 + random.randint(0, self.max_len - 10)
                new_x = int(xy[-1][0] + length * math.sin(angle))
                new_y = int(xy[-1][1] + length * math.cos(angle))
                xy.append((new_x, new_y))
            draw = ImageDraw.Draw(mask)
            draw.line(xy, fill=255, width=width)
            for v in xy:
                draw.ellipse((v[0] - width // 2, v[1] - width // 2, v[0] + width // 2, v[1] + width // 2), fill=255)
        mask = T.ToTensor()(mask)
        return mask

    def _get_mask_directory(self, shape: tuple[int, int]):
        mask = Image.open(random.choice(self.mask_paths))
        mask = T.Resize(shape)(mask)
        mask = T.ToTensor()(mask)
        mask[mask < 0.5] = 0
        mask[mask >= 0.5] = 1
        return mask


def _test():
    import torchvision.datasets
    import matplotlib.pyplot as plt

    t = T.Compose([T.Resize((256, 256)), T.ToTensor(), T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])])
    base_dataset = torchvision.datasets.CelebA(root='../data', split='train', transform=t)
    mask_dataset = DatasetWithMask(dataset=base_dataset, mask_type='brushes', mask_fill=base_dataset, smooth_radius=1)
    X, img, noise, img = mask_dataset[123]
    plt.imshow(T.ToPILImage()((X + 1) / 2))
    plt.show()


if __name__ == '__main__':
    _test()
