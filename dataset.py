import torch
from torch.utils.data import Dataset


def make_binary_mask(img_size: tuple[int, int], mask_shape: str,
                     width_range: tuple[int, int] = None, height_range: tuple[int, int] = None, num_rectangles: int = None) -> torch.Tensor:
    """ Make various kinds of masks. The masked area is denoted as 1, and other part is denoted as 0.

    Args:
        img_size: (height, width) of the input image.
        mask_shape: shape of the mask, one of {'center', 'rectangle', 'strokes'}.

            - 'center': The masked area is a single rectangle at the center of the input image.
            - 'rectangles': The masked area is mlutiple rectangles at random position with random width and height
                chosen from width_range and height_range.
            - 'strokes': The masked area simulates strokes.

        width_range: only used when mask_shape is 'rectangles'.
            The width of each rectangle is randomly chosen from width_range.
        height_range: only used when mask_shape is 'rectangles'.
            The height of each rectangle is randomly chosen from height_range.
        num_rectangles: only used when mask_shape is 'rectangles'.
            Number of rectangles.

    Returns:
        A binary tensor representing mask. 1 implies masked area and 0 implies unmasked area.
    """
    assert mask_shape in ['center', 'rectangles', 'strokes']
    mask = torch.zeros(img_size)
    if mask_shape == 'center':
        u, d = img_size[0] // 4, img_size[0] // 4 * 3
        l, r = img_size[1] // 4, img_size[1] // 4 * 3
        mask[u:d, l:r] = 1
    elif mask_shape == 'rectangles':
        assert width_range is not None and height_range is not None
        assert num_rectangles is not None and num_rectangles >= 1
        for i in range(num_rectangles):
            w = torch.randint(width_range[0], width_range[1], (1,)).item()
            h = torch.randint(height_range[0], height_range[1], (1,)).item()
            mini = torch.randint(0, img_size[0] - h, (1,)).item()
            minj = torch.randint(0, img_size[1] - w, (1,)).item()
            mask[mini:mini+h, minj:minj+w] = 1
    else:  # strokes
        pass
    return mask.unsqueeze(0)


class DatasetWithMask(Dataset):
    def __init__(self, dataset, mask_shape: str, mask_fill: float or torch.Tensor or str,
                 width_range: tuple[int, int] = None, height_range: tuple[int, int] = None, num_rectangles: int = None):
        assert isinstance(mask_fill, float) or isinstance(mask_fill, torch.Tensor) or mask_fill in ['random', 'natural']
        if isinstance(mask_fill, float):
            assert -1. <= mask_fill <= 1.
        if isinstance(mask_fill, torch.Tensor):
            assert (-1. <= mask_fill <= 1.).all()

        self.dataset = dataset
        self.mask_shape = mask_shape
        self.mask_fill = mask_fill
        self.width_range = width_range
        self.height_range = height_range
        self.num_rectangles = num_rectangles

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, item):
        """
        Returns:
            img, mask_img, mask

        The deteriorated image can be calculated as: (1 - mask) * img + mask * mask_img
        """
        img = self.dataset[item]
        if len(img) >= 2:
            img = img[0]
        if isinstance(self.mask_fill, float) or isinstance(self.mask_fill, torch.Tensor):
            mask_img = torch.ones_like(img) * self.mask_fill
        elif self.mask_fill == 'random':
            mask_img = torch.rand_like(img)
        else:  # natural
            rnd = torch.randint(0, len(self.dataset), (1,))
            mask_img = self.dataset[rnd]
            if len(mask_img) >= 2:
                mask_img = mask_img[0]
        mask = make_binary_mask(img_size=(img.shape[1], img.shape[2]),
                                mask_shape=self.mask_shape,
                                width_range=self.width_range,
                                height_range=self.height_range,
                                num_rectangles=self.num_rectangles)
        return img, mask_img, mask


def _test():
    import torchvision.transforms as T
    import torchvision.datasets
    import matplotlib.pyplot as plt

    t = T.Compose([T.Resize((256, 256)), T.ToTensor()])
    # base_dataset = torchvision.datasets.CIFAR10(root='../data', train=True, transform=t)
    base_dataset = torchvision.datasets.CelebA(root='../data', split='train', transform=t)
    # base_dataset = torchvision.datasets.Places365(root='../data', split='train-standard', download=True)
    mask_dataset = DatasetWithMask(dataset=base_dataset, mask_shape='rectangles', mask_fill='natural',
                                   width_range=(96, 128), height_range=(96, 128), num_rectangles=2)
    img, mask_img, mask = mask_dataset[0]
    det_img = mask * mask_img + (1 - mask) * img
    plt.imshow(T.ToPILImage()(det_img))
    plt.show()


if __name__ == '__main__':
    _test()
