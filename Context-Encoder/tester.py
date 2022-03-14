import os
from PIL import Image
from tqdm import tqdm
import torch
from torch.utils.data import DataLoader
import torchvision.transforms as T
import torchvision.datasets as dset
from torchvision.utils import save_image

import models
from dataset import DatasetWithMask
from utils import metrics


class Tester:
    def __init__(self, model_path: str, use_gpu: bool = True, img_channels: int = 3):
        ckpt = torch.load(model_path, map_location='cpu')
        self.device = torch.device('cuda' if torch.cuda.is_available() and use_gpu else 'cpu')
        print('using device:', self.device)
        self.img_channels = img_channels

        self.G = models.Generator(img_channels=self.img_channels)
        self.G.load_state_dict(ckpt['G'])
        self.G.to(device=self.device)

    @torch.no_grad()
    def evaluate(self, dataset: str, dataroot: str, mask_root: str, batch_size: int):
        transforms = T.Compose([T.Resize((128, 128)), T.ToTensor(), T.Normalize(mean=[0.5]*self.img_channels, std=[0.5]*self.img_channels)])
        if dataset == 'celeba':
            test_dataset = dset.CelebA(root=dataroot, split='test', transform=transforms, download=False)
        else:
            raise ValueError(f'Dataset {dataset} is not available now.')
        test_dataset = DatasetWithMask(test_dataset, mask_fill=0., mask_root=mask_root)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, num_workers=4, pin_memory=True)

        self.G.eval()
        mse, psnr, ssim = 0, 0, 0
        total = 0
        for X, img, mask_img, mask in tqdm(test_loader, desc='Evaluating', ncols=120, leave=False, colour='yellow'):
            bs = X.shape[0]
            img = img.to(device=self.device, dtype=torch.float32)
            X = X.to(device=self.device, dtype=torch.float32)
            inpainted_img = self.G(X)
            mse += metrics.MSE((img[:, :, 32:96, 32:96] + 1) / 2, (inpainted_img + 1) / 2, batch=True) * bs
            psnr += metrics.PSNR((img[:, :, 32:96, 32:96] + 1) / 2, (inpainted_img + 1) / 2, batch=True) * bs
            ssim += metrics.SSIM((img[:, :, 32:96, 32:96] + 1) / 2, (inpainted_img + 1) / 2, batch=True) * bs
            total += bs
        return mse / total, psnr / total, ssim / total

    @torch.no_grad()
    def predict(self, img_path, save_path):
        assert os.path.exists(img_path) and os.path.isfile(img_path)
        self.G.eval()
        transforms = T.Compose([T.Resize((128, 128)), T.ToTensor(), T.Normalize(mean=[0.5]*self.img_channels, std=[0.5]*self.img_channels)])
        img = Image.open(img_path)
        img = transforms(img).unsqueeze(0)
        img = img.to(device=self.device)
        img[:, :, 32:96, 32:96] = self.G(img)
        save_image(img, save_path, normalize=True, value_range=(-1, 1))
