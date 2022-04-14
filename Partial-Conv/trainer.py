import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from tqdm import tqdm
import matplotlib.pyplot as plt
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import torchvision.datasets as dset
import torchvision.transforms as T
import torchvision.utils

import models
from loss import ReconstructLoss, PerceptualLoss, StyleLoss, TVLoss
from utils.general_utils import parse_config, makedirs
from dataset import DatasetWithMask


class Trainer:
    def __init__(self, config_path: str):
        self.config, self.device, self.log_root = parse_config(config_path)
        makedirs(os.path.join(self.log_root, 'samples'))
        self.train_dataset, self.test_dataset, self.train_loader, self.test_loader, self.img_channels = self._get_data()
        self.G, self.optimizerG, self.Loss_reconstruct, self.Loss_perceptual, self.Loss_style, self.Loss_tv = self._prepare_training()
        self.writer = SummaryWriter(os.path.join(self.log_root, 'tensorboard'))
        self.sample_testid = torch.randint(0, len(self.test_dataset), (12, ))

    def _get_data(self):
        print('==> Getting data...')
        if self.config['dataset'] == 'celeba':
            img_channels = 3
            transforms = T.Compose([T.Resize((128, 128)), T.ToTensor(), T.Normalize(mean=[0.5]*img_channels, std=[0.5]*img_channels)])
            train_dataset = dset.CelebA(root=self.config['dataroot'], split='train', transform=transforms, download=False)
            test_dataset = dset.CelebA(root=self.config['dataroot'], split='test', transform=transforms, download=False)
        else:
            raise ValueError(f'Dataset {self.config["dataset"]} is not available now.')
        train_dataset = DatasetWithMask(train_dataset, mask_fill=0., mask_root=self.config['mask_root'])
        test_dataset = DatasetWithMask(test_dataset, mask_fill=0., mask_root=self.config['mask_root'])
        train_loader = DataLoader(train_dataset, batch_size=self.config['batch_size'], shuffle=True, num_workers=4, pin_memory=True)
        test_loader = DataLoader(test_dataset, batch_size=self.config['batch_size'], num_workers=4, pin_memory=True)
        return train_dataset, test_dataset, train_loader, test_loader, img_channels

    def _prepare_training(self):
        print('==> Preparing training...')
        G = models.Generator(self.img_channels)
        G.to(device=self.device)
        optimizerG = optim.Adam(G.parameters(), lr=self.config['adam']['lr'], betas=self.config['adam']['betas'])
        vgg16 = models.VGG16FeatureExtractor()
        vgg16.to(device=self.device)
        Loss_reconstruct = ReconstructLoss()
        Loss_perceptual = PerceptualLoss(vgg16)
        Loss_style = StyleLoss(vgg16)
        Loss_tv = TVLoss()
        return G, optimizerG, Loss_reconstruct, Loss_perceptual, Loss_style, Loss_tv

    def load_model(self, model_path):
        ckpt = torch.load(model_path, map_location='cpu')
        self.G.load_state_dict(ckpt)
        self.G.to(device=self.device)

    def save_model(self, model_path):
        torch.save(self.G.state_dict(), model_path)

    def train(self):
        print('==> Training...')
        for ep in range(self.config['epochs']):
            self.train_one_epoch(ep)

            if self.config['sample_per_epochs'] and (ep + 1) % self.config['sample_per_epochs'] == 0:
                self.sample_generator(ep, os.path.join(self.log_root, 'samples', f'epoch_{ep}.png'))

            if self.config['save_per_epochs'] and (ep + 1) % self.config['save_per_epochs'] == 0:
                self.save_model(os.path.join(self.log_root, 'ckpt', f'epoch_{ep}.pt'))

        self.save_model(os.path.join(self.log_root, 'model.pt'))
        self.writer.close()

    def train_one_epoch(self, ep):
        self.G.train()
        with tqdm(self.train_loader, desc=f'Epoch {ep}', ncols=120) as pbar:
            for it, (X, img, noise, mask) in enumerate(pbar):
                img = img.to(device=self.device, dtype=torch.float32)
                X = X.to(device=self.device, dtype=torch.float32)
                mask = mask.expand(X.shape).to(device=self.device, dtype=torch.float32)

                inpainted_img = self.G(X, mask)
                loss_hole, loss_valid = self.Loss_reconstruct(inpainted_img, img, mask)
                loss_per = self.Loss_perceptual(inpainted_img, img, mask)
                loss_style = self.Loss_style(inpainted_img, img, mask)
                loss_tv = self.Loss_tv(inpainted_img)
                loss = (self.config['lambda_hole'] * loss_hole +
                        self.config['lambda_valid'] * loss_valid +
                        self.config['lambda_perceptual'] * loss_per +
                        self.config['lambda_style'] * loss_style +
                        self.config['lambda_tv'] * loss_tv)
                self.optimizerG.zero_grad()
                loss.backward()
                self.optimizerG.step()

                self.writer.add_scalar('G/total_loss', loss.item(), it + ep * len(self.train_loader))
                self.writer.add_scalar('G/loss_hole', loss_hole.item(), it + ep * len(self.train_loader))
                self.writer.add_scalar('G/loss_valid', loss_valid.item(), it + ep * len(self.train_loader))
                self.writer.add_scalar('G/loss_per', loss_per.item(), it + ep * len(self.train_loader))
                self.writer.add_scalar('G/loss_style', loss_style.item(), it + ep * len(self.train_loader))
                self.writer.add_scalar('G/loss_tv', loss_tv.item(), it + ep * len(self.train_loader))

    @torch.no_grad()
    def sample_generator(self, ep: int, savepath: str):
        self.G.eval()
        imgs, deteriorated_imgs, masks = [], [], []
        for i in self.sample_testid:
            X, img, noise, mask = self.test_dataset[i]
            imgs.append(img.to(device=self.device))
            deteriorated_imgs.append(X.to(device=self.device))
            masks.append(mask.expand(X.shape).to(device=self.device))
        imgs = torch.stack(imgs, dim=0)
        deteriorated_imgs = torch.stack(deteriorated_imgs, dim=0)
        masks = torch.stack(masks, dim=0)
        inpainted_imgs = self.G(deteriorated_imgs, masks).cpu()
        show_imgs = []
        for i in range(12):
            show_imgs.extend([imgs[i].cpu(), deteriorated_imgs[i].cpu(), inpainted_imgs[i].cpu()])
        show_imgs = torch.stack(show_imgs, dim=0)
        show_imgs = torchvision.utils.make_grid(show_imgs, nrow=6, normalize=True, value_range=(-1, 1))
        fig, ax = plt.subplots(1, 1)
        ax.imshow(torch.permute(show_imgs, [1, 2, 0]))
        ax.set_axis_off()
        ax.set_title(f'Epoch {ep}')
        fig.savefig(savepath, dpi=150, bbox_inches='tight')
        plt.close(fig)
