import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from tqdm import tqdm
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import torchvision.datasets as dset
import torchvision.transforms as T
import torchvision.utils

import models
from utils.general_utils import parse_config
from dataset import DatasetWithMask


class Trainer:
    def __init__(self, config_path: str):
        self.config, self.device, self.log_root = parse_config(config_path)
        if not os.path.exists(os.path.join(self.log_root, 'samples')):
            os.makedirs(os.path.join(self.log_root, 'samples'))
        self.train_dataset, self.test_dataset, self.train_loader, self.test_loader, self.img_channels = self._get_data()
        self.G, self.D, self.optimizerG, self.optimizerD, self.BCE, self.MSE = self._prepare_training()
        self.writer = SummaryWriter(os.path.join(self.log_root, 'tensorboard'))
        self.sample_testid = torch.randint(0, len(self.test_dataset), (12, ))

    def _get_data(self):
        print('==> Getting data...')
        transforms = T.Compose([T.Resize((128, 128)), T.ToTensor(), T.Normalize(mean=[0.5]*3, std=[0.5]*3)])
        if self.config['dataset'] == 'celeba':
            train_dataset = dset.CelebA(root=self.config['dataroot'], split='train', transform=transforms, download=False)
            test_dataset = dset.CelebA(root=self.config['dataroot'], split='test', transform=transforms, download=False)
            img_channels = 3
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
        D = models.Discriminator(self.img_channels)
        G.to(device=self.device)
        D.to(device=self.device)
        optimizerG = optim.Adam(G.parameters(), lr=self.config['adam']['lr'], betas=self.config['adam']['betas'])
        optimizerD = optim.Adam(D.parameters(), lr=self.config['adam']['lr'], betas=self.config['adam']['betas'])
        BCE = nn.BCELoss()
        MSE = nn.MSELoss()
        return G, D, optimizerG, optimizerD, BCE, MSE

    def load_model(self, model_path):
        ckpt = torch.load(model_path, map_location='cpu')
        self.G.load_state_dict(ckpt['G'])
        self.D.load_state_dict(ckpt['D'])
        self.G.to(device=self.device)
        self.D.to(device=self.device)

    def save_model(self, model_path):
        torch.save({'G': self.G.state_dict(), 'D': self.D.state_dict()}, model_path)

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
        self.D.train()
        with tqdm(self.train_loader, desc=f'Epoch {ep}', ncols=120) as pbar:
            for it, (X, img, noise, mask) in enumerate(pbar):
                img = img.to(device=self.device, dtype=torch.float32)
                X = X.to(device=self.device, dtype=torch.float32)

                # --------- train discriminator --------- #
                # min -(E[log(D(X))] + E[log(1-D(G(X')))])
                inpainted_img = self.G(X).detach()
                d_real, d_fake = self.D(img[:, :, 32:96, 32:96]), self.D(inpainted_img)
                lossD = self.BCE(d_real, torch.ones_like(d_real)) + self.BCE(d_fake, torch.zeros_like(d_fake))
                self.optimizerD.zero_grad()
                lossD.backward()
                self.optimizerD.step()
                self.writer.add_scalar('D/loss', lossD.item(), it + ep * len(self.train_loader))

                # --------- train generator --------- #
                # min lambda_adv * E[-log(D(G(X'))] + lambda_rec * E[L2(G(X'),X)]
                inpainted_img = self.G(X)
                d_fake = self.D(inpainted_img)
                loss_adv = self.BCE(d_fake, torch.ones_like(d_fake))
                loss_rec = self.MSE(inpainted_img, img[:, :, 32:96, 32:96])
                lossG = self.config['lambda_rec'] * loss_rec + self.config['lambda_adv'] * loss_adv
                self.optimizerG.zero_grad()
                lossG.backward()
                self.optimizerG.step()
                self.writer.add_scalar('G/total_loss', lossG.item(), it + ep * len(self.train_loader))
                self.writer.add_scalar('G/adv_loss', loss_adv.item(), it + ep * len(self.train_loader))
                self.writer.add_scalar('G/rec_loss', loss_rec.item(), it + ep * len(self.train_loader))

    @torch.no_grad()
    def sample_generator(self, ep: int, savepath: str):
        self.G.eval()
        imgs, deteriorated_imgs = [], []
        for i in self.sample_testid:
            X, img, noise, mask = self.test_dataset[i]
            imgs.append(img)
            deteriorated_imgs.append(X)
        imgs = torch.stack(imgs, dim=0)
        deteriorated_imgs = torch.stack(deteriorated_imgs, dim=0)
        rec_imgs = imgs.clone()
        rec_imgs[:, :, 32:96, 32:96] = self.G(deteriorated_imgs.to(device=self.device)).cpu()
        show_imgs = []
        for i in range(12):
            show_imgs.extend([imgs[i], deteriorated_imgs[i], rec_imgs[i]])
        show_imgs = torch.stack(show_imgs, dim=0)
        show_imgs = torchvision.utils.make_grid(show_imgs, nrow=6, normalize=True, value_range=(-1, 1))
        fig, ax = plt.subplots(1, 1)
        ax.imshow(torch.permute(show_imgs, [1, 2, 0]))
        ax.set_axis_off()
        ax.set_title(f'Epoch {ep}')
        fig.savefig(savepath, dpi=150, bbox_inches='tight')
        plt.close(fig)
