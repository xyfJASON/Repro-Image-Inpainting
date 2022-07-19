import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import yaml
import shutil
import datetime
import argparse
from tqdm import tqdm

import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import torchvision.datasets as dset
import torchvision.transforms as T
from torchvision.utils import save_image

import models
from loss import ReconstructLoss, PerceptualLoss, StyleLoss, TVLoss
from utils.general_utils import makedirs
from dataset import DatasetWithMask


class Trainer:
    def __init__(self, config_path: str):
        # ====================================================== #
        # CONFIGURATIONS
        # ====================================================== #
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)

        self.log_root = os.path.join('runs', datetime.datetime.now().strftime('exp-%Y-%m-%d-%H-%M-%S'))
        print('log directory:', self.log_root)

        self.device = torch.device('cuda' if self.config['use_gpu'] and torch.cuda.is_available() else 'cpu')
        print('using device:', self.device)

        if self.config.get('save_per_epochs'):
            makedirs(os.path.join(self.log_root, 'ckpt'))

        makedirs(os.path.join(self.log_root, 'tensorboard'))
        self.writer = SummaryWriter(os.path.join(self.log_root, 'tensorboard'))

        if self.config.get('sample_per_epochs'):
            makedirs(os.path.join(self.log_root, 'samples'))

        if not os.path.exists(os.path.join(self.log_root, 'config.yml')):
            shutil.copyfile(config_path, os.path.join(self.log_root, 'config.yml'))

        # ====================================================== #
        # DATA
        # ====================================================== #
        print('==> Getting data...')
        if self.config['dataset'] == 'celeba':
            self.img_channels = 3
            transforms = T.Compose([T.Resize((self.config['img_size'], self.config['img_size'])),
                                    T.ToTensor(),
                                    T.Normalize(mean=[0.5]*self.img_channels, std=[0.5]*self.img_channels)])
            self.train_dataset = dset.CelebA(root=self.config['dataroot'], split='train', transform=transforms, download=False)
            self.test_dataset = dset.CelebA(root=self.config['dataroot'], split='test', transform=transforms, download=False)
        else:
            raise ValueError(f'Dataset {self.config["dataset"]} is not available now.')

        self.train_dataset = DatasetWithMask(self.train_dataset, mask_type=self.config['mask_type'], mask_fill=0.)
        self.test_dataset = DatasetWithMask(self.test_dataset, mask_type=self.config['mask_type'], mask_fill=0.)
        self.train_loader = DataLoader(self.train_dataset, batch_size=self.config['batch_size'], shuffle=True, num_workers=4, pin_memory=True)
        self.test_loader = DataLoader(self.test_dataset, batch_size=self.config['batch_size'], num_workers=4, pin_memory=True)

        # ====================================================== #
        # DEFINE MODELS, OPTIMIZERS, etc.
        # ====================================================== #
        self.G = models.Generator(self.img_channels, self.config['n_layer'])
        self.G.to(device=self.device)
        self.optimizerG = optim.Adam(self.G.parameters(), lr=self.config['adam']['lr'], betas=self.config['adam']['betas'])
        vgg16 = models.VGG16FeatureExtractor()
        vgg16.to(device=self.device)
        self.Loss_reconstruct = ReconstructLoss()
        self.Loss_perceptual = PerceptualLoss(vgg16)
        self.Loss_style = StyleLoss(vgg16)
        self.Loss_tv = TVLoss()

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
                self.sample_generator(os.path.join(self.log_root, 'samples', f'epoch_{ep}.png'))

            if self.config['save_per_epochs'] and (ep + 1) % self.config['save_per_epochs'] == 0:
                self.save_model(os.path.join(self.log_root, 'ckpt', f'epoch_{ep}.pt'))

        self.save_model(os.path.join(self.log_root, 'model.pt'))
        self.writer.close()

    def train_one_epoch(self, ep):
        self.G.train()
        with tqdm(self.train_loader, desc=f'Epoch {ep}', ncols=120) as pbar:
            for it, (X, gt_img, noise, mask) in enumerate(pbar):
                gt_img = gt_img.to(device=self.device, dtype=torch.float32)
                X = X.to(device=self.device, dtype=torch.float32)
                mask = mask.expand(X.shape).to(device=self.device, dtype=torch.float32)

                inpainted_img = self.G(X, mask)
                loss_hole, loss_valid = self.Loss_reconstruct(inpainted_img, gt_img, mask)
                loss_per = self.Loss_perceptual(inpainted_img, gt_img, mask)
                loss_style = self.Loss_style(inpainted_img, gt_img, mask)
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
    def sample_generator(self, savepath: str):
        self.G.eval()
        gt_imgs, deteriorated_imgs, masks = [], [], []
        sample_testid = torch.randint(0, len(self.test_dataset), (12, ))
        for i in sample_testid:
            X, gt_img, noise, mask = self.test_dataset[i]
            gt_imgs.append(gt_img.to(device=self.device))
            deteriorated_imgs.append(X.to(device=self.device))
            masks.append(mask.expand(X.shape).to(device=self.device))
        gt_imgs = torch.stack(gt_imgs, dim=0)
        deteriorated_imgs = torch.stack(deteriorated_imgs, dim=0)
        masks = torch.stack(masks, dim=0)
        inpainted_imgs = self.G(deteriorated_imgs, masks)
        show_imgs = []
        for i in range(12):
            show_imgs.extend([gt_imgs[i].cpu(), deteriorated_imgs[i].cpu(), inpainted_imgs[i].cpu()])
        show_imgs = torch.stack(show_imgs, dim=0)
        save_image(show_imgs, savepath, nrow=6, normalize=True, value_range=(-1, 1))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_path', default='./config.yml', help='path to training configuration file')
    args = parser.parse_args()

    trainer = Trainer(args.config_path)
    trainer.train()
