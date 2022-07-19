import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import argparse
from tqdm import tqdm
import torch
from torch.utils.data import DataLoader
import torchvision.transforms as T
import torchvision.datasets as dset

import models
from utils import metrics
from dataset import DatasetWithMask


@torch.no_grad()
def evaluate(G, data_loader, device):
    G.eval()
    mse, psnr, ssim = 0, 0, 0
    total = 0
    for X, img, noise, mask in tqdm(data_loader, desc='Evaluating', ncols=120, leave=False, colour='yellow'):
        bs = X.shape[0]
        img = img.to(device=device, dtype=torch.float32)
        X = X.to(device=device, dtype=torch.float32)
        mask = mask.expand(X.shape).to(device=device, dtype=torch.float32)
        output = G(X, mask)
        output_comp = mask * output + (1 - mask) * img
        mse += metrics.MSE((img + 1) / 2, (output_comp + 1) / 2, batch=True) * bs
        psnr += metrics.PSNR((img + 1) / 2, (output_comp + 1) / 2, batch=True) * bs
        ssim += metrics.SSIM((img + 1) / 2, (output_comp + 1) / 2, batch=True) * bs
        total += bs
    return mse / total, psnr / total, ssim / total


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', required=True, help='path to the saved model')
    parser.add_argument('--img_size', type=int, default=128, help='size of the image')
    parser.add_argument('--img_channels', type=int, default=3, help='# of image channels')
    parser.add_argument('--n_layer', type=int, default=7, help='# of layers in generator')
    parser.add_argument('--dataset', choices=['celeba'], required=True, help='dataset to evaluate on. Options: celeba')
    parser.add_argument('--dataroot', required=True, help='path to pre-downloaded dataset')
    parser.add_argument('--mask_type', required=True, help="a string of {'center', 'rectangles', 'brushes'} "
                                                           "or path to pre-downloaded mask images")
    parser.add_argument('--batch_size', type=int, default=128, help='batch size')
    parser.add_argument('--cpu', action='store_true', help='use cpu instead of cuda')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() and not args.cpu else 'cpu')
    ckpt = torch.load(args.model_path, map_location='cpu')
    G = models.Generator(img_channels=args.img_channels, n_layer=args.n_layer)
    G.load_state_dict(ckpt)
    G.to(device=device)

    transforms = T.Compose([T.Resize((args.img_size, args.img_size)),
                            T.ToTensor(),
                            T.Normalize(mean=[0.5]*args.img_channels, std=[0.5]*args.img_channels)])
    if args.dataset == 'celeba':
        test_dataset = dset.CelebA(root=args.dataroot, split='test', transform=transforms, download=False)
    else:
        raise ValueError(f'Dataset {args.dataset} is not available now.')
    test_dataset = DatasetWithMask(test_dataset, mask_type=args.mask_type, mask_fill=0.)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, num_workers=4, pin_memory=True)

    mse, psnr, ssim = evaluate(G, test_loader, device)
    print(f'mse: {mse}, psnr: {psnr}, ssim: {ssim}')


if __name__ == '__main__':
    main()
