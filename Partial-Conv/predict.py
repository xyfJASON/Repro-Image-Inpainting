import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import argparse
from tqdm import tqdm
from PIL import Image
import torch
import torchvision.transforms as T
from torchvision.utils import save_image

import models
from utils.general_utils import makedirs


@torch.no_grad()
def predict_single(G, args, img_path, mask_path, save_path):
    assert os.path.exists(img_path) and os.path.isfile(mask_path) and os.path.isfile(img_path)
    device = torch.device('cuda' if torch.cuda.is_available() and not args.cpu else 'cpu')
    G.eval()
    transforms = T.Compose([T.Resize((args.img_size, args.img_size)),
                            T.ToTensor(),
                            T.Normalize(mean=[0.5]*args.img_channels, std=[0.5]*args.img_channels)])
    mask_transforms = T.Compose([T.Resize((args.img_size, args.img_size)), T.ToTensor()])
    img = Image.open(img_path)
    img = transforms(img).unsqueeze(0)
    img = img.to(device=device)
    mask = Image.open(mask_path)
    mask = mask_transforms(mask).unsqueeze(0)
    mask = mask.expand(img.shape).to(device=device)
    output = G(img, mask)
    output_comp = mask * output + (1 - mask) * img
    save_image(output_comp, save_path, normalize=True, value_range=(-1, 1))


@torch.no_grad()
def predict(args):
    device = torch.device('cuda' if torch.cuda.is_available() and not args.cpu else 'cpu')
    ckpt = torch.load(args.model_path, map_location='cpu')
    G = models.Generator(img_channels=args.img_channels, n_layer=args.n_layer)
    G.load_state_dict(ckpt)
    G.to(device=device)

    img_dir = os.path.join(args.predict_dir, 'img')
    mask_dir = os.path.join(args.predict_dir, 'mask')
    fake_dir = os.path.join(args.predict_dir, 'fake')
    assert os.path.exists(img_dir) and os.path.isdir(img_dir), f'{img_dir} is not a valid directory'
    assert os.path.exists(mask_dir) and os.path.isdir(mask_dir), f'{mask_dir} is not a valid directory'
    makedirs(fake_dir)
    files = os.listdir(img_dir)
    for file in tqdm(files, desc='Predicting', ncols=120):
        img_path = os.path.join(img_dir, file)
        mask_path = os.path.join(mask_dir, file)
        if os.path.isfile(img_path) and os.path.splitext(file)[1] in ['.jpg', '.png', '.jpeg']:
            assert os.path.isfile(mask_path), f'{mask_path} does not exist'
            save_path = os.path.join(fake_dir, file)
            predict_single(G, args, img_path=img_path, mask_path=mask_path, save_path=save_path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', required=True, help='path to the saved model')
    parser.add_argument('--img_size', type=int, default=128, help='size of the image')
    parser.add_argument('--img_channels', type=int, default=3, help='# of image channels')
    parser.add_argument('--n_layer', type=int, default=7, help='# of layers in generator')
    parser.add_argument('--predict_dir', required=True, help='directory containing images to be inpainted')
    parser.add_argument('--cpu', action='store_true', help='use cpu instead of cuda')
    args = parser.parse_args()
    predict(args)


if __name__ == '__main__':
    main()
