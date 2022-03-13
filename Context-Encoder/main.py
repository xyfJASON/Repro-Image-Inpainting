import os
import argparse

from trainer import Trainer
from tester import Tester


def train(config_path):
    assert config_path
    trainer = Trainer(config_path)
    trainer.train()


def evaluate(model_path, dataset, dataroot, batch_size):
    assert model_path and dataset and dataroot and batch_size
    tester = Tester(model_path=model_path)
    mse, psnr = tester.evaluate(dataset=dataset, dataroot=dataroot, batch_size=batch_size)
    print(f'mse: {mse}, psnr: {psnr}')


def predict(model_path, dir_path):
    assert model_path and dir_path
    assert os.path.exists(dir_path) and os.path.isdir(dir_path)
    tester = Tester(model_path=model_path)
    files = os.listdir(dir_path)
    if not os.path.exists(os.path.join(dir_path, 'fake')):
        os.makedirs(os.path.join(dir_path, 'fake'))
    for file in files:
        if os.path.isfile(os.path.join(dir_path, file)) and os.path.splitext(file)[1] in ['.jpg', '.png', '.jpeg']:
            img_path = os.path.join(dir_path, file)
            save_path = os.path.join(dir_path, 'fake', file)
            tester.predict(img_path=img_path, save_path=save_path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', required=True, choices=['train', 'evaluate', 'predict'])
    parser.add_argument('--config_path', help='train')
    parser.add_argument('--model_path', help='evaluate / predict')
    parser.add_argument('--dataset', choices=['celeba'], help='evaluate')
    parser.add_argument('--dataroot', help='evaluate')
    parser.add_argument('--batch_size', default=256, help='evaluate')
    parser.add_argument('--predict_dir', help='predict')
    args = parser.parse_args()

    if args.mode == 'train':
        train(args.config_path)
    elif args.mode == 'evaluate':
        evaluate(args.model_path, args.dataset, args.dataroot, args.batch_size)
    elif args.mode == 'predict':
        predict(args.model_path, args.predict_dir)
    else:
        raise ValueError
