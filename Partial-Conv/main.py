import argparse

from trainer import Trainer
from tester import Tester


def train(config_path):
    trainer = Trainer(config_path)
    trainer.train()


def evaluate(model_path, img_channels, dataset, dataroot, mask_root, batch_size):
    tester = Tester(model_path=model_path, img_channels=img_channels)
    mse, psnr, ssim = tester.evaluate(dataset=dataset, dataroot=dataroot, mask_root=mask_root, batch_size=batch_size)
    print(f'mse: {mse}, psnr: {psnr}, ssim: {ssim}')


def predict(model_path, img_channels, predict_dir):
    tester = Tester(model_path=model_path, img_channels=img_channels)
    tester.predict(predict_dir)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', required=True, choices=['train', 'evaluate', 'predict'], help='choose a running mode')
    # ==================== For training ==================== #
    parser.add_argument('--config_path', default='./config.yml', help='path to training configuration file')
    # ==================== For evaluating or predicting ==================== #
    parser.add_argument('--model_path', help='path to the saved model')
    parser.add_argument('--img_channels', type=int, default=3, help='# of image channels')
    # ==================== For evaluating only ==================== #
    parser.add_argument('--dataset', choices=['celeba'], help='dataset to evaluate on. Options: celeba')
    parser.add_argument('--dataroot', help='path to pre-downloaded dataset')
    parser.add_argument('--mask_root', help='path to pre-downloaded mask images')
    parser.add_argument('--batch_size', type=int, default=128, help='batch size')
    # ==================== For predicting only ==================== #
    parser.add_argument('--predict_dir', help='directory containing images to be inpainted')
    args = parser.parse_args()

    if args.mode == 'train':
        assert args.config_path, 'missing argument config_path'
        train(args.config_path)
    elif args.mode == 'evaluate':
        assert args.model_path, 'missing argument model_path'
        assert args.img_channels, 'missing argument img_channels'
        assert args.dataset, 'missing argument dataset'
        assert args.dataroot, 'missing argument dataroot'
        assert args.mask_root, 'missing argument mask_root'
        assert args.batch_size, 'missing argument batch_size'
        evaluate(args.model_path, args.img_channels, args.dataset, args.dataroot, args.mask_root, args.batch_size)
    elif args.mode == 'predict':
        assert args.model_path, 'missing argument model_path'
        assert args.img_channels, 'missing argument img_channels'
        assert args.predict_dir, 'missing argument predict_dir'
        predict(args.model_path, args.img_channels, args.predict_dir)
    else:
        raise ValueError
