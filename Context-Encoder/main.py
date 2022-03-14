import os
import argparse

from trainer import Trainer
from tester import Tester


def train(config_path):
    assert config_path
    trainer = Trainer(config_path)
    trainer.train()


def evaluate(model_path, dataset, dataroot, mask_root, batch_size):
    assert model_path and dataset and dataroot and batch_size
    tester = Tester(model_path=model_path)
    mse, psnr, ssim = tester.evaluate(dataset=dataset, dataroot=dataroot, mask_root=mask_root, batch_size=batch_size)
    print(f'mse: {mse}, psnr: {psnr}, ssim: {ssim}')


def predict(model_path, predict_dir):
    assert model_path and predict_dir
    assert os.path.exists(predict_dir) and os.path.isdir(predict_dir)
    tester = Tester(model_path=model_path)
    files = os.listdir(predict_dir)
    if not os.path.exists(os.path.join(predict_dir, 'fake')):
        os.makedirs(os.path.join(predict_dir, 'fake'))
    for file in files:
        if os.path.isfile(os.path.join(predict_dir, file)) and os.path.splitext(file)[1] in ['.jpg', '.png', '.jpeg']:
            img_path = os.path.join(predict_dir, file)
            save_path = os.path.join(predict_dir, 'fake', file)
            tester.predict(img_path=img_path, save_path=save_path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', required=True, choices=['train', 'evaluate', 'predict'], help='choose a running mode')
    
    # ==================== For training ==================== #
    parser.add_argument('--config_path', default='./config.yml', help='path to training configuration file')
    
    # ==================== For evaluating or predicting ==================== #
    parser.add_argument('--model_path', help='path to the saved model')
    
    # ==================== For evaluating only ==================== #
    parser.add_argument('--dataset', choices=['celeba'], help='choose a dataset to evaluate on')
    parser.add_argument('--dataroot', help='path to downloaded dataset')
    parser.add_argument('--mask_root', help='path to the directory that contains mask images')
    parser.add_argument('--batch_size', type=int, default=256, help='batch size')
    
    # ==================== For predicting only ==================== #
    parser.add_argument('--predict_dir', help='path to the directory that contains images to be inpainted')
    
    args = parser.parse_args()

    if args.mode == 'train':
        assert args.config_path, 'missing argument config_path'
        train(args.config_path)
    elif args.mode == 'evaluate':
        assert args.model_path, 'missing argument model_path'
        assert args.dataset, 'missing argument dataset'
        assert args.dataroot, 'missing argument dataroot'
        assert args.mask_root, 'missing argument mask_root'
        assert args.batch_size, 'missing argument batch_size'
        evaluate(args.model_path, args.dataset, args.dataroot, args.mask_root, args.batch_size)
    elif args.mode == 'predict':
        assert args.model_path, 'missing argument model_path'
        assert args.predict_dir, 'missing argument predict_dir'
        predict(args.model_path, args.predict_dir)
    else:
        raise ValueError
