import torch


class InpaintingEvaluator:
    def __init__(self, C, H, W):
        """ Images should be in range [0, 1].

        Args:
            C: #channel of images
            H: height of images
            W: width of images
        """
        self.C, self.H, self.W = C, H, W
        self.mse = []

    def reset(self):
        self.mse = []

    def update(self, imgs_true, imgs_fake):
        """ Add a batch of image true-fake pairs. """
        assert (0 <= imgs_true).all() and (imgs_true <= 1).all()
        assert (0 <= imgs_fake).all() and (imgs_fake <= 1).all()
        assert imgs_true.shape == imgs_fake.shape
        assert imgs_true.shape[1:] == (self.C, self.H, self.W)
        batch_size = imgs_true.shape[0]
        mse = torch.mean((imgs_true.reshape(batch_size, -1).float() -
                          imgs_fake.reshape(batch_size, -1).float()) ** 2, dim=1)
        self.mse.append(mse)

    def MSE(self):
        """
        Return:
            average mse of all image pairs
        """
        mse = torch.cat(self.mse)
        return mse.mean().item()

    def PSNR(self):
        """
        Return:
            average psnr of all image pairs
        """
        mse = torch.cat(self.mse)
        psnr = 20 * torch.log10(255.0 / torch.sqrt(mse))
        return psnr.mean().item()


if __name__ == '__main__':
    img_true = torch.rand((1, 3, 32, 32))
    img_fake = torch.rand((1, 3, 32, 32))
    evaluator = InpaintingEvaluator(C=3, H=32, W=32)
    evaluator.update(img_true, img_fake)
    print(evaluator.MSE(), evaluator.PSNR())
