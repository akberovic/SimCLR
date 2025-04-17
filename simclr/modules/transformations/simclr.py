import torchvision.transforms as transforms
from PIL import ImageFilter
import random

class RandomSharpness(object):
    """
    Apply a sharpen filter to the image with a probability of p.
    """
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, img):
        if random.random() < self.p:
            return img.filter(ImageFilter.SHARPEN)
        return img

class TransformsSimCLR:
    """
    A stochastic data augmentation module that transforms any given data example randomly,
    resulting in two correlated views of the same example, denoted x̃i and x̃j,
    which we consider as a positive pair.
    """
    def __init__(self, size):
        s = 1
        color_jitter = transforms.ColorJitter(
            0.8 * s, 0.8 * s, 0.8 * s, 0.2 * s
        )
        self.train_transform = transforms.Compose([
            transforms.RandomResizedCrop(size=size),
            transforms.RandomHorizontalFlip(),          # Horizontal flip with 0.5 probability
            transforms.RandomVerticalFlip(),            # Added vertical flip
            transforms.RandomRotation(degrees=15),      # Added random rotation within ±15°
            transforms.RandomApply([color_jitter], p=0.8),# Color jitter
            transforms.RandomGrayscale(p=0.2),           # Random grayscale conversion
            transforms.RandomApply([
                transforms.GaussianBlur(
                    kernel_size=(int(0.1 * size) // 2 * 2 + 1)
                )
            ], p=0.5),                                # Added Gaussian blur with 50% probability
            transforms.RandomApply([RandomSharpness(p=1.0)], p=0.5), # Added random sharpness with 50% probability
            transforms.ToTensor(),
        ])

        self.test_transform = transforms.Compose([
            transforms.Resize(size=size),
            transforms.ToTensor(),
        ])

    def __call__(self, x):
        return self.train_transform(x), self.train_transform(x)
