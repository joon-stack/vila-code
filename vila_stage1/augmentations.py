import torch
import torch.nn.functional as F
import torchvision.transforms as TF
import random


class RandomShiftAug:
    def __init__(self, pad=14):
        self.pad = pad

    def __call__(self, x):
        n, c, h, w = x.size()
        assert h == w
        padding = tuple([self.pad] * 4)
        x = F.pad(x, padding, "replicate")
        eps = 1.0 / (h + 2 * self.pad)
        arange = torch.linspace(-1.0 + eps, 1.0 - eps, h + 2 * self.pad, device=x.device, dtype=x.dtype)[:h]
        arange = arange.unsqueeze(0).repeat(h, 1).unsqueeze(2)
        base_grid = torch.cat([arange, arange.transpose(1, 0)], dim=2)
        base_grid = base_grid.unsqueeze(0).repeat(n, 1, 1, 1)
        shift = torch.randint(0, 2 * self.pad + 1, size=(n, 1, 1, 2), device=x.device, dtype=x.dtype)
        shift *= 2.0 / (h + 2 * self.pad)
        grid = base_grid + shift
        return F.grid_sample(x, grid, padding_mode="zeros", align_corners=False)


class RandomRotateAug:
    def __init__(self, pad=17, degrees=180) -> None:
        self.degrees = degrees
        self.pad = pad
        self.rotate = TF.RandomRotation(degrees=degrees, interpolation=TF.InterpolationMode.BILINEAR, expand=False)

    def __call__(self, x):
        n, c, h, w = x.shape
        x = TF.functional.pad(x, self.pad, padding_mode="edge")
        x = self.rotate(x)
        x = TF.functional.center_crop(x, output_size=(h, w))
        return x


class RandomPerspectiveAug:
    def __init__(self, pad=30, scale=0.5, p=1.0) -> None:
        self.pad = pad
        self.perspective = TF.RandomPerspective(
            distortion_scale=scale,
            p=p,
        )

    def __call__(self, x):
        n, c, h, w = x.shape
        x = TF.functional.pad(x, self.pad, padding_mode="edge")
        x = self.perspective(x)
        x = TF.functional.center_crop(x, output_size=(h, w))
        return x


class ColorJitterAug:
    def __init__(self, brightness=0.8, contrast=0.8, saturation=0.8, hue=0.2):
        self.brightness = brightness
        self.contrast = contrast
        self.saturation = saturation
        self.hue = hue

    def __call__(self, x):
        return TF.functional.adjust_brightness(
            TF.functional.adjust_contrast(
                TF.functional.adjust_saturation(
                    TF.functional.adjust_hue(x, random.uniform(-self.hue, self.hue)),
                    random.uniform(1-self.saturation, 1+self.saturation)
                ),
                random.uniform(1-self.contrast, 1+self.contrast)
            ),
            random.uniform(1-self.brightness, 1+self.brightness)
        )


class RandomGrayscaleAug:
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, x):
        if random.random() < self.p:
            return TF.functional.rgb_to_grayscale(x, num_output_channels=3)
        return x


class GaussianBlurAug:
    def __init__(self, kernel_size=23, sigma=(0.1, 2.0)):
        self.kernel_size = kernel_size
        self.sigma = sigma

    def __call__(self, x):
        sigma = random.uniform(self.sigma[0], self.sigma[1])
        return TF.functional.gaussian_blur(x, kernel_size=self.kernel_size, sigma=sigma)


class RandomResizedCropAug:
    def __init__(self, size, scale=(0.08, 1.0), ratio=(3./4., 4./3.)):
        self.size = size
        self.scale = scale
        self.ratio = ratio

    def __call__(self, x):
        n, c, h, w = x.shape
        # RandomResizedCrop for each image in batch
        result = torch.zeros_like(x)
        for i in range(n):
            result[i] = TF.functional.resized_crop(
                x[i:i+1], 
                top=random.randint(0, max(1, h - int(h * self.scale[1]))),
                left=random.randint(0, max(1, w - int(w * self.scale[1]))),
                height=random.randint(int(h * self.scale[0]), int(h * self.scale[1])),
                width=random.randint(int(w * self.scale[0]), int(w * self.scale[1])),
                size=(h, w)
            ).squeeze(0)
        return result


class ComposeAugs:
    def __init__(self, augs):
        self.augs = augs

    def __call__(self, x):
        for aug in self.augs:
            x = aug(x)
        return x


def get_aug(aug_choice, img_resolution):
    if img_resolution == 64:
        rotate_pad, shift_pad = 17, 14
    elif img_resolution == 128 or img_resolution == 256:
        rotate_pad, shift_pad = 28, 28
    else:
        raise RuntimeError("only images with 64px and 128px are supported")

    augs = {
        "rotate": RandomRotateAug(rotate_pad, degrees=90),
        "shift": RandomShiftAug(shift_pad),
        "perspective": RandomPerspectiveAug(),
        "colorjitter": ColorJitterAug(brightness=0.8, contrast=0.8, saturation=0.8, hue=0.2),
        "grayscale": RandomGrayscaleAug(p=0.5),
        "blur": GaussianBlurAug(kernel_size=23, sigma=(0.1, 2.0)),
        "resizedcrop": RandomResizedCropAug(size=img_resolution, scale=(0.08, 1.0), ratio=(3./4., 4./3.)),
        "nothing": lambda x: x,
    }
    composed_aug = ComposeAugs(augs=[augs[aug] for aug in aug_choice.split("-")])
    return composed_aug


class Augmenter:
    def __init__(self, img_resolution):
        # Enhanced augmentation set with new color and geometric augmentations
        self.augs = [
            # Original geometric augmentations
            get_aug("nothing", img_resolution),
            get_aug("shift", img_resolution),
            get_aug("rotate", img_resolution),
            get_aug("perspective", img_resolution),
            get_aug("shift-rotate", img_resolution),
            get_aug("rotate-shift", img_resolution),
            get_aug("rotate-perspective", img_resolution),
            get_aug("perspective-rotate", img_resolution),
            
            # # New color augmentations
            # get_aug("colorjitter", img_resolution),
            # get_aug("grayscale", img_resolution),
            # get_aug("blur", img_resolution),
            # get_aug("resizedcrop", img_resolution),
            
            # # Color + Geometric combinations
            # get_aug("colorjitter-shift", img_resolution),
            # get_aug("colorjitter-rotate", img_resolution),
            # get_aug("grayscale-shift", img_resolution),
            # get_aug("grayscale-rotate", img_resolution),
            # get_aug("blur-shift", img_resolution),
            # get_aug("blur-rotate", img_resolution),
            # get_aug("resizedcrop-colorjitter", img_resolution),
            # get_aug("resizedcrop-grayscale", img_resolution),
            
            # # Triple combinations
            # get_aug("colorjitter-shift-rotate", img_resolution),
            # get_aug("grayscale-shift-rotate", img_resolution),
            # get_aug("blur-shift-rotate", img_resolution),
        ]
        self.num_augs = len(self.augs)

    def __call__(self, x):
        x_ = x.clone()
        n, c, h, w = x_.shape

        # Select one aug for each image uniformly random
        image_augs = torch.randint(self.num_augs, (n,))
        # Apply each aug
        for i, apply_aug in enumerate(self.augs):
            # Augment the indicies
            selected_inds = image_augs == i
            if torch.sum(selected_inds) > 0:
                x_[selected_inds] = apply_aug(x_[selected_inds])

        return x_
