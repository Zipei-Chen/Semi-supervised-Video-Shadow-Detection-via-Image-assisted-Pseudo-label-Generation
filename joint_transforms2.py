import random

from PIL import Image


class Compose(object):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, img, mask, uncertain, manual_random=None):
        assert img.size == mask.size
        for t in self.transforms:
            img, mask, uncertain = t(img, mask, uncertain, manual_random)
        return img, mask, uncertain


class RandomHorizontallyFlip(object):
    def __call__(self, img, mask, uncertain, manual_random=None):
        if manual_random is None:
            if random.random() < 0.5:
                return img.transpose(Image.FLIP_LEFT_RIGHT), mask.transpose(Image.FLIP_LEFT_RIGHT), uncertain.transpose(Image.FLIP_LEFT_RIGHT)
            return img, mask, uncertain
        else:
            if manual_random < 0.5:
                return img.transpose(Image.FLIP_LEFT_RIGHT), mask.transpose(Image.FLIP_LEFT_RIGHT), uncertain.transpose(Image.FLIP_LEFT_RIGHT)
            return img, mask, uncertain


class Resize(object):
    def __init__(self, size):
        self.size = tuple(reversed(size))  # size: (h, w)

    def __call__(self, img, mask, uncertain, manual_random=None):
        assert img.size == mask.size
        return img.resize(self.size, Image.BILINEAR), mask.resize(self.size, Image.NEAREST), uncertain.resize(self.size, Image.NEAREST)
