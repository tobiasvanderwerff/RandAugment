import random
from typing import Any, Dict, List

import numpy as np
import albumentations as A


class RandAugment(A.BaseCompose):
    """
    Applies a series of random transforms as described in the RandAugment paper.

    Args:
        num_transforms (int): Number of transforms to apply. Default: 3.
        magnitude (int): Magnitude of each transform (0 to 10). Default: 3.
        p (float): Probability of applying the transform. Default: 1.0.
    """

    def __init__(self, num_transforms: int = 3, magnitude: int = 3, p: float = 1.0):
        assert 0 <= magnitude <= 10, "Magnitude must be between 0 and 10."

        super().__init__(randaugment_transforms(magnitude), p)

        assert 1 <= num_transforms <= len(self.transforms), (
            "Number of transforms must be between 1 and the number of available transforms."
        )

        self.num_transforms = num_transforms
        self.magnitude = magnitude

    def __call__(self, *arg: Any, force_apply: bool = False, **data: Any) -> Dict[str, Any]:
        if force_apply or random.random() < self.p:
            transforms = self.sample_transforms()
            for t in transforms:
                data = t(force_apply=True, **data)
        return data

    def sample_transforms(self) -> List[A.BasicTransform]:
        random_state = np.random.RandomState(random.randint(0, (1 << 32) - 1))
        idx = random_state.choice(len(self.transforms), self.num_transforms, replace=False)
        return [self.transforms[i] for i in idx]


def randaugment_transforms(magnitude: int = 4):
    """
    Returns list of albumentations transforms used for RandAugment, each with a given magnitude.

    Based on https://github.com/adam-mehdi/MuarAugment/blob/master/muar/transform_lists.py

    Args:
        magnitude (int): Magnitude of each transform in the returned list.
    """
    M = magnitude

    # Augmentations as close as possible to the original RandAugment paper,
    # with magnitudes copied from other sources, e.g. https://github.com/ildoonet/pytorch-randaugment/blob/master/RandAugment/augmentations.py
    transform_list = [
        A.RandomBrightnessContrast(brightness_limit=0, contrast_limit=.1+M*.15, always_apply=True),
        A.RandomBrightnessContrast(brightness_limit=.1+M*.15, contrast_limit=0, always_apply=True),
        A.Solarize(threshold=255-M*25, always_apply=True),
        A.Equalize(always_apply=True),
        A.RGBShift(r_shift_limit=M*10, g_shift_limit=M*10, b_shift_limit=M*10, always_apply=True),
        A.Sharpen(M/10, always_apply=True),
        A.Posterize(num_bits=8-int(M*4/10), always_apply=True),
        A.Rotate(limit=M*5, always_apply=True),
        A.Affine(shear={"x": (-M*10, M*10)}, always_apply=True),
        A.Affine(shear={"y": (-M*10, M*10)}, always_apply=True),
        A.Affine(translate_percent={"x": (-M*.05, M*.05)}, always_apply=True),
        A.Affine(translate_percent={"y": (-M*.05, M*.05)}, always_apply=True),
        A.NoOp(always_apply=True),
    ]
    return transform_list