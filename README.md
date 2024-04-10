# RandAugment

A straightforward implementation of [RandAugment](https://arxiv.org/pdf/1909.13719.pdf) using the `albumentations` library. It can be used in pipelines that require augmentation of annotations along with the image, such as bounding boxes, segmentation masks, and keypoints.

Tested using `albumentations==1.4.3`.

## Example usage

See `demo.ipynb` for an interactive demo.

### Example 1: Images

```python
from randaugment import RandAugment

image = ...  # your image here

transform = RandAugment(num_transforms=3, magnitude=3)

img_augmented = transform(image=image)['image']
```

### Example 2: Images + bounding boxes

```python

image = ...  # your image here
bboxes = ... # your bboxes here

# The RandAugment class can be combined with other Albumentations transforms.
transform = A.Compose([
    RandAugment(num_transforms=5, magnitude=3),
    A.SmallestMaxSize(max_size=800),
], bbox_params=A.BboxParams(format='coco'))

transformed = transform(image=image, bboxes=bboxes)

img_augmented = transformed["image"]
bboxes_augmented = transformed["bboxes"]
```

### Example 3: Sample transforms manually

```python
randaug = RandAugment(num_transforms=3, magnitude=3)
transforms = randaug.sample_transforms()
```