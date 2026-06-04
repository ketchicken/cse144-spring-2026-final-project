import os
import torch
from torchvision import datasets
from PIL import Image
from torchvision.transforms import v2 as tfv2
from torch.utils.data import Dataset

class TestSet(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(os.listdir(self.root_dir))

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_id = f"{idx}.jpg"
        img_path = os.path.join(self.root_dir, img_id)
        sample = Image.open(img_path)

        if self.transform:
            sample = self.transform(sample)

        return sample, img_id

class NumericImageFolder(datasets.ImageFolder):
    """ImageFolder but the labels correspond to the right numbers

    Args:
        root (string): Root directory path.
        transform (callable, optional): A function/transform that  takes in an PIL image
            and returns a transformed version. E.g, ``transforms.RandomCrop``

     Attributes:
        classes (list): List of the class names.
        class_to_idx (dict): Dict with items (class_name, class_index).
        imgs (list): List of (image path, class_index) tuples
    """
    def find_classes(self, directory):
        """
        Overrides the default alphanumeric class-to-index generation.
        """
        # Define your explicit custom mapping here
        custom_mapping = {str(k):k for k in range(0, 100)}

        # Extract the unique class names list
        classes = list(custom_mapping.keys())

        return classes, custom_mapping
    
class TransformsAugments():
    def __init__(self, mix_factor):
        cutmix = tfv2.CutMix(alpha=mix_factor, num_classes=100)
        mixup = tfv2.MixUp(alpha=mix_factor, num_classes=100)
        self.cutmix_or_mixup = tfv2.RandomChoice([cutmix, mixup])

    def get_transforms(self, resize, magnitude, color_jitter_factor=0.2, degrees_rotation=15, mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)):
        return tfv2.Compose([
                    # Augmentations
                    tfv2.RandAugment(num_ops=2, magnitude=magnitude, interpolation=tfv2.InterpolationMode.BILINEAR), # random augmentation
                    tfv2.RandomHorizontalFlip(),        # Flip Horizontal
                    tfv2.ColorJitter(brightness=color_jitter_factor, contrast=color_jitter_factor), # adjust brightness and contrast

                    # Normalization
                    tfv2.Resize((resize,resize), interpolation=tfv2.InterpolationMode.BILINEAR),
                    tfv2.CenterCrop((resize,resize)),
                    tfv2.ToImage(),
                    tfv2.ToDtype(torch.float32, scale=True),
                    tfv2.Normalize(mean, std)
                ])

