import os
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import random


class RGBSegmentationDataset(Dataset):
    def __init__(self, image_dir, label_dir, image_size=(320, 240), augment=False):
        self.image_paths = sorted([
            os.path.join(image_dir, fname)
            for fname in os.listdir(image_dir)
            if fname.endswith(('.png', '.jpg'))
        ])
        self.label_paths = sorted([
            os.path.join(label_dir, fname)
            for fname in os.listdir(label_dir)
            if fname.endswith(('.png', '.jpg'))
        ])
        assert len(self.image_paths) == len(self.label_paths), "Mismatched image-label count"

        self.augment = augment
        self.image_size = image_size

        # Define transforms
        self.to_tensor = transforms.ToTensor()
        self.resize = transforms.Resize(image_size, interpolation=Image.BILINEAR)

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        label_path = self.label_paths[idx]

        image = Image.open(img_path).convert("RGB")
        label = Image.open(label_path)

        # Resize
        image = self.resize(image)
        label = transforms.Resize(self.image_size, interpolation=Image.NEAREST)(label)

        # Augmentations
        if self.augment:
            if random.random() > 0.5:
                image = transforms.functional.hflip(image)
                label = transforms.functional.hflip(label)
            angle = random.uniform(-10, 10)
            image = transforms.functional.rotate(image, angle, interpolation=transforms.InterpolationMode.BILINEAR)
            label = transforms.functional.rotate(label, angle, interpolation=transforms.InterpolationMode.NEAREST)

        return self.to_tensor(image), torch.from_numpy(np.array(label)).long()


def get_dataloader(image_dir, label_dir, batch_size=8, shuffle=True, num_workers=4, image_size=(320, 240), augment=False):
    dataset = RGBSegmentationDataset(
        image_dir=image_dir,
        label_dir=label_dir,
        image_size=image_size,
        augment=augment
    )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True
    )
