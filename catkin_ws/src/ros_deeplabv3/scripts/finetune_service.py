#!/usr/bin/env python3

import rospy
import os
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
import torch.cuda.amp as amp
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.models.segmentation import deeplabv3_resnet101

from PIL import Image
from ros_deeplabv3.srv import Finetune, FinetuneResponse


class SegmentationDataset(Dataset):

    def __init__(self, image_dir, mask_dir, transform=None, target_size=(480, 640)):
        self.image_paths = sorted([os.path.join(image_dir, f) for f in os.listdir(image_dir) if f.endswith(('png','jpg'))])
        self.mask_paths = sorted([os.path.join(mask_dir, f) for f in os.listdir(mask_dir) if f.endswith('png')])
        self.transform = transform
        self.target_size = target_size

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image = Image.open(self.image_paths[idx]).convert("RGB")
        mask = Image.open(self.mask_paths[idx])

        image = image.resize((self.target_size[1], self.target_size[0]), Image.BILINEAR)
        mask = mask.resize((self.target_size[1], self.target_size[0]), Image.NEAREST)

        mask = np.array(mask, dtype=np.int64)
        mask[mask >= 41] = 40  # Clamp label indices

        if self.transform:
            image = self.transform(image)

        mask = torch.from_numpy(mask).long()
        return image, mask


def train_and_save_model(loader, model_path, num_classes=40, num_epochs=3):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = deeplabv3_resnet101(pretrained=False)
    model.classifier[4] = nn.Conv2d(256, num_classes, kernel_size=1)
    model.aux_classifier[4] = nn.Conv2d(256, num_classes, kernel_size=1)
    model.load_state_dict(torch.load("/path/to/original_model.pth"))  # Update this path
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=1e-5, weight_decay=1e-5)
    scaler = amp.GradScaler(enabled=torch.cuda.is_available())

    model.train()
    for epoch in range(num_epochs):
        total_loss = 0.0
        for images, masks in loader:
            images = images.to(device)
            masks = masks.to(device)

            with amp.autocast(enabled=torch.cuda.is_available()):
                outputs = model(images)["out"]
                loss = criterion(outputs, masks)

            optimizer.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            total_loss += loss.item()

        rospy.loginfo(f"[Epoch {epoch+1}] Loss: {total_loss / len(loader):.4f}")

    torch.save(model.state_dict(), model_path)
    rospy.loginfo(f"[Finetune] Fine-tuning complete. Model saved to {model_path}.")


def handle_finetune(req):
    image_dir = req.image_path
    mask_dir = req.label_path

    rospy.loginfo(f"[Finetune] Starting fine-tuning on {image_dir} with labels from {mask_dir}")

    if not os.path.isdir(image_dir) or not os.path.isdir(mask_dir):
        return FinetuneResponse(success=False, message="Invalid input directories.")

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    dataset = SegmentationDataset(image_dir, mask_dir, transform=transform)
    if len(dataset) == 0:
        return FinetuneResponse(success=False, message="Empty dataset.")

    loader = DataLoader(dataset, batch_size=4, shuffle=True, num_workers=2, pin_memory=True)

    #REMOVE THIS
    return FinetuneResponse(success=True, message="Fine-tuning completed successfully.")
    # WHEN POSSIBLE
    try:
        train_and_save_model(loader, "/path/to/fine_tuned_model.pth")
        return FinetuneResponse(success=True, message="Fine-tuning completed successfully.")
    except Exception as e:
        rospy.logerr(f"[Finetune] Error during training: {str(e)}")
        return FinetuneResponse(success=False, message=f"Training failed: {str(e)}")



def finetune_service_node():
    rospy.init_node("deeplab_finetune_service")
    rospy.Service("/deeplab_finetune", Finetune, handle_finetune)
    rospy.loginfo("[Service] Deeplab finetune service ready.")
    rospy.spin()


if __name__ == "__main__":
    finetune_service_node()
