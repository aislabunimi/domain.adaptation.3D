#!/usr/bin/env python3
import rospy
import torch
import torchvision
from std_msgs.msg import String
from ros_deeplabv3.srv import Finetune, FinetuneResponse
from torch.utils.data import DataLoader
from torchvision import transforms
import os
from PIL import Image

# Your custom dataset (images + masks)
class SegmentationDataset(torch.utils.data.Dataset):
    def __init__(self, image_dir, mask_dir, transform=None):
        self.image_paths = sorted([os.path.join(image_dir, f) for f in os.listdir(image_dir)])
        self.mask_paths = sorted([os.path.join(mask_dir, f) for f in os.listdir(mask_dir)])
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image = Image.open(self.image_paths[idx]).convert("RGB")
        mask = Image.open(self.mask_paths[idx])
        if self.transform:
            image = self.transform(image)
            mask = transforms.ToTensor()(mask).squeeze().long()
        return image, mask

def train(model, dataloader, num_epochs=5):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    model.train()

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    criterion = torch.nn.CrossEntropyLoss()

    for epoch in range(num_epochs):
        for images, masks in dataloader:
            images = images.to(device)
            masks = masks.to(device)

            optimizer.zero_grad()
            outputs = model(images)['out']
            loss = criterion(outputs, masks)
            loss.backward()
            optimizer.step()
        rospy.loginfo(f"Epoch {epoch+1}: Loss={loss.item():.4f}")
    return model

def handle_finetune(req):
    try:
        dataset_path = req.dataset_path
        num_epochs = req.num_epochs
        image_dir = os.path.join(dataset_path, "images")
        mask_dir = os.path.join(dataset_path, "masks")

        transform = transforms.Compose([
            transforms.Resize((520, 520)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])
        dataset = SegmentationDataset(image_dir, mask_dir, transform)
        dataloader = DataLoader(dataset, batch_size=4, shuffle=True)

        model = torchvision.models.segmentation.deeplabv3_resnet101(pretrained=True)
        num_classes = req.num_classes if hasattr(req, 'num_classes') else 40  # Default to 40 for NYU
        model.classifier[4] = torch.nn.Conv2d(256, num_classes, kernel_size=1)

        rospy.loginfo("Starting fine-tuning...")
        model = train(model, dataloader, num_epochs)

        # Save model
        torch.save(model.state_dict(), os.path.expanduser("~/deeplab_finetuned.pth"))
        return FinetuneResponse(success=True, message="Model finetuned and saved.")

    except Exception as e:
        return FinetuneResponse(success=False, message=str(e))

def finetune_server():
    rospy.init_node('finetune_service')
    rospy.Service('finetune_model', Finetune, handle_finetune)
    rospy.loginfo("Finetune service ready.")
    rospy.spin()

if __name__ == "__main__":
    finetune_server()
