#%%
import os
import torch
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
import yaml
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm  # 进度条

# 设置CUDA_LAUNCH_BLOCKING
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

# 定义变换
transform = transforms.Compose([
    transforms.Resize((200, 200)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

class TrafficDataset(Dataset):
    def __init__(self, images_dir, labels_dir, transform=None):
        self.images_dir = images_dir
        self.labels_dir = labels_dir
        self.image_files = os.listdir(images_dir)
        self.transform = transform

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        image_file = self.image_files[idx]
        image_path = os.path.join(self.images_dir, image_file)
        image = Image.open(image_path).convert("RGB")
        width, height = image.size
        
        label_file = os.path.splitext(image_file)[0] + ".txt"
        label_path = os.path.join(self.labels_dir, label_file)
        
        labels = []
        with open(label_path, "r") as f:
            lines = f.read().strip().split("\n")
            for line in lines:
                parts = line.split()
                if len(parts) != 5:
                    continue
                class_id, x_center, y_center, width_ratio, height_ratio = map(float, parts)
                x_center *= width
                y_center *= height
                box_width = width_ratio * width
                box_height = height_ratio * height
                x_min = x_center - box_width / 2
                y_min = y_center - box_height / 2
                x_max = x_center + box_width / 2
                y_max = y_center + box_height / 2
                if (x_max - x_min) > 0 and (y_max - y_min) > 0:
                    labels.append([class_id, x_min, y_min, x_max, y_max])
        
        if self.transform:
            # Apply the transform to the image
            transformed_image = self.transform(image)
            # Calculate the scaling factors
            new_width, new_height = transformed_image.shape[2], transformed_image.shape[1]
            scale_x = new_width / width
            scale_y = new_height / height
            # Apply the scaling to the bounding boxes
            for label in labels:
                label[1] *= scale_x
                label[2] *= scale_y
                label[3] *= scale_x
                label[4] *= scale_y
        
        if len(labels) > 0:
            labels = torch.tensor(labels)
        else:
            labels = torch.empty((0, 5))
        
        # Debug: print labels for the current image
        print(f"Image: {image_file}, Labels: {labels}")
        
        return transformed_image, labels

def show_batch(train_loader, device):
    data_iter = iter(train_loader)
    images, targets = next(data_iter)
    
    # Plotting the first batch of images with annotations
    fig, axs = plt.subplots(nrows=1, ncols=min(len(images), 5), figsize=(20, 5))
    if len(images) == 1:
        axs = [axs]  # Ensure axs is always a list of axes

    for i, (image, target) in enumerate(zip(images, targets)):
        if i >= 5:
            break
        
        image = image.to(device)
        mean = torch.tensor([0.5, 0.5, 0.5]).to(device)
        std = torch.tensor([0.5, 0.5, 0.5]).to(device)
        image = image * std[:, None, None] + mean[:, None, None]
        image = image.cpu().clamp(0, 1)
        
        image = transforms.ToPILImage()(image)
        ax = axs[i]
        ax.imshow(image)
        boxes = target[:, 1:].cpu().numpy()
        class_ids = target[:, 0].cpu().numpy().astype(int)
        
        for box, class_id in zip(boxes, class_ids):
            x_min, y_min, x_max, y_max = box
            rect = plt.Rectangle((x_min, y_min), x_max - x_min, y_max - y_min, fill=False, color='red', linewidth=2)
            ax.add_patch(rect)
            ax.text(x_min, y_min, f'Class {class_id}', color='white', fontsize=12, bbox=dict(facecolor='red', alpha=0.5))
        
        ax.axis('off')
    plt.show()

def main():
    # Load parameters from YAML file
    with open('./trafic/data_1.yaml', 'r', encoding='utf-8') as f:
        prmtr = yaml.load(f, Loader=yaml.FullLoader)
        train_images = prmtr['train_images']
        train_labels = prmtr['train_labels']
        val_images = prmtr['val_images']
        val_labels = prmtr['val_labels']
        batch_size = prmtr['batch_sizes']
        num_workers = prmtr['num_workers']
        num_classes = prmtr['num_classes']
        num_epochs = prmtr['num_epochs']

    # Create training dataset and data loader
    train_dataset = TrafficDataset(train_images, train_labels, transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, collate_fn=collate_fn)

    # Get device
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    print(device)

    # Display batch of images
    show_batch(train_loader, device)
    
    # Further training and validation code would go here

if __name__ == "__main__":
    main()

# %%
