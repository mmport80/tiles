import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import functional as F
from torchvision.models.detection import keypointrcnn_resnet50_fpn
from torchvision.models.detection import KeypointRCNN_ResNet50_FPN_Weights
from pycocotools.coco import COCO
from PIL import Image
import time

def resize_and_pad(image, target_size):
    ratio = min(target_size[0] / image.size[0], target_size[1] / image.size[1])
    new_size = tuple([int(x * ratio) for x in image.size])
    image = image.resize(new_size, Image.LANCZOS)
    
    new_image = Image.new("RGB", target_size)
    paste_position = ((target_size[0] - new_size[0]) // 2, (target_size[1] - new_size[1]) // 2)
    new_image.paste(image, paste_position)
    
    return np.array(new_image)

def load_coco_data(annotation_file, image_dir, target_size=(256, 256), max_images=50):
    coco = COCO(annotation_file)
    image_ids = coco.getImgIds()
    images = []
    all_segmentations = []
    filenames = []

    for img_id in image_ids[:max_images]:
        img_info = coco.loadImgs(img_id)[0]
        img_path = os.path.join(image_dir, img_info['file_name'])

        ann_ids = coco.getAnnIds(imgIds=img_id)
        img_anns = coco.loadAnns(ann_ids)

        if not img_anns or not os.path.exists(img_path):
            continue

        try:
            img = Image.open(img_path).convert('RGB')
            original_width, original_height = img.size
            img = img.rotate(90, expand=True)
            
            img_resized = resize_and_pad(img, target_size)
            
            # Calculate scaling and padding factors
            w_scale = target_size[0] / img.height
            h_scale = target_size[1] / img.width
            w_pad = (target_size[0] - img.height * w_scale) / 2
            h_pad = (target_size[1] - img.width * h_scale) / 2
            
            img_segmentations = []
            for ann in img_anns:
                if 'segmentation' in ann:
                    seg = []
                    for i in range(0, len(ann['segmentation'][0]), 2):
                        x_original = ann['segmentation'][0][i]
                        y_original = ann['segmentation'][0][i+1]
                        
                        x_rotated = y_original
                        y_rotated = original_width - x_original
                        
                        x_scaled = x_rotated * w_scale + w_pad
                        y_scaled = y_rotated * h_scale + h_pad
                        
                        seg.append(x_scaled)
                        seg.append(y_scaled)
                    
                    img_segmentations.append(seg)
                        
            images.append(np.array(img_resized, dtype=np.float32) / 255.0)
            all_segmentations.append(img_segmentations)
            filenames.append(img_info['file_name'])

        except Exception as e:
            print(f"Error processing {img_path}: {e}")

    return np.array(images), all_segmentations, filenames

class TileKeypointDataset(Dataset):
    def __init__(self, images, segmentations, filenames):
        self.images = images
        self.segmentations = segmentations
        self.filenames = filenames

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img = self.images[idx]
        segmentations = self.segmentations[idx]
        
        # Convert to tensor
        img_tensor = F.to_tensor(img)
        
        # Prepare keypoints and boxes
        keypoints = []
        boxes = []
        labels = []
        
        for seg in segmentations:
            if len(seg) == 8:  # 4 points
                # Keypoints: reshape to 4x3 (x, y, visibility)
                kpts = np.array(seg).reshape(-1, 2)
                # Add visibility flag (2 = visible)
                kpts = np.column_stack([kpts, np.ones(4) * 2])
                keypoints.append(kpts.flatten())
                
                # Bounding box from keypoints
                x_coords = [seg[i] for i in range(0, len(seg), 2)]
                y_coords = [seg[i + 1] for i in range(0, len(seg), 2)]
                xmin, xmax = min(x_coords), max(x_coords)
                ymin, ymax = min(y_coords), max(y_coords)
                
                if xmin < xmax and ymin < ymax:
                    boxes.append([xmin, ymin, xmax, ymax])
                    labels.append(1)  # tile class
        
        if not keypoints:
            # Handle empty cases
            keypoints = torch.zeros((0, 12), dtype=torch.float32)  # 4 points * 3
            boxes = torch.zeros((0, 4), dtype=torch.float32)
            labels = torch.zeros(0, dtype=torch.int64)
        else:
            keypoints = torch.as_tensor(np.array(keypoints), dtype=torch.float32)
            boxes = torch.as_tensor(np.array(boxes), dtype=torch.float32)
            labels = torch.as_tensor(np.array(labels), dtype=torch.int64)
        
        target = {
            "boxes": boxes,
            "labels": labels,
            "keypoints": keypoints,
            "image_id": torch.tensor([idx])
        }
        
        return img_tensor, target

def get_keypoint_model():
    # Load pre-trained KeypointRCNN
    model = keypointrcnn_resnet50_fpn(weights=KeypointRCNN_ResNet50_FPN_Weights.DEFAULT)
    
    # Modify for 4 keypoints instead of 17
    model.roi_heads.keypoint_predictor.kps_score_lowres = nn.ConvTranspose2d(512, 4, 4, 2, 1)
    model.roi_heads.keypoint_predictor.kps_score_upsampled = nn.ConvTranspose2d(4, 4, 4, 2, 1)
    
    return model

def train_one_epoch(model, optimizer, data_loader, device, epoch):
    model.train()
    epoch_loss = 0.0
    start_time = time.time()

    for batch_idx, (images, targets) in enumerate(data_loader):
        images = list(image.to(device) for image in images)
        targets = [{k: v.to(device) if isinstance(v, torch.Tensor) else v 
                   for k, v in t.items()} for t in targets]

        loss_dict = model(images, targets)
        losses = sum(loss for loss in loss_dict.values())

        optimizer.zero_grad()
        losses.backward()
        optimizer.step()

        epoch_loss += losses.item()
        
        if batch_idx % 10 == 0:
            print(f"Epoch [{epoch}][{batch_idx}/{len(data_loader)}] Loss: {losses.item():.4f}")

    print(f"Epoch {epoch} Average Loss: {epoch_loss / len(data_loader):.4f}")

def main():
    # Load data
    images, segmentations, filenames = load_coco_data('instances_fixed.json', './images', max_images=100)
    print(f"Loaded {len(images)} images")
    
    # Split data
    val_split = int(len(images) * 0.2)
    train_dataset = TileKeypointDataset(images[:-val_split], segmentations[:-val_split], filenames[:-val_split])
    val_dataset = TileKeypointDataset(images[-val_split:], segmentations[-val_split:], filenames[-val_split:])
    
    # Data loaders
    train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True, 
                             collate_fn=lambda x: tuple(zip(*x)))
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False,
                           collate_fn=lambda x: tuple(zip(*x)))
    
    # Model setup
    model = get_keypoint_model()
    device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
    model.to(device)
    
    optimizer = torch.optim.SGD(model.parameters(), lr=0.005, momentum=0.9, weight_decay=0.0005)
    
    # Training
    num_epochs = 5
    for epoch in range(num_epochs):
        train_one_epoch(model, optimizer, train_loader, device, epoch)
    
    print("Training completed!")
    
    # Save model
    torch.save(model.state_dict(), 'keypoint_model.pth')
    print("Model saved!")

if __name__ == "__main__":
    main()
