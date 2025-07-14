import os
import numpy as np
from PIL import Image
from pycocotools.coco import COCO
from pycocotools import mask as mask_utils
import cv2
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import functional as F
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection import fasterrcnn_mobilenet_v3_large_fpn, fasterrcnn_mobilenet_v3_large_320_fpn
from torchvision.models.detection import FasterRCNN_MobileNet_V3_Large_FPN_Weights, FasterRCNN_MobileNet_V3_Large_320_FPN_Weights
from torchvision.models.detection import FasterRCNN
import torch
import time
import matplotlib.pyplot as plt
from exif import Image as ExifImage
import logging
from torchmetrics.detection import MeanAveragePrecision

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


def resize_and_pad(image, target_size):
    # Resize the image while maintaining aspect ratio
    ratio = min(target_size[0] / image.size[0], target_size[1] / image.size[1])
    new_size = tuple([int(x * ratio) for x in image.size])
    image = image.resize(new_size, Image.LANCZOS)

    # Create a new image with a black background
    new_image = Image.new("RGB", target_size)

    # Paste the resized image onto the center of the new image
    paste_position = ((target_size[0] - new_size[0]) // 2, (target_size[1] - new_size[1]) // 2)
    new_image.paste(image, paste_position)

    return np.array(new_image)

def load_coco_data(annotation_file, image_dir, target_size=(224, 224), max_images=2):
    coco = COCO(annotation_file)
    image_ids = coco.getImgIds()
    images = []
    all_segmentations = []  # Store all segmentations here
    filenames = []

    for img_id in image_ids[:max_images]:
        img_info = coco.loadImgs(img_id)[0]
        img_path = os.path.join(image_dir, img_info['file_name'])

        ann_ids = coco.getAnnIds(imgIds=img_id)
        img_anns = coco.loadAnns(ann_ids)

        if not img_anns:
            logging.warning(f"Skipping image {img_info['file_name']} due to lack of annotations")
            continue

        if os.path.exists(img_path):
            try:
                img = Image.open(img_path).convert('RGB')
                original_width, original_height = img.size
                img = img.rotate(90, expand=True)

                # Resize and pad the image
                img_resized = resize_and_pad(img, target_size)

                # Calculate scaling and padding factors
                w_scale = target_size[0] / img.height
                h_scale = target_size[1] / img.width
                w_pad = (target_size[0] - img.height * w_scale) / 2
                h_pad = (target_size[1] - img.width * h_scale) / 2
                
                img_segmentations = [] # Store segmentations for this image
                for ann in img_anns:
                    if 'segmentation' in ann:
                        # Adjust segmentation coordinates
                        seg = []
                        for i in range(0, len(ann['segmentation'][0]), 2):
                            # Rotate the original points 90 degrees clockwise
                            x_original = ann['segmentation'][0][i]
                            y_original = ann['segmentation'][0][i+1]
                            
                            x_rotated = y_original
                            y_rotated = original_width - x_original
                            
                            # Scale and pad
                            x_scaled = x_rotated * w_scale + w_pad
                            y_scaled = y_rotated * h_scale + h_pad
                            
                            seg.append(x_scaled)
                            seg.append(y_scaled)
                        
                        img_segmentations.append(seg)
                        
                # Normalize image and add to lists
                images.append(np.array(img_resized, dtype=np.float32) / 255.0)
                all_segmentations.append(img_segmentations)
                filenames.append(img_info['file_name'])

            except Exception as e:
                logging.error(f"Error processing image {img_path}: {e}")

    return np.array(images), all_segmentations, filenames

def train_validate_split(data, segmentations, filenames, val_split=0.2): # Modified annotations to segmentations
    num_val = int(len(data) * val_split)
    
    # Shuffle data, annotations, and filenames together
    combined = list(zip(data, segmentations, filenames))
    np.random.shuffle(combined)
    data, segmentations, filenames = zip(*combined)
    
    # Split into training and validation sets
    train_data = data[:-num_val]
    train_segmentations = segmentations[:-num_val] # Modified annotations to segmentations
    train_filenames = filenames[:-num_val]
    
    val_data = data[-num_val:]
    val_segmentations = segmentations[-num_val:] # Modified annotations to segmentations
    val_filenames = filenames[-num_val:]
    
    return list(train_data), list(train_segmentations), list(train_filenames), list(val_data), list(val_segmentations), list(val_filenames) # Modified annotations to segmentations

def evaluate_model(model, X, annotations):
    predictions = model(X)
    # In a real scenario, you'd compare predictions with annotations
    # For now, we'll just return a random metric
    return np.random.rand()

################################################################################################

def compute_area(points):
    """Compute the area of the polygon using the shoelace formula."""
    return 0.5 * np.abs(np.dot(points[:, 0], np.roll(points[:, 1], 1)) - np.dot(points[:, 1], np.roll(points[:, 0], 1)))

def reorder_points(segmentation):
    """
    Reorder segmentation points to follow the top-left counter-clockwise convention.
    
    Args:
    segmentation (list): List of x, y coordinates [x1, y1, x2, y2, x3, y3, x4, y4]
    
    Returns:
    list: Reordered list of x, y coordinates
    """
    points = np.array(segmentation).reshape(-1, 2)
    
    # Find the top-left point (min x + min y)
    tl_idx = np.argmin(points.sum(axis=1))
    points = np.roll(points, -tl_idx, axis=0)
    
    # Ensure counter-clockwise order
    if compute_area(points) < 0:
        points = points[::-1] # Reverse the order of points
    
    return points.flatten().tolist()

def check_annotation_order(segmentation):
    """
    Check if the segmentation points follow the top-left counter-clockwise convention.
    
    Args:
    segmentation (list): List of x, y coordinates [x1, y1, x2, y2, x3, y3, x4, y4]
    
    Returns:
    bool: True if the order is correct, False otherwise
    """
    points = np.array(segmentation).reshape(-1, 2)
    
    # Check if it starts from top-left
    if np.argmin(points.sum(axis=1)) != 0:
        return False
    
    # Check if it's counter-clockwise
    return compute_area(points) > 0

def reorder_segmentations(segmentations):
    """
    Reorder points in segmentations to follow the top-left counter-clockwise convention.
    
    Args:
    segmentations (list): List of segmentation lists
    
    Returns:
    list: List of segmentation lists with reordered points
    """
    reordered_segmentations = []
    for segmentation in segmentations:
        # Ensure it's a polygon with 4 points
        if len(segmentation) == 8:
            reordered_segmentation = reorder_points(segmentation)
        else:
            reordered_segmentation = segmentation  # Keep as is if not 4 points
        reordered_segmentations.append(reordered_segmentation)
    return reordered_segmentations

def count_max_tiles(segmentations, filenames):
    max_tiles = 0
    max_tile_filename = ""
    for img_segs, filename in zip(segmentations, filenames):
        # Count tiles only if they are polygons with 4 points
        tile_count = sum(1 for seg in img_segs if len(seg) == 8)
        if tile_count > max_tiles:
            max_tiles = tile_count
            max_tile_filename = filename
    return max_tiles, max_tile_filename

################################################################################################

def convert_annotations_to_masks(coco, annotations, image_info, output_dir):
    """
    Convert COCO annotations to binary masks and save them.
    
    Args:
    coco (COCO): COCO object from pycocotools
    annotations (list): List of annotation dictionaries for an image
    image_info (dict): COCO image info dictionary
    output_dir (str): Directory to save the masks
    
    Returns:
    None
    """
    # Create a blank mask for the image
    mask = np.zeros((image_info['height'], image_info['width']), dtype=np.uint8)
    
    for ann in annotations:
        if 'segmentation' in ann and len(ann['segmentation'][0]) == 8:
            # Convert polygon to rle
            rle = mask_utils.frPyObjects([ann['segmentation'][0]], image_info['height'], image_info['width'])
            m = mask_utils.decode(rle)
            
            # Add this tile's mask to the overall mask
            mask = np.logical_or(mask, m[:, :, 0]).astype(np.uint8)
    
    # Save the mask
    filename = os.path.splitext(image_info['file_name'])[0] + '_mask.png'
    cv2.imwrite(os.path.join(output_dir, filename), mask * 255)


################################################################################################

class TileDataset(Dataset):
    def __init__(self, images, segmentations, filenames, max_tiles, coco): # Added filenames
        self.images = images
        self.segmentations = segmentations
        self.filenames = filenames
        self.max_tiles = max_tiles
        self.coco = coco
        self.image_ids = [coco.getImgIds()[i] for i in range(len(images))]

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image_id = self.image_ids[idx]
        img_info = self.coco.loadImgs(image_id)[0]
        img = self.images[idx]
        filename = self.filenames[idx]

        segmentations = self.segmentations[idx]

        # Convert PIL Image to tensor
        img_tensor = F.to_tensor(img)

        # Prepare target
        boxes = []
        labels = []

        for seg in segmentations:
            # Convert segmentation to bounding box
            x_coords = [seg[i] for i in range(0, len(seg), 2)]
            y_coords = [seg[i + 1] for i in range(0, len(seg), 2)]

            xmin = min(x_coords)
            xmax = max(x_coords)
            ymin = min(y_coords)
            ymax = max(y_coords)

            # Filter out invalid boxes with zero height or width
            if xmin < xmax and ymin < ymax:
                boxes.append([xmin, ymin, xmax, ymax])
                labels.append(1)  # Assuming all are tiles (class label 1)

        if not boxes:
            # Handle cases where no valid boxes are found
            boxes = torch.zeros((0, 4), dtype=torch.float32)
            labels = torch.zeros(0, dtype=torch.int64)
        else:
            # Convert lists to tensors
            boxes = torch.as_tensor(boxes, dtype=torch.float32)
            labels = torch.as_tensor(labels, dtype=torch.int64)

        # Create target dictionary
        target = {
            "boxes": boxes,
            "labels": labels,
            "image_id": torch.tensor([idx]),
            "area": (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0]) if boxes.numel() > 0 else torch.tensor([]),
            "iscrowd": torch.zeros((boxes.shape[0],), dtype=torch.int64) if boxes.numel() > 0 else torch.tensor([])
        }

        return img_tensor, target

################################################################################################

def get_model(num_keypoints):
    # Load a pre-trained Faster R-CNN model
    #model = torchvision.models.detection.fasterrcnn_resnet50_fpn(weights="DEFAULT")
    model = fasterrcnn_mobilenet_v3_large_fpn(weights=FasterRCNN_MobileNet_V3_Large_FPN_Weights.DEFAULT)
    
    # Get the number of input features for the classifier
    in_features = model.roi_heads.box_predictor.cls_score.in_features

    # Replace the pre-trained head with a new one
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, 2)  # 2 classes: background and tile

    # Modify the RPN to predict 8 values (4 points * 2 coordinates)
    #model.rpn.head.conv = torch.nn.Conv2d(256, 256 * 8, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))

    return model


def train_one_epoch(model, optimizer, data_loader, device, epoch, print_freq):
    model.train()
    
    # Simple metric tracking (instead of MetricLogger)
    start_time = time.time()
    epoch_loss = 0.0

    for batch_idx, (images, targets) in enumerate(data_loader):
        images = list(image.to(device) for image in images)
        
        targets_formatted = []
        for t in targets:
            target_dict = {
                'boxes': t['boxes'].to(device),
                'labels': t['labels'].to(device),
                'image_id': t['image_id'],
                'area': t['area'],
                'iscrowd': t['iscrowd']
            }
            targets_formatted.append(target_dict)

        loss_dict = model(images, targets_formatted)
        losses = sum(loss for loss in loss_dict.values())

        optimizer.zero_grad()
        losses.backward()
        optimizer.step()

        epoch_loss += losses.item()

        if batch_idx % print_freq == 0:
            print(f"Epoch: [{epoch}][{batch_idx}/{len(data_loader)}] "
                  f"Loss: {losses.item():.4f} "
                  f"Time: {time.time() - start_time:.4f}")

    print(f"Epoch {epoch} Loss: {epoch_loss / len(data_loader):.4f}")


def evaluate(model, data_loader, device):
    """
    Evaluates the model on a given dataset using Mean Average Precision (mAP).

    Args:
        model: The trained model.
        data_loader: DataLoader for the evaluation dataset.
        device: The device to run the evaluation on (e.g., 'cpu', 'cuda', 'mps').
    """

    model.eval()  # Set the model to evaluation mode

    metric = MeanAveragePrecision()

    with torch.no_grad():  # Disable gradient calculations during evaluation
        for images, targets in data_loader:
            images = list(image.to(device) for image in images)

            # Use the model to make predictions
            predictions = model(images)

            # Format targets to match predictions for mAP calculation
            formatted_targets = []
            for t in targets:
                formatted_targets.append({
                    'boxes': t['boxes'].to(device),
                    'labels': t['labels'].to(device)
                })
            
            metric.update(predictions, formatted_targets)

    # Compute the final mAP score
    result = metric.compute()

    print(f"mAP: {result['map'].item():.4f}")
    print(f"mAP_50: {result['map_50'].item():.4f}")
    print(f"mAP_75: {result['map_75'].item():.4f}")
    print(f"mAR: {result['mar_100'].item():.4f}")

################################################################################################

def main():
    annotation_file = 'instances_default.json'
    image_directory = './images'

    # Load your images, segmentations, and filenames
    all_images, all_segmentations, filenames = load_coco_data(annotation_file, image_directory, max_images = 10)

    # Reorder segmentations for each image
    all_segmentations_reordered = [reorder_segmentations(img_segs) for img_segs in all_segmentations]

    # Check the order of segmentations after reordering
    all_correct = True
    for img_segs in all_segmentations_reordered:
        for seg in img_segs:
            if len(seg) == 8:  # Check only if it's a 4-point polygon
                if not check_annotation_order(seg):
                    print(f"Segmentation {seg} is still incorrect after reordering.")
                    all_correct = False

    if all_correct:
        print("All segmentations are now in the correct order.")

    print(f"Total images: {len(all_images)}")
    
    # Split into training and validation sets
    X_train, y_train, train_filenames, X_val, y_val, val_filenames = train_validate_split(all_images, all_segmentations_reordered, filenames)

    # Count max tiles and get filename
    max_tiles, max_tile_filename = count_max_tiles(all_segmentations_reordered, filenames)
    print(f"Maximum number of tiles in any single image: {max_tiles}")
    print(f"This maximum occurs in the file: {max_tile_filename}")

    # Create COCO object for dataset
    coco = COCO(annotation_file)

    # Create datasets
    train_dataset = TileDataset(X_train, y_train, train_filenames, max_tiles, coco)
    val_dataset = TileDataset(X_val, y_val, val_filenames, max_tiles, coco)

    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True, num_workers=0, collate_fn=lambda x: tuple(zip(*x)))
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False, num_workers=0, collate_fn=lambda x: tuple(zip(*x)))

    # Get the model
    num_keypoints = 4  # We have 4 points for each tile
    model = get_model(num_keypoints)

    # Move model to the right device
    device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
    model.to(device)

    # Construct an optimizer
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=0.005, momentum=0.9, weight_decay=0.0005)

    # Learning rate scheduler
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.1)

    # Number of epochs
    num_epochs = 10

    for epoch in range(num_epochs):
        # Train for one epoch
        train_one_epoch(model, optimizer, train_loader, device, epoch, print_freq=1)

        # Update the learning rate
        lr_scheduler.step()

        # Evaluate on the validation dataset
        evaluate(model, val_loader, device=device)

    print("Training completed!")

main()