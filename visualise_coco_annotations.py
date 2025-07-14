import json
import os
import cv2
import numpy as np
from pycocotools import mask as mask_utils
from collections import defaultdict

def load_coco_annotations(json_file):
    with open(json_file, 'r') as f:
        return json.load(f)

def draw_segmentation(image, segmentation, color):
    if isinstance(segmentation, dict):  # RLE format
        mask = mask_utils.decode(segmentation)
    elif isinstance(segmentation, list):  # Polygon format
        mask = mask_utils.decode(mask_utils.frPyObjects(segmentation, image.shape[0], image.shape[1]))
    else:
        raise ValueError(f"Unsupported segmentation format: {type(segmentation)}")
    
    if mask.ndim == 2:
        mask = mask[:,:,np.newaxis]
    elif mask.ndim == 3 and mask.shape[2] > 1:
        mask = np.any(mask, axis=2)
        mask = mask[:,:,np.newaxis]
    
    overlay = np.full(image.shape, color, dtype=np.uint8)
    alpha = 0.5
    cv2.addWeighted(np.where(mask == 1, overlay, image).astype(np.uint8), 
                    alpha, 
                    image, 
                    1 - alpha, 
                    0, 
                    image)
    return image

def visualize_annotations(annotations_file, images_folder, output_folder):
    coco_data = load_coco_annotations(annotations_file)
    
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    # Create a mapping of image_id to file_name for quick lookup
    image_id_to_file = {img['id']: img['file_name'] for img in coco_data['images']}
    
    # Group annotations by image_id
    annotations_by_image = defaultdict(list)
    for annotation in coco_data['annotations']:
        annotations_by_image[annotation['image_id']].append(annotation)
    
    for image_id, annotations in annotations_by_image.items():
        image_file = image_id_to_file[image_id]
        image_path = os.path.join(images_folder, image_file)
        
        if not os.path.exists(image_path):
            print(f"Image not found: {image_path}")
            continue
        
        image = cv2.imread(image_path)
        if image is None:
            print(f"Failed to load image: {image_path}")
            continue
        
        # Convert BGR to RGB
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        for annotation in annotations:
            # Generate a random color for this annotation
            color = np.random.randint(0, 255, 3).tolist()
            
            try:
                # Draw the segmentation
                image = draw_segmentation(image, annotation['segmentation'], color)
            except Exception as e:
                print(f"Error processing annotation for image {image_file}: {str(e)}")
                continue
        
        # Convert back to BGR for saving
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        
        # Save the annotated image
        output_path = os.path.join(output_folder, f"annotated_{image_file}")
        cv2.imwrite(output_path, image)
        print(f"Saved annotated image: {output_path}")

if __name__ == "__main__":
    annotations_file = "instances_default.json"
    images_folder = "images"
    output_folder = "output"
    
    visualize_annotations(annotations_file, images_folder, output_folder)