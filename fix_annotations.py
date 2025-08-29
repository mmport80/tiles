import json
import numpy as np
from pycocotools.coco import COCO

def compute_area(points):
    """Compute the area of the polygon using the shoelace formula."""
    return 0.5 * np.abs(np.dot(points[:, 0], np.roll(points[:, 1], 1)) - np.dot(points[:, 1], np.roll(points[:, 0], 1)))

def reorder_points(segmentation):
    """Reorder segmentation points to follow top-left counter-clockwise convention."""
    points = np.array(segmentation).reshape(-1, 2)
    
    # Find the top-left point (min x + min y)
    tl_idx = np.argmin(points.sum(axis=1))
    points = np.roll(points, -tl_idx, axis=0)
    
    # Ensure counter-clockwise order
    if compute_area(points) < 0:
        points = points[::-1]  # Reverse order
    
    return points.flatten().tolist()

def fix_annotations(input_file, output_file):
    """Fix annotations and save cleaned COCO file."""
    
    # Load original data
    with open(input_file, 'r') as f:
        coco_data = json.load(f)
    
    print(f"Original annotations: {len(coco_data['annotations'])}")
    
    # Filter and fix annotations
    fixed_annotations = []
    filtered_count = 0
    reordered_count = 0
    
    for ann in coco_data['annotations']:
        if 'segmentation' not in ann:
            continue
            
        seg = ann['segmentation'][0]
        
        # Filter out non-quad polygons
        if len(seg) != 8:
            filtered_count += 1
            continue
            
        # Reorder points
        reordered_seg = reorder_points(seg)
        
        # Check if reordering was needed
        if reordered_seg != seg:
            reordered_count += 1
            
        # Update annotation
        ann['segmentation'][0] = reordered_seg
        fixed_annotations.append(ann)
    
    # Update annotations in data
    coco_data['annotations'] = fixed_annotations
    
    # Save fixed file
    with open(output_file, 'w') as f:
        json.dump(coco_data, f)
    
    print(f"Filtered out: {filtered_count} non-quad polygons")
    print(f"Reordered: {reordered_count} annotations")
    print(f"Final annotations: {len(fixed_annotations)}")
    print(f"Saved to: {output_file}")
    
    return {
        'original': len(coco_data['annotations']) + filtered_count,
        'filtered': filtered_count,
        'reordered': reordered_count,
        'final': len(fixed_annotations)
    }

if __name__ == "__main__":
    input_file = 'instances_default.json'
    output_file = 'instances_fixed.json'
    
    results = fix_annotations(input_file, output_file)
