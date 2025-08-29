import json
import numpy as np
from pycocotools.coco import COCO

def compute_area(points):
    """Compute the area of the polygon using the shoelace formula."""
    return 0.5 * np.abs(np.dot(points[:, 0], np.roll(points[:, 1], 1)) - np.dot(points[:, 1], np.roll(points[:, 0], 1)))

def check_annotation_order(segmentation):
    """
    Check if segmentation points follow top-left counter-clockwise convention.
    
    Args:
    segmentation (list): List of x, y coordinates [x1, y1, x2, y2, x3, y3, x4, y4]
    
    Returns:
    bool: True if order is correct, False otherwise
    """
    if len(segmentation) != 8:
        return None  # Not a 4-point polygon
        
    points = np.array(segmentation).reshape(-1, 2)
    
    # Check if it starts from top-left
    if np.argmin(points.sum(axis=1)) != 0:
        return False
    
    # Check if it's counter-clockwise
    return compute_area(points) > 0

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

def check_all_annotations(annotation_file):
    """Check annotation order for all polygons in COCO file."""
    coco = COCO(annotation_file)
    
    total_annotations = 0
    correct_order = 0
    incorrect_order = 0
    non_quad_polygons = 0
    
    print(f"Checking annotations in: {annotation_file}")
    print("-" * 50)
    
    for ann_id, annotation in coco.anns.items():
        if 'segmentation' not in annotation:
            continue
            
        seg = annotation['segmentation'][0]  # First segmentation
        
        if len(seg) != 8:
            non_quad_polygons += 1
            continue
            
        total_annotations += 1
        order_check = check_annotation_order(seg)
        
        if order_check:
            correct_order += 1
        else:
            incorrect_order += 1
            
        # Show first few problematic examples
        if not order_check and incorrect_order <= 3:
            img_info = coco.loadImgs(annotation['image_id'])[0]
            print(f"Incorrect order in {img_info['file_name']}: {seg}")
            corrected = reorder_points(seg)
            print(f"Corrected:   {corrected}")
            print()
    
    print(f"Results:")
    print(f"  4-point polygons: {total_annotations}")
    print(f"  Correct order: {correct_order}")
    print(f"  Incorrect order: {incorrect_order}")
    print(f"  Non-quad polygons: {non_quad_polygons}")
    
    if total_annotations > 0:
        print(f"  Accuracy: {correct_order/total_annotations:.1%}")
    
    return {
        'total': total_annotations,
        'correct': correct_order,
        'incorrect': incorrect_order,
        'non_quad': non_quad_polygons
    }

if __name__ == "__main__":
    annotation_file = 'instances_fixed.json'
    results = check_all_annotations(annotation_file)
