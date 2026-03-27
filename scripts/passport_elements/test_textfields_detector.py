#!/usr/bin/env python3
"""
Script to test text fields detector pipeline on dataset and calculate IoU metrics.
Only considers bounding boxes, not classes.
"""

import json
import sys
import os
from pathlib import Path
import numpy as np
from typing import List, Tuple, Dict
import argparse

# Add the project root to the path
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

from russian_docs_ocr.document_processing.pipeline_modules.textfields_detector import TextFieldsDetector


def calculate_iou(box1: List[float], box2: List[float]) -> float:
    """
    Calculate Intersection over Union (IoU) between two bounding boxes.
    
    Args:
        box1: [x1, y1, x2, y2] format
        box2: [x1, y1, x2, y2] format
        
    Returns:
        IoU value between 0 and 1
    """
    # Convert to [x1, y1, x2, y2] format if needed
    if len(box1) == 4 and len(box2) == 4:
        x1_1, y1_1, x2_1, y2_1 = box1
        x1_2, y1_2, x2_2, y2_2 = box2
    else:
        raise ValueError("Boxes must be in [x1, y1, x2, y2] format")
    
    # Calculate intersection
    x1_i = max(x1_1, x1_2)
    y1_i = max(y1_1, y1_2)
    x2_i = min(x2_1, x2_2)
    y2_i = min(y2_1, y2_2)
    
    if x2_i <= x1_i or y2_i <= y1_i:
        return 0.0
    
    intersection = (x2_i - x1_i) * (y2_i - y1_i)
    
    # Calculate union
    area1 = (x2_1 - x1_1) * (y2_1 - y1_1)
    area2 = (x2_2 - x1_2) * (y2_2 - y1_2)
    union = area1 + area2 - intersection
    
    return intersection / union if union > 0 else 0.0


def match_boxes(pred_boxes: List[List[float]], gt_boxes: List[List[float]], 
                iou_threshold: float = 0.5) -> Tuple[List[float], List[bool]]:
    """
    Match predicted boxes with ground truth boxes using IoU.
    
    Args:
        pred_boxes: List of predicted bounding boxes
        gt_boxes: List of ground truth bounding boxes
        iou_threshold: Minimum IoU for a match
        
    Returns:
        Tuple of (matched_ious, matched_flags)
    """
    if not pred_boxes or not gt_boxes:
        return [], []
    
    # Calculate IoU matrix
    iou_matrix = np.zeros((len(pred_boxes), len(gt_boxes)))
    for i, pred_box in enumerate(pred_boxes):
        for j, gt_box in enumerate(gt_boxes):
            iou_matrix[i, j] = calculate_iou(pred_box, gt_box)
    
    # Greedy matching
    matched_ious = []
    matched_flags = []
    used_gt = set()
    
    for pred_idx in range(len(pred_boxes)):
        best_iou = 0
        best_gt_idx = -1
        
        for gt_idx in range(len(gt_boxes)):
            if gt_idx not in used_gt and iou_matrix[pred_idx, gt_idx] > best_iou:
                best_iou = iou_matrix[pred_idx, gt_idx]
                best_gt_idx = gt_idx
        
        if best_iou >= iou_threshold and best_gt_idx != -1:
            matched_ious.append(best_iou)
            matched_flags.append(True)
            used_gt.add(best_gt_idx)
        else:
            matched_ious.append(0.0)
            matched_flags.append(False)
    
    return matched_ious, matched_flags


def load_coco_annotations(annotations_path: Path) -> Dict:
    """
    Load COCO format annotations.
    
    Args:
        annotations_path: Path to COCO annotations file
        
    Returns:
        Dictionary with image_id -> annotations mapping
    """
    with open(annotations_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # Group annotations by image_id
    annotations_by_image = {}
    for ann in data['annotations']:
        image_id = ann['image_id']
        if image_id not in annotations_by_image:
            annotations_by_image[image_id] = []
        annotations_by_image[image_id].append(ann)
    
    # Create image_id to filename mapping
    image_id_to_filename = {}
    for img in data['images']:
        image_id_to_filename[img['id']] = img['file_name']
    
    return {
        'annotations_by_image': annotations_by_image,
        'image_id_to_filename': image_id_to_filename
    }


def evaluate_textfields_detector(dataset_path: Path, model_format: str = 'ONNX', 
                                device: str = 'cpu', iou_threshold: float = 0.5,
                                verbose: bool = False) -> Dict:
    """
    Evaluate text fields detector on dataset.
    
    Args:
        dataset_path: Path to dataset directory
        model_format: Model format (ONNX, TFlite, etc.)
        device: Device to run inference on
        iou_threshold: IoU threshold for matching
        verbose: Verbose output
        
    Returns:
        Dictionary with evaluation metrics
    """
    # Initialize detector
    detector = TextFieldsDetector(model_format=model_format, device=device, verbose=verbose)
    
    # Load annotations
    annotations_path = dataset_path / 'train' / 'instances_Train.json'
    if not annotations_path.exists():
        raise FileNotFoundError(f"Annotations file not found: {annotations_path}")
    
    coco_data = load_coco_annotations(annotations_path)
    annotations_by_image = coco_data['annotations_by_image']
    image_id_to_filename = coco_data['image_id_to_filename']
    
    # Evaluation metrics
    total_images = 0
    total_gt_boxes = 0
    total_pred_boxes = 0
    total_matched_boxes = 0
    all_ious = []
    
    results_per_image = []
    
    # Process each image with annotations
    for image_id, annotations in annotations_by_image.items():
        if not annotations:
            continue
            
        filename = image_id_to_filename.get(image_id)
        if not filename:
            continue
            
        image_path = dataset_path / 'train' / filename
        if not image_path.exists():
            print(f"Warning: Image file not found: {image_path}")
            continue
        
        try:
            # Get ground truth boxes
            gt_boxes = []
            for ann in annotations:
                bbox = ann['bbox']  # [x, y, width, height]
                # Convert to [x1, y1, x2, y2] format
                x1, y1, w, h = bbox
                gt_boxes.append([x1, y1, x1 + w, y1 + h])
            
            # Run detector
            result = detector.predict(str(image_path))
            pred_boxes_raw = result.get('TextFieldsDetector', {}).get('bbox', [])
            
            # Convert predicted boxes to [x1, y1, x2, y2] format
            pred_boxes = []
            for box in pred_boxes_raw:
                if len(box) >= 4:
                    # YOLO detector returns [x1, y1, x2, y2, confidence, class_id, ...]
                    pred_boxes.append([float(box[0]), float(box[1]), float(box[2]), float(box[3])])
            
            # Match boxes
            matched_ious, matched_flags = match_boxes(pred_boxes, gt_boxes, iou_threshold)
            
            # Update metrics
            total_images += 1
            total_gt_boxes += len(gt_boxes)
            total_pred_boxes += len(pred_boxes)
            total_matched_boxes += sum(matched_flags)
            all_ious.extend([iou for iou, matched in zip(matched_ious, matched_flags) if matched])
            
            # Store per-image results
            image_result = {
                'filename': filename,
                'gt_boxes': len(gt_boxes),
                'pred_boxes': len(pred_boxes),
                'matched_boxes': sum(matched_flags),
                'avg_iou': np.mean([iou for iou, matched in zip(matched_ious, matched_flags) if matched]) if any(matched_flags) else 0.0,
                'max_iou': max(matched_ious) if matched_ious else 0.0
            }
            results_per_image.append(image_result)
            
            if verbose:
                print(f"Image {filename}: GT={len(gt_boxes)}, Pred={len(pred_boxes)}, "
                      f"Matched={sum(matched_flags)}, Avg IoU={image_result['avg_iou']:.3f}")
                
        except Exception as e:
            print(f"Error processing {filename}: {e}")
            continue
    
    # Calculate final metrics
    precision = total_matched_boxes / total_pred_boxes if total_pred_boxes > 0 else 0.0
    recall = total_matched_boxes / total_gt_boxes if total_gt_boxes > 0 else 0.0
    f1_score = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    mean_iou = np.mean(all_ious) if all_ious else 0.0
    
    results = {
        'total_images': total_images,
        'total_gt_boxes': total_gt_boxes,
        'total_pred_boxes': total_pred_boxes,
        'total_matched_boxes': total_matched_boxes,
        'precision': precision,
        'recall': recall,
        'f1_score': f1_score,
        'mean_iou': mean_iou,
        'iou_threshold': iou_threshold,
        'results_per_image': results_per_image
    }
    
    return results


def main():
    parser = argparse.ArgumentParser(description='Evaluate text fields detector on dataset')
    parser.add_argument('--dataset_path', type=str, 
                       default='dataset/dataset_for_passport_elements_detector',
                       help='Path to dataset directory')
    parser.add_argument('--model_format', type=str, default='ONNX',
                       choices=['ONNX', 'TFlite', 'OpenVINO', 'CoreML'],
                       help='Model format to use')
    parser.add_argument('--device', type=str, default='cpu',
                       help='Device to run inference on')
    parser.add_argument('--iou_threshold', type=float, default=0.5,
                       help='IoU threshold for matching')
    parser.add_argument('--verbose', action='store_true',
                       help='Verbose output')
    parser.add_argument('--output_file', type=str,
                       help='Output file to save results')
    
    args = parser.parse_args()
    
    dataset_path = Path(args.dataset_path)
    if not dataset_path.exists():
        print(f"Error: Dataset path does not exist: {dataset_path}")
        return 1
    
    print(f"Evaluating text fields detector on dataset: {dataset_path}")
    print(f"Model format: {args.model_format}")
    print(f"Device: {args.device}")
    print(f"IoU threshold: {args.iou_threshold}")
    print("-" * 50)
    
    try:
        results = evaluate_textfields_detector(
            dataset_path=dataset_path,
            model_format=args.model_format,
            device=args.device,
            iou_threshold=args.iou_threshold,
            verbose=args.verbose
        )
        
        # Print results
        print("\nEvaluation Results:")
        print(f"Total images processed: {results['total_images']}")
        print(f"Total ground truth boxes: {results['total_gt_boxes']}")
        print(f"Total predicted boxes: {results['total_pred_boxes']}")
        print(f"Total matched boxes: {results['total_matched_boxes']}")
        print(f"Precision: {results['precision']:.4f}")
        print(f"Recall: {results['recall']:.4f}")
        print(f"F1 Score: {results['f1_score']:.4f}")
        print(f"Mean IoU: {results['mean_iou']:.4f}")
        
        # Save results if output file specified
        if args.output_file:
            output_path = Path(args.output_file)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Save detailed results
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(results, f, indent=2, ensure_ascii=False)
            
            # Save summary CSV
            csv_path = output_path.with_suffix('.csv')
            import csv
            with open(csv_path, 'w', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                writer.writerow(['Metric', 'Value'])
                writer.writerow(['Total Images', results['total_images']])
                writer.writerow(['Total GT Boxes', results['total_gt_boxes']])
                writer.writerow(['Total Pred Boxes', results['total_pred_boxes']])
                writer.writerow(['Total Matched Boxes', results['total_matched_boxes']])
                writer.writerow(['Precision', f"{results['precision']:.4f}"])
                writer.writerow(['Recall', f"{results['recall']:.4f}"])
                writer.writerow(['F1 Score', f"{results['f1_score']:.4f}"])
                writer.writerow(['Mean IoU', f"{results['mean_iou']:.4f}"])
                writer.writerow(['IoU Threshold', results['iou_threshold']])
            
            print(f"\nResults saved to: {output_path}")
            print(f"Summary saved to: {csv_path}")
        
        return 0
        
    except Exception as e:
        print(f"Error during evaluation: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == '__main__':
    exit(main()) 