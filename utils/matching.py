from __future__ import annotations

from typing import Dict, List, Tuple
import numpy as np


def calculate_iou(box1: Dict[str, float], box2: Dict[str, float]) -> float:
	"""
	Calculate IoU between two normalized YOLO boxes given as dicts:
	{ 'x_center': cx, 'y_center': cy, 'width': w, 'height': h }
	Coordinates are in [0,1] relative terms and interpreted consistently across the codebase.
	"""
	x1_1 = box1['x_center'] - box1['width'] / 2.0
	y1_1 = box1['y_center'] - box1['height'] / 2.0
	x2_1 = box1['x_center'] + box1['width'] / 2.0
	y2_1 = box1['y_center'] + box1['height'] / 2.0

	x1_2 = box2['x_center'] - box2['width'] / 2.0
	y1_2 = box2['y_center'] - box2['height'] / 2.0
	x2_2 = box2['x_center'] + box2['width'] / 2.0
	y2_2 = box2['y_center'] + box2['height'] / 2.0

	x1_i = max(x1_1, x1_2)
	y1_i = max(y1_1, y1_2)
	x2_i = min(x2_1, x2_2)
	y2_i = min(y2_1, y2_2)

	if x2_i <= x1_i or y2_i <= y1_i:
		return 0.0

	intersection = (x2_i - x1_i) * (y2_i - y1_i)
	area1 = box1['width'] * box1['height']
	area2 = box2['width'] * box2['height']
	union = area1 + area2 - intersection
	return float(intersection / union) if union > 0 else 0.0


def hungarian_matching(pred_boxes: List[Dict[str, float]],
                       gt_boxes: List[Dict[str, float]],
                       iou_threshold: float) -> Tuple[List[Tuple[int, int]], List[int], List[int]]:
	"""
	Greedy one-to-one matching approximating Hungarian behavior:
	- Build IoU matrix
	- Consider only pairs >= iou_threshold
	- Sort by IoU desc, then take each pair if both prediction and GT are unused
	Returns:
	- matched_pairs: list of (pred_idx, gt_idx)
	- unmatched_preds: list of pred indices
	- unmatched_gts: list of gt indices
	"""
	if not pred_boxes or not gt_boxes:
		return [], list(range(len(pred_boxes))), list(range(len(gt_boxes)))

	iou_matrix = np.zeros((len(pred_boxes), len(gt_boxes)), dtype=np.float32)
	for i, pb in enumerate(pred_boxes):
		for j, gb in enumerate(gt_boxes):
			iou_matrix[i, j] = calculate_iou(pb, gb)

	candidates: List[Tuple[float, int, int]] = []
	for i in range(len(pred_boxes)):
		for j in range(len(gt_boxes)):
			if iou_matrix[i, j] >= iou_threshold:
				candidates.append((float(iou_matrix[i, j]), i, j))
	candidates.sort(reverse=True)

	matched_pairs: List[Tuple[int, int]] = []
	matched_preds = set()
	matched_gts = set()
	for iou, i, j in candidates:
		if i not in matched_preds and j not in matched_gts:
			matched_pairs.append((i, j))
			matched_preds.add(i)
			matched_gts.add(j)

	unmatched_preds = [i for i in range(len(pred_boxes)) if i not in matched_preds]
	unmatched_gts = [j for j in range(len(gt_boxes)) if j not in matched_gts]
	return matched_pairs, unmatched_preds, unmatched_gts


def match_and_count(pred_boxes: List[Dict[str, float]],
                    gt_boxes: List[Dict[str, float]],
                    iou_thresh: float) -> Tuple[int, int, int]:
	"""
	Return (tp, fp, fn) using greedy Hungarian-style one-to-one matching at IoU >= iou_thresh.
	Inputs are normalized YOLO dict boxes for consistency across evaluators.
	"""
	matched_pairs, unmatched_preds, unmatched_gts = hungarian_matching(pred_boxes, gt_boxes, iou_thresh)
	tp = len(matched_pairs)
	fp = len(unmatched_preds)
	fn = len(unmatched_gts)
	return tp, fp, fn



