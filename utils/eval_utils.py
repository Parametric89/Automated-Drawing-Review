from __future__ import annotations

from pathlib import Path
import csv
from typing import Dict


def append_eval_row(csv_path: str | Path, row: Dict[str, int | float | str]) -> None:
	"""
	Append one evaluation summary row; writes header on first use.
	Columns: gt_set, resolution, total_gt, matched_gt_0_4, total_fp, num_images
	"""
	header = ['gt_set', 'resolution', 'total_gt', 'matched_gt_0_4', 'total_fp', 'num_images']
	p = Path(csv_path)
	exists = p.exists()
	p.parent.mkdir(parents=True, exist_ok=True)
	with p.open('a', newline='', encoding='utf-8') as f:
		w = csv.DictWriter(f, fieldnames=header)
		if not exists:
			w.writeheader()
		w.writerow(row)



