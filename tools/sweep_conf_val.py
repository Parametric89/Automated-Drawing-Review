"""
Confidence sweep on validation set for a trained model.

Sweeps confidence thresholds and evaluates Recall@0.4, FP/image, worst-bin recall,
and small-panel recall to understand model behavior at different confidence levels.
"""

from pathlib import Path
import argparse
import json

from run_v7_ablation_refined import evaluate_val_metrics, IOU_THRESHOLD

DEFAULT_RUN_DIR = Path(r"runs/train/v7_speed_yolov8s_winning2")
DEFAULT_CONF_SWEEP = [0.05, 0.10, 0.20, 0.30, 0.40]


def parse_args():
    parser = argparse.ArgumentParser(description="Confidence sweep on validation set.")
    parser.add_argument(
        "--run-dir",
        type=Path,
        default=DEFAULT_RUN_DIR,
        help="Training run directory containing weights/best.pt",
    )
    parser.add_argument(
        "--confs",
        type=float,
        nargs="+",
        default=DEFAULT_CONF_SWEEP,
        help="Confidence thresholds to evaluate (space separated).",
    )
    parser.add_argument(
        "--out",
        type=Path,
        help="Optional JSON output path. Defaults to <run_dir>/conf_sweep_val.json",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    run_dir = args.run_dir
    conf_sweep = args.confs
    out_path = args.out or (run_dir / "conf_sweep_val.json")

    print("=" * 80)
    print("CONFIDENCE SWEEP ON VAL SET")
    print("=" * 80)
    print(f"Run dir: {run_dir}")
    print(f"IoU threshold fixed at: {IOU_THRESHOLD}")
    print()

    if not run_dir.exists():
        print(f"ERROR: Run directory does not exist: {run_dir}")
        print("Please update --run-dir to point to your winning model's run directory.")
        return

    results = []

    for conf in conf_sweep:
        print("=" * 80)
        print(f"Evaluating at conf = {conf:.2f}")
        print("=" * 80)

        metrics = evaluate_val_metrics(
            run_dir=run_dir,
            conf_threshold=conf,
            iou_threshold=IOU_THRESHOLD,
        )

        if metrics is None:
            print("No metrics returned (check weights path).")
            continue

        recall = metrics['recall_at_04']
        worst_bin_recall = metrics['worst_bin_recall']
        worst_bin_name = metrics['worst_bin_name']
        fp_per_image = metrics['fp_per_image']
        small_panel_recall = metrics['small_panel_recall']
        precision = metrics['total_tp'] / max((metrics['total_tp'] + metrics['total_fp']), 1)

        print(f"Recall@0.4      : {recall:.3f}")
        print(f"Precision       : {precision:.3f}")
        print(f"Worst-bin ({worst_bin_name}) recall: {worst_bin_recall:.3f}")
        print(f"FP/image        : {fp_per_image:.2f}")
        print(f"Small-panel R   : {small_panel_recall:.3f}")
        print(f"TP: {metrics['total_tp']}, FP: {metrics['total_fp']}, FN: {metrics['total_fn']}, GT: {metrics['total_gt']}")
        print()

        results.append({
            'conf': conf,
            'recall_at_04': recall,
            'precision': precision,
            'worst_bin_recall': worst_bin_recall,
            'worst_bin_name': worst_bin_name,
            'fp_per_image': fp_per_image,
            'small_panel_recall': small_panel_recall,
            'total_tp': metrics['total_tp'],
            'total_fp': metrics['total_fp'],
            'total_fn': metrics['total_fn'],
            'total_gt': metrics['total_gt'],
        })

    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)

    print("=" * 80)
    print("Sweep complete.")
    print(f"Saved results to: {out_path}")
    print()
    print("Analysis:")
    print("- Does Recall@0.4 increase meaningfully when conf → 0.05?")
    print("  If it stays stuck around ~0.7, the model genuinely isn't seeing the missing panels.")
    print("- How badly does FP/image blow up at low conf?")
    print("  That tells you what calibration might buy you later.")


if __name__ == "__main__":
    main()

