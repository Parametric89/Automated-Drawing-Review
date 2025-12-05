"""
IoU sweep on YOLOv8s winning weights at fixed confidence.

Purpose:
    Determine whether remaining false negatives are near misses (geometry issue)
    or genuine misses (content/coverage issue).

Procedure:
    - Hold confidence at 0.40.
    - Sweep IoU thresholds: {0.20, 0.30, 0.40, 0.50}.
    - For each IoU, log overall recall and FP/image using the shared evaluator.
    - Save results to JSON alongside the run directory.
"""

from pathlib import Path
import argparse
import json

from run_v7_ablation_refined import evaluate_val_metrics


def parse_args():
    parser = argparse.ArgumentParser(description="IoU sweep on validation set.")
    parser.add_argument(
        "--run-dir",
        type=Path,
        default=Path(r"runs/train/v7_speed_yolov8s_winning2"),
        help="Training run directory containing weights/best.pt",
    )
    parser.add_argument(
        "--conf",
        type=float,
        default=0.40,
        help="Confidence threshold to hold fixed during the IoU sweep.",
    )
    parser.add_argument(
        "--ious",
        type=float,
        nargs="+",
        default=[0.20, 0.30, 0.40, 0.50],
        help="IoU thresholds to evaluate (space separated).",
    )
    parser.add_argument(
        "--out",
        type=Path,
        help="Optional JSON output path. Defaults to <run_dir>/iou_sweep_confXX.json",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_dir = args.run_dir
    conf_threshold = args.conf
    iou_sweep = args.ious
    default_out = f"iou_sweep_conf{conf_threshold:.2f}.json"
    out_path = args.out or (run_dir / default_out)

    if not run_dir.exists():
        raise FileNotFoundError(f"Run directory does not exist: {run_dir}")

    print("=" * 80)
    print(f"IOU SWEEP @ fixed confidence (conf = {conf_threshold:.2f})")
    print("=" * 80)
    print()

    results = []

    for iou in iou_sweep:
        print(f"{'=' * 80}")
        print(f"IoU threshold: {iou:.2f}")
        print(f"{'=' * 80}")

        metrics = evaluate_val_metrics(
            run_dir=run_dir,
            conf_threshold=conf_threshold,
            iou_threshold=iou,
        )

        if metrics is None:
            print("No metrics returned (check weights/path)")
            print()
            continue

        recall = metrics["recall_at_04"]
        fp_per_image = metrics["fp_per_image"]
        precision = metrics['total_tp'] / max((metrics['total_tp'] + metrics['total_fp']), 1)

        print(f"Recall@IoU {iou:.2f} : {recall:.3f}")
        print(f"Precision           : {precision:.3f}")
        print(f"FP/image            : {fp_per_image:.2f}")
        print(f"Worst-bin ({metrics['worst_bin_name']}) recall: {metrics['worst_bin_recall']:.3f}")
        print(f"Small-panel recall  : {metrics['small_panel_recall']:.3f}")
        print(f"TP={metrics['total_tp']}  FP={metrics['total_fp']}  FN={metrics['total_fn']}  GT={metrics['total_gt']}")
        print()

        results.append(
            {
                "iou": iou,
                "conf": conf_threshold,
                "recall": recall,
                "precision": precision,
                "fp_per_image": fp_per_image,
                "worst_bin_recall": metrics["worst_bin_recall"],
                "worst_bin_name": metrics["worst_bin_name"],
                "small_panel_recall": metrics["small_panel_recall"],
                "total_tp": metrics["total_tp"],
                "total_fp": metrics["total_fp"],
                "total_fn": metrics["total_fn"],
                "total_gt": metrics["total_gt"],
            }
        )

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    print("=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"{'IoU':<6} {'Recall':<10} {'Precision':<12} {'FP/image':<10}")
    print("-" * 46)
    for r in results:
        print(f"{r['iou']:<6.2f} {r['recall']:<10.3f} {r['precision']:<12.3f} {r['fp_per_image']:<10.2f}")
    print()
    print("Interpretation:")
    print("  • If recall jumps at low IoU (e.g., >0.90 at IoU=0.20) but drops by IoU=0.40,")
    print("    boxes are often near-miss — focus on label geometry/tiling.")
    print("  • If recall remains flat across IoU levels, misses are genuine —")
    print("    focus on coverage, hard negatives, or capacity.")
    print()
    print(f"Sweep complete. Results saved to: {out_path}")


if __name__ == "__main__":
    main()



