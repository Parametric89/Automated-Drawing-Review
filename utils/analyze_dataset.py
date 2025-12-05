import argparse
import subprocess
import sys
from pathlib import Path


def run(cmd):
    try:
        print("$", " ".join(cmd))
        out = subprocess.run(cmd, check=True, capture_output=True, text=True)
        if out.stdout:
            print(out.stdout.strip())
        if out.stderr:
            print(out.stderr.strip())
        return True
    except subprocess.CalledProcessError as e:
        print("ERROR:", e.stdout, e.stderr)
        return False


def main():
    ap = argparse.ArgumentParser(description="One-click dataset QA: schema, pairing, leakage, overlays, distribution")
    ap.add_argument("--base-dir", required=True, help="Dataset base directory (e.g., datasets/rcp_dual_seg_v3)")
    ap.add_argument("--overlay-limit", type=int, default=80)
    args = ap.parse_args()

    base = Path(args.base_dir)
    if not base.exists():
        raise SystemExit(f"Base dir not found: {base}")

    py = sys.executable
    # Discover dataset leaf folders dynamically (e.g., augmented1k_gtcenter1k_11_22_33)
    splits = ["train", "val", "test"]
    discovered = set()
    for s in splits:
        img_split_dir = base / "images" / s
        if not img_split_dir.exists():
            continue
        for child in img_split_dir.iterdir():
            if child.is_dir():
                name = child.name
                # Skip raw/fullsize folders
                if name.lower() in {"fullsize", "fullsize_panels"}:
                    continue
                discovered.add(name)
    # Fallback to legacy defaults if nothing found
    sets = sorted(discovered) if discovered else ["tiled1k", "cropped1k", "augmented1k"]

    print("=== DATASET QA ===")
    print("Base:", base)

    # Validate labels and pairing
    for leaf in sets:
        print(f"\n--- Validating {leaf} ---")
        label_dirs = []
        for s in splits:
            img_dir = base / "images" / s / leaf
            lbl_dir = base / "labels" / s / leaf
            if lbl_dir.exists():
                label_dirs.append(str(lbl_dir))
                run([py, "utils/validate_yolo_labels.py", "--labels-dir", str(lbl_dir)])
            if img_dir.exists() and lbl_dir.exists():
                run([py, "utils/check_pairing.py", "--images", str(img_dir), "--labels", str(lbl_dir)])

        # Cross-split leakage
        run([py, "utils/check_split_leakage.py", "--base", str(base), "--folder", leaf])

        # Overlays (sample)
        for s in splits:
            img_dir = base / "images" / s / leaf
            lbl_dir = base / "labels" / s / leaf
            out_dir = Path("runs/overlays") / f"{base.name}_{leaf}_{s}"
            if img_dir.exists() and lbl_dir.exists():
                run([py, "utils/overlay_labels_folder.py", "--images", str(img_dir), "--labels", str(lbl_dir), "--out", str(out_dir), "--limit", str(args.overlay_limit)])

        # Class distribution
        if label_dirs:
            print(f"\nClass distribution for {leaf}:")
            run([py, "utils/class_distribution.py", "--dirs", *label_dirs])

    print("\nDataset QA completed.")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        # Make analyzer non-fatal when called via Hub; print and exit 0
        print("ERROR:", e)
        sys.exit(0)


