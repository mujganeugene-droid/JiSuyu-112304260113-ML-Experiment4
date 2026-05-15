from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.append(str(SCRIPT_DIR))

from eval_map50 import load_ground_truth, load_predictions, score_predictions
from make_submission import collect_model_predictions, merge_predictions, write_submission


def parse_list(raw: str, cast) -> list:
    return [cast(item.strip()) for item in raw.split(",") if item.strip()]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Search inference parameters on a labeled split.")
    parser.add_argument("--models", nargs="+", required=True, help="One or more model checkpoint paths")
    parser.add_argument("--image-dir", default="val/images", help="Labeled image directory")
    parser.add_argument("--label-dir", default="val/labels", help="Label directory")
    parser.add_argument("--num-classes", type=int, default=15, help="Number of classes")
    parser.add_argument("--confs", default="0.001,0.005,0.01,0.02", help="Comma-separated conf thresholds")
    parser.add_argument("--ious", default="0.5,0.6,0.7", help="Comma-separated model NMS IoU thresholds")
    parser.add_argument("--imgsizes", default="640", help="Comma-separated inference image sizes")
    parser.add_argument("--max-dets", default="50,100", help="Comma-separated max_det values")
    parser.add_argument("--device", default="0", help="CUDA device id or cpu")
    parser.add_argument("--batch", type=int, default=16, help="Inference batch size")
    parser.add_argument("--half", action="store_true", help="Use FP16 inference when supported")
    parser.add_argument("--tta", action="store_true", help="Enable TTA during search")
    parser.add_argument("--fusion", choices=["none", "wbf"], default="wbf", help="How to combine models")
    parser.add_argument("--fusion-iou", type=float, default=0.55, help="IoU threshold for fusion")
    parser.add_argument("--topk", type=int, default=10, help="How many best settings to print")
    parser.add_argument("--save-csv", default="search_results.csv", help="Where to save search results")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    image_dir = Path(args.image_dir).resolve()
    label_dir = Path(args.label_dir).resolve()
    image_paths = sorted(path for path in image_dir.iterdir() if path.is_file())
    image_names = [path.name for path in image_paths]
    ground_truth = load_ground_truth(image_dir, label_dir)

    rows: list[dict[str, object]] = []
    for conf in parse_list(args.confs, float):
        for iou in parse_list(args.ious, float):
            for imgsz in parse_list(args.imgsizes, int):
                for max_det in parse_list(args.max_dets, int):
                    print(f"Evaluating conf={conf} iou={iou} imgsz={imgsz} max_det={max_det}")
                    per_model_predictions = [
                        collect_model_predictions(
                            model_path=model_path,
                            image_paths=image_paths,
                            conf=conf,
                            iou=iou,
                            imgsz=imgsz,
                            device=args.device,
                            batch=args.batch,
                            half=args.half,
                            max_det=max_det,
                            tta=args.tta,
                        )
                        for model_path in args.models
                    ]
                    merged = merge_predictions(
                        per_model_predictions=per_model_predictions,
                        image_names=image_names,
                        fusion=args.fusion,
                        fusion_iou=args.fusion_iou,
                        max_det=max_det,
                    )
                    tmp_csv = SCRIPT_DIR / "_tmp_search_submission.csv"
                    write_submission(merged, tmp_csv)
                    predictions = load_predictions(tmp_csv)
                    map50, _, _ = score_predictions(predictions, ground_truth, num_classes=args.num_classes)
                    rows.append(
                        {
                            "map50": map50,
                            "conf": conf,
                            "iou": iou,
                            "imgsz": imgsz,
                            "max_det": max_det,
                            "tta": args.tta,
                            "fusion": args.fusion,
                            "fusion_iou": args.fusion_iou,
                        }
                    )

    rows.sort(key=lambda row: float(row["map50"]), reverse=True)
    save_csv = Path(args.save_csv).resolve()
    with save_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["map50", "conf", "iou", "imgsz", "max_det", "tta", "fusion", "fusion_iou"],
        )
        writer.writeheader()
        writer.writerows(rows)

    print(f"Saved search results to: {save_csv}")
    for row in rows[: args.topk]:
        print(row)


if __name__ == "__main__":
    main()
