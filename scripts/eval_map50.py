from __future__ import annotations

import argparse
import csv
from collections import Counter, defaultdict
from pathlib import Path


def xywh_to_xyxy(box: tuple[float, float, float, float]) -> tuple[float, float, float, float]:
    x_center, y_center, width, height = box
    return (
        x_center - width / 2.0,
        y_center - height / 2.0,
        x_center + width / 2.0,
        y_center + height / 2.0,
    )


def iou(box1: tuple[float, float, float, float], box2: tuple[float, float, float, float]) -> float:
    x11, y11, x12, y12 = box1
    x21, y21, x22, y22 = box2
    inter_x1 = max(x11, x21)
    inter_y1 = max(y11, y21)
    inter_x2 = min(x12, x22)
    inter_y2 = min(y12, y22)
    inter_w = max(0.0, inter_x2 - inter_x1)
    inter_h = max(0.0, inter_y2 - inter_y1)
    inter = inter_w * inter_h
    if inter <= 0:
        return 0.0
    area1 = max(0.0, x12 - x11) * max(0.0, y12 - y11)
    area2 = max(0.0, x22 - x21) * max(0.0, y22 - y21)
    union = area1 + area2 - inter
    return inter / union if union > 0 else 0.0


def load_ground_truth(image_dir: Path, label_dir: Path) -> dict[str, list[dict[str, object]]]:
    ground_truth: dict[str, list[dict[str, object]]] = {}
    for image_path in sorted(p for p in image_dir.iterdir() if p.is_file()):
        label_path = label_dir / f"{image_path.stem}.txt"
        records: list[dict[str, object]] = []
        if label_path.exists():
            text = label_path.read_text(encoding="utf-8").strip()
            if text:
                for line in text.splitlines():
                    cls, x_center, y_center, width, height = line.split()
                    records.append(
                        {
                            "class_id": int(cls),
                            "box": xywh_to_xyxy((float(x_center), float(y_center), float(width), float(height))),
                        }
                    )
        ground_truth[image_path.name] = records
    return ground_truth


def load_predictions(csv_path: Path) -> list[dict[str, object]]:
    predictions: list[dict[str, object]] = []
    with csv_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            predictions.append(
                {
                    "image_id": row["image_id"],
                    "class_id": int(row["class_id"]),
                    "confidence": float(row["confidence"]),
                    "box": xywh_to_xyxy(
                        (
                            float(row["x_center"]),
                            float(row["y_center"]),
                            float(row["width"]),
                            float(row["height"]),
                        )
                    ),
                }
            )
    return predictions


def ap_from_pr(recalls: list[float], precisions: list[float]) -> float:
    sampled = []
    for threshold in [i / 100 for i in range(101)]:
        precision_at_threshold = max((p for r, p in zip(recalls, precisions) if r >= threshold), default=0.0)
        sampled.append(precision_at_threshold)
    return sum(sampled) / len(sampled)


def score_predictions(
    predictions: list[dict[str, object]],
    ground_truth: dict[str, list[dict[str, object]]],
    num_classes: int,
) -> tuple[float, dict[int, float], dict[int, int]]:
    gt_by_class_and_image: dict[int, dict[str, list[tuple[float, float, float, float]]]] = defaultdict(lambda: defaultdict(list))
    gt_counts = Counter()
    for image_id, records in ground_truth.items():
        for record in records:
            class_id = int(record["class_id"])
            gt_by_class_and_image[class_id][image_id].append(record["box"])  # type: ignore[arg-type]
            gt_counts[class_id] += 1

    predictions_by_class: dict[int, list[dict[str, object]]] = defaultdict(list)
    for prediction in predictions:
        predictions_by_class[int(prediction["class_id"])].append(prediction)

    class_aps: dict[int, float] = {}
    for class_id in range(num_classes):
        gt_total = gt_counts[class_id]
        if gt_total == 0:
            continue

        preds = sorted(predictions_by_class[class_id], key=lambda item: float(item["confidence"]), reverse=True)
        matched: dict[str, list[bool]] = {
            image_id: [False] * len(boxes) for image_id, boxes in gt_by_class_and_image[class_id].items()
        }
        tps: list[int] = []
        fps: list[int] = []

        for pred in preds:
            image_id = str(pred["image_id"])
            pred_box = pred["box"]  # type: ignore[assignment]
            best_iou = 0.0
            best_idx = -1
            gt_boxes = gt_by_class_and_image[class_id].get(image_id, [])
            for idx, gt_box in enumerate(gt_boxes):
                if matched[image_id][idx]:
                    continue
                overlap = iou(pred_box, gt_box)
                if overlap > best_iou:
                    best_iou = overlap
                    best_idx = idx

            if best_iou >= 0.5 and best_idx >= 0:
                matched[image_id][best_idx] = True
                tps.append(1)
                fps.append(0)
            else:
                tps.append(0)
                fps.append(1)

        cum_tp = 0
        cum_fp = 0
        recalls: list[float] = []
        precisions: list[float] = []
        for tp, fp in zip(tps, fps):
            cum_tp += tp
            cum_fp += fp
            recalls.append(cum_tp / gt_total)
            precisions.append(cum_tp / max(1, cum_tp + cum_fp))
        class_aps[class_id] = ap_from_pr(recalls, precisions) if recalls else 0.0

    map50 = sum(class_aps.values()) / len(class_aps) if class_aps else 0.0
    return map50, class_aps, dict(gt_counts)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Score a submission.csv against YOLO labels with mAP@0.5.")
    parser.add_argument("--csv", required=True, help="Prediction CSV path")
    parser.add_argument("--image-dir", required=True, help="Image directory, e.g. val/images")
    parser.add_argument("--label-dir", required=True, help="Label directory, e.g. val/labels")
    parser.add_argument("--num-classes", type=int, default=15, help="Number of classes")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    csv_path = Path(args.csv).resolve()
    image_dir = Path(args.image_dir).resolve()
    label_dir = Path(args.label_dir).resolve()

    ground_truth = load_ground_truth(image_dir, label_dir)
    predictions = load_predictions(csv_path)
    map50, class_aps, gt_counts = score_predictions(predictions, ground_truth, num_classes=args.num_classes)

    print(f"mAP@0.5={map50:.6f}")
    for class_id in sorted(class_aps):
        print(f"class={class_id} ap50={class_aps[class_id]:.6f} gt={gt_counts.get(class_id, 0)}")


if __name__ == "__main__":
    main()
