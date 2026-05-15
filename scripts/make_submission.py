from __future__ import annotations

import argparse
import csv
import gc
from collections import defaultdict
from pathlib import Path

import torch
from ultralytics import YOLO


Prediction = tuple[float, float, float, float, float, int]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate submission.csv with optional TTA and model fusion.")
    parser.add_argument("--models", nargs="+", required=True, help="One or more model checkpoint paths")
    parser.add_argument("--source", default="test/images", help="Image directory to run inference on")
    parser.add_argument("--output", default="submission.csv", help="Output CSV file")
    parser.add_argument("--conf", type=float, default=0.001, help="Confidence threshold")
    parser.add_argument("--iou", type=float, default=0.6, help="NMS IoU threshold inside each model")
    parser.add_argument("--imgsz", type=int, default=640, help="Inference image size")
    parser.add_argument("--device", default="0", help="CUDA device id or cpu")
    parser.add_argument("--batch", type=int, default=16, help="Inference batch size")
    parser.add_argument("--half", action="store_true", help="Use FP16 inference when supported")
    parser.add_argument("--max-det", type=int, default=50, help="Max detections per image")
    parser.add_argument("--tta", action="store_true", help="Enable Ultralytics test-time augmentation")
    parser.add_argument("--fusion", choices=["none", "wbf"], default="wbf", help="How to combine multiple models")
    parser.add_argument("--fusion-iou", type=float, default=0.55, help="IoU threshold used for box fusion")
    return parser.parse_args()


def compute_iou(box1: tuple[float, float, float, float], box2: tuple[float, float, float, float]) -> float:
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


def clip01(value: float) -> float:
    return max(0.0, min(1.0, value))


def xyxy_to_xywhn(box: tuple[float, float, float, float]) -> tuple[float, float, float, float]:
    x1, y1, x2, y2 = box
    width = clip01(x2 - x1)
    height = clip01(y2 - y1)
    x_center = clip01((x1 + x2) / 2.0)
    y_center = clip01((y1 + y2) / 2.0)
    return x_center, y_center, width, height


def collect_model_predictions(
    model_path: str,
    image_paths: list[Path],
    conf: float,
    iou: float,
    imgsz: int,
    device: str,
    batch: int,
    half: bool,
    max_det: int,
    tta: bool,
) -> dict[str, list[Prediction]]:
    model = YOLO(model_path)
    collected: dict[str, list[Prediction]] = {}
    results = model.predict(
        source=[str(path) for path in image_paths],
        conf=conf,
        iou=iou,
        imgsz=imgsz,
        device=device,
        batch=batch,
        half=half,
        max_det=max_det,
        augment=tta,
        save=False,
        verbose=False,
    )
    for image_path, result in zip(image_paths, results):
        image_name = image_path.name
        preds: list[Prediction] = []
        if result.boxes is not None:
            xyxyn = result.boxes.xyxyn.cpu().tolist()
            confs = result.boxes.conf.cpu().tolist()
            classes = result.boxes.cls.cpu().tolist()
            for box, score, cls in zip(xyxyn, confs, classes):
                preds.append((box[0], box[1], box[2], box[3], float(score), int(cls)))
        collected[image_name] = preds
    del results
    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return collected


def fuse_class_predictions(predictions: list[Prediction], iou_thr: float, model_count: int) -> list[Prediction]:
    clusters: list[list[Prediction]] = []
    for pred in sorted(predictions, key=lambda item: item[4], reverse=True):
        box = pred[:4]
        placed = False
        for cluster in clusters:
            ref_box = cluster[0][:4]
            if compute_iou(box, ref_box) >= iou_thr:
                cluster.append(pred)
                placed = True
                break
        if not placed:
            clusters.append([pred])

    fused: list[Prediction] = []
    for cluster in clusters:
        weights = [item[4] for item in cluster]
        weight_sum = sum(weights)
        x1 = sum(item[0] * w for item, w in zip(cluster, weights)) / weight_sum
        y1 = sum(item[1] * w for item, w in zip(cluster, weights)) / weight_sum
        x2 = sum(item[2] * w for item, w in zip(cluster, weights)) / weight_sum
        y2 = sum(item[3] * w for item, w in zip(cluster, weights)) / weight_sum
        confidence = sum(item[4] for item in cluster) / max(1, len(cluster))
        confidence *= len(cluster) / max(1, model_count)
        cls = cluster[0][5]
        fused.append((clip01(x1), clip01(y1), clip01(x2), clip01(y2), clip01(confidence), cls))
    return fused


def merge_predictions(
    per_model_predictions: list[dict[str, list[Prediction]]],
    image_names: list[str],
    fusion: str,
    fusion_iou: float,
    max_det: int,
) -> dict[str, list[Prediction]]:
    merged: dict[str, list[Prediction]] = {}
    model_count = len(per_model_predictions)
    for image_name in image_names:
        combined: list[Prediction] = []
        for model_predictions in per_model_predictions:
            combined.extend(model_predictions.get(image_name, []))
        if fusion == "none" or model_count == 1:
            merged[image_name] = sorted(combined, key=lambda item: item[4], reverse=True)[:max_det]
            continue

        grouped: dict[int, list[Prediction]] = defaultdict(list)
        for pred in combined:
            grouped[pred[5]].append(pred)

        fused_preds: list[Prediction] = []
        for class_predictions in grouped.values():
            fused_preds.extend(fuse_class_predictions(class_predictions, iou_thr=fusion_iou, model_count=model_count))
        merged[image_name] = sorted(fused_preds, key=lambda item: item[4], reverse=True)[:max_det]
    return merged


def write_submission(predictions: dict[str, list[Prediction]], output_path: Path) -> None:
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["image_id", "class_id", "x_center", "y_center", "width", "height", "confidence"],
        )
        writer.writeheader()
        for image_name in sorted(predictions):
            for x1, y1, x2, y2, confidence, class_id in predictions[image_name]:
                x_center, y_center, width, height = xyxy_to_xywhn((x1, y1, x2, y2))
                writer.writerow(
                    {
                        "image_id": image_name,
                        "class_id": class_id,
                        "x_center": x_center,
                        "y_center": y_center,
                        "width": width,
                        "height": height,
                        "confidence": confidence,
                    }
                )


def main() -> None:
    args = parse_args()
    image_paths = sorted(path for path in Path(args.source).resolve().iterdir() if path.is_file())
    image_names = [path.name for path in image_paths]
    per_model_predictions = [
        collect_model_predictions(
            model_path=model_path,
            image_paths=image_paths,
            conf=args.conf,
            iou=args.iou,
            imgsz=args.imgsz,
            device=args.device,
            batch=args.batch,
            half=args.half,
            max_det=args.max_det,
            tta=args.tta,
        )
        for model_path in args.models
    ]
    predictions = merge_predictions(
        per_model_predictions=per_model_predictions,
        image_names=image_names,
        fusion=args.fusion,
        fusion_iou=args.fusion_iou,
        max_det=args.max_det,
    )
    output_path = Path(args.output).resolve()
    write_submission(predictions, output_path)
    print(f"Saved submission to: {output_path}")


if __name__ == "__main__":
    main()
