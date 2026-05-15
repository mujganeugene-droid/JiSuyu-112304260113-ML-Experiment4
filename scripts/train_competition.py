from __future__ import annotations

import argparse
from pathlib import Path

from ultralytics import YOLO


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a stronger YOLO baseline for the traffic sign competition.")
    parser.add_argument("--model", required=True, help="Pretrained checkpoint path, e.g. yolov8m.pt")
    parser.add_argument("--data", default="data.yaml", help="YOLO data yaml path")
    parser.add_argument("--epochs", type=int, default=150, help="Training epochs")
    parser.add_argument("--imgsz", type=int, default=640, help="Training image size")
    parser.add_argument("--batch", type=int, default=-1, help="Batch size, -1 lets Ultralytics auto-tune")
    parser.add_argument("--device", default="0", help="CUDA device id or cpu")
    parser.add_argument("--workers", type=int, default=8, help="Dataloader workers")
    parser.add_argument("--project", default="runs/traffic_sign", help="Output project directory")
    parser.add_argument("--name", default="exp", help="Experiment name")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--patience", type=int, default=40, help="Early stopping patience")
    parser.add_argument("--close-mosaic", type=int, default=10, help="Disable mosaic in the last N epochs")
    parser.add_argument("--cache", default="ram", help="Ultralytics cache mode: ram, disk, or False")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    model = YOLO(args.model)
    results = model.train(
        data=str(Path(args.data).resolve()),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        workers=args.workers,
        project=args.project,
        name=args.name,
        seed=args.seed,
        patience=args.patience,
        cache=args.cache,
        pretrained=True,
        optimizer="auto",
        lr0=0.003,
        lrf=0.01,
        weight_decay=0.0005,
        warmup_epochs=3.0,
        cos_lr=True,
        amp=True,
        save=True,
        plots=True,
        val=True,
        rect=False,
        degrees=3.0,
        translate=0.08,
        scale=0.45,
        shear=1.0,
        perspective=0.0,
        flipud=0.0,
        fliplr=0.5,
        hsv_h=0.015,
        hsv_s=0.4,
        hsv_v=0.25,
        mosaic=0.8,
        mixup=0.05,
        copy_paste=0.0,
        erasing=0.15,
        close_mosaic=args.close_mosaic,
        max_det=50,
    )
    save_dir = Path(results.save_dir).resolve()
    print(f"Training finished. Save dir: {save_dir}")
    print(f"Best weights: {save_dir / 'weights' / 'best.pt'}")


if __name__ == "__main__":
    main()
