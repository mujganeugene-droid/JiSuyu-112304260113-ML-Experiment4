from __future__ import annotations

import argparse
import os
from pathlib import Path

import yaml


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create YOLO train/val txt files for train+val final training.")
    parser.add_argument("--root", default=".", help="Dataset root")
    parser.add_argument("--output-dir", default="splits/trainval_final", help="Output directory")
    parser.add_argument("--val-source", choices=["train", "val"], default="train", help="Which split to use for monitoring")
    return parser.parse_args()


def list_images(image_dir: Path) -> list[Path]:
    return sorted(path for path in image_dir.iterdir() if path.is_file())


def write_list(path: Path, items: list[Path]) -> None:
    path.write_text("\n".join(os.path.abspath(str(item)) for item in items) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    root = Path(os.path.abspath(args.root))
    output_dir = Path(os.path.abspath(args.output_dir))
    output_dir.mkdir(parents=True, exist_ok=True)

    train_images = list_images(root / "train" / "images")
    val_images = list_images(root / "val" / "images")
    all_images = train_images + val_images

    train_txt = output_dir / "trainval.txt"
    monitor_txt = output_dir / "monitor.txt"
    write_list(train_txt, all_images)
    write_list(monitor_txt, train_images if args.val_source == "train" else val_images)

    data = yaml.safe_load((root / "data.yaml").read_text(encoding="utf-8"))
    data["train"] = str(train_txt)
    data["val"] = str(monitor_txt)
    (output_dir / "trainval.yaml").write_text(yaml.safe_dump(data, sort_keys=False, allow_unicode=True), encoding="utf-8")

    print(f"trainval_images={len(all_images)}")
    print(f"monitor_images={len(train_images if args.val_source == 'train' else val_images)}")
    print(f"yaml={output_dir / 'trainval.yaml'}")


if __name__ == "__main__":
    main()
