from __future__ import annotations

import argparse
import random
from collections import Counter
from dataclasses import dataclass
import os
from pathlib import Path

import yaml


@dataclass(frozen=True)
class ImageRecord:
    image_path: Path
    label_path: Path
    classes: frozenset[int]
    num_boxes: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a class-aware holdout split from YOLO data.")
    parser.add_argument("--root", default=".", help="Dataset root that contains train/images and train/labels")
    parser.add_argument("--fraction", type=float, default=0.15, help="Holdout fraction from train split")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--output-dir", default="splits/holdout_seed42", help="Output directory")
    return parser.parse_args()


def load_records(root: Path) -> list[ImageRecord]:
    image_dir = root / "train" / "images"
    label_dir = root / "train" / "labels"
    records: list[ImageRecord] = []
    for image_path in sorted(p for p in image_dir.iterdir() if p.is_file()):
        label_path = label_dir / f"{image_path.stem}.txt"
        classes: set[int] = set()
        num_boxes = 0
        if label_path.exists():
            text = label_path.read_text(encoding="utf-8").strip()
            if text:
                for line in text.splitlines():
                    cls = int(line.split()[0])
                    classes.add(cls)
                    num_boxes += 1
        records.append(
            ImageRecord(
                image_path=Path(os.path.abspath(image_path)),
                label_path=Path(os.path.abspath(label_path)),
                classes=frozenset(classes),
                num_boxes=num_boxes,
            )
        )
    return records


def greedy_holdout(records: list[ImageRecord], fraction: float, seed: int) -> tuple[list[ImageRecord], list[ImageRecord]]:
    rng = random.Random(seed)
    target_size = max(1, round(len(records) * fraction))
    class_image_counts = Counter()
    for record in records:
        for cls in record.classes:
            class_image_counts[cls] += 1

    class_targets = {
        cls: max(1, round(count * fraction))
        for cls, count in class_image_counts.items()
        if count > 0
    }
    chosen: list[ImageRecord] = []
    chosen_set: set[Path] = set()
    holdout_counts = Counter()

    candidates = records[:]
    rng.shuffle(candidates)

    while len(chosen) < target_size:
        deficits = {cls: class_targets[cls] - holdout_counts[cls] for cls in class_targets}
        unmet = {cls: value for cls, value in deficits.items() if value > 0}
        if not unmet:
            break

        best_record = None
        best_score = -1.0
        for record in candidates:
            if record.image_path in chosen_set or not record.classes:
                continue
            score = sum(unmet.get(cls, 0) / class_image_counts[cls] for cls in record.classes)
            score += 0.01 * len(record.classes)
            score += 0.001 * record.num_boxes
            if score > best_score:
                best_score = score
                best_record = record

        if best_record is None:
            break

        chosen.append(best_record)
        chosen_set.add(best_record.image_path)
        for cls in best_record.classes:
            holdout_counts[cls] += 1

    remaining = [record for record in candidates if record.image_path not in chosen_set]
    remaining.sort(key=lambda record: (len(record.classes), record.num_boxes), reverse=True)
    rng.shuffle(remaining)
    for record in remaining:
        if len(chosen) >= target_size:
            break
        chosen.append(record)
        chosen_set.add(record.image_path)

    holdout = sorted(chosen, key=lambda record: record.image_path.name)
    train_split = sorted(
        [record for record in records if record.image_path not in chosen_set],
        key=lambda record: record.image_path.name,
    )
    return train_split, holdout


def write_split(root: Path, output_dir: Path, train_records: list[ImageRecord], holdout_records: list[ImageRecord]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    train_txt = output_dir / "train.txt"
    holdout_txt = output_dir / "holdout.txt"
    train_txt.write_text("\n".join(str(record.image_path) for record in train_records) + "\n", encoding="utf-8")
    holdout_txt.write_text("\n".join(str(record.image_path) for record in holdout_records) + "\n", encoding="utf-8")

    data = yaml.safe_load((root / "data.yaml").read_text(encoding="utf-8"))
    data["train"] = str(train_txt.resolve())
    data["val"] = str(holdout_txt.resolve())
    (output_dir / "holdout.yaml").write_text(yaml.safe_dump(data, sort_keys=False, allow_unicode=True), encoding="utf-8")

    class_counter = Counter()
    for record in holdout_records:
        for cls in record.classes:
            class_counter[cls] += 1
    summary_lines = [
        f"train_images={len(train_records)}",
        f"holdout_images={len(holdout_records)}",
        f"holdout_class_image_counts={dict(sorted(class_counter.items()))}",
    ]
    (output_dir / "summary.txt").write_text("\n".join(summary_lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    root = Path(os.path.abspath(args.root))
    records = load_records(root)
    train_records, holdout_records = greedy_holdout(records, fraction=args.fraction, seed=args.seed)
    output_dir = Path(os.path.abspath(args.output_dir))
    write_split(root, output_dir, train_records, holdout_records)
    print(f"Created holdout split at: {output_dir}")
    print(f"Train images: {len(train_records)}")
    print(f"Holdout images: {len(holdout_records)}")


if __name__ == "__main__":
    main()
