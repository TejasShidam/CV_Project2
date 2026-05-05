from __future__ import annotations

from pathlib import Path
from typing import Optional

import pandas as pd

from .parser import (
    IMAGE_EXTENSIONS,
    extract_plate_text_from_image,
    get_image_bbox,
    parse_annotation_xml,
)


def _verify_from_xml(annotation_dir: Path, image_dir: Path) -> pd.DataFrame:
    rows: list[dict[str, object]] = []

    for xml_path in sorted(annotation_dir.glob("*.xml")):
        try:
            records = parse_annotation_xml(xml_path, image_dir)
        except Exception as exc:  # noqa: BLE001
            rows.append(
                {
                    "xml_path": str(xml_path),
                    "object_index": -1,
                    "image_name": "",
                    "image_path": "",
                    "xmin": 0,
                    "ymin": 0,
                    "xmax": 0,
                    "ymax": 0,
                    "plate_text": "",
                    "status": "invalid",
                    "reason": f"parse_error:{exc}",
                }
            )
            continue

        if not records:
            rows.append(
                {
                    "xml_path": str(xml_path),
                    "object_index": -1,
                    "image_name": "",
                    "image_path": "",
                    "xmin": 0,
                    "ymin": 0,
                    "xmax": 0,
                    "ymax": 0,
                    "plate_text": "",
                    "status": "invalid",
                    "reason": "no_objects",
                }
            )
            continue

        for record in records:
            bbox = record.bbox
            plate_text = record.plate_text
            status = "ok"
            reason = ""

            if not record.image_path.exists():
                status = "invalid"
                reason = "missing_image"
            elif bbox[2] <= bbox[0] or bbox[3] <= bbox[1]:
                status = "invalid"
                reason = "invalid_bbox"
            elif not plate_text:
                status = "invalid"
                reason = "missing_plate_text"

            rows.append(
                {
                    "xml_path": str(xml_path),
                    "object_index": record.object_index,
                    "image_name": record.image_name,
                    "image_path": str(record.image_path),
                    "xmin": bbox[0],
                    "ymin": bbox[1],
                    "xmax": bbox[2],
                    "ymax": bbox[3],
                    "plate_text": plate_text,
                    "status": status,
                    "reason": reason,
                }
            )

    return pd.DataFrame(rows)


def _iter_dataset_image_dirs(data_root: Path, image_dir: Path) -> list[Path]:
    # Prioritize the explicitly provided image_dir
    candidates = [image_dir]
    
    # Only add other candidates if they differ from the primary image_dir
    # These are potential fallback locations
    defaults = [
        data_root / "in" / "in",
        data_root / "Indian Plates" / "positive",
        data_root / "Indian Number Plate" / "Indian Number Plate",
    ]
    
    for d in defaults:
        if d not in candidates:
            candidates.append(d)

    unique_dirs: list[Path] = []
    seen: set[str] = set()
    for candidate in candidates:
        try:
            resolved = candidate.resolve()
        except Exception:
            resolved = candidate

        key = str(resolved)
        if key in seen:
            continue
        seen.add(key)

        if resolved.exists() and resolved.is_dir():
            unique_dirs.append(resolved)
    
    # If we have a specific image_dir that works, we probably don't want to 
    # scan everything else if it contains thousands of images.
    # However, for this project, let's just return the first one that exists 
    # OR all of them if the user expects a merged dataset.
    # Given the previous context, the user might want a merged dataset.
    # But since it's too slow, let's just return the primary one if it exists.
    if unique_dirs and image_dir.exists():
        return [image_dir.resolve()]

    return unique_dirs


def _verify_from_folders(data_root: Path, image_dir: Path) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    image_dirs = _iter_dataset_image_dirs(data_root, image_dir)
    seen_image_names: set[str] = set()

    for folder in image_dirs:
        for image_path in sorted(folder.iterdir()):
            if not image_path.is_file() or image_path.suffix.lower() not in IMAGE_EXTENSIONS:
                continue
            if image_path.name in seen_image_names:
                continue

            seen_image_names.add(image_path.name)

            bbox = get_image_bbox(image_path)
            plate_text = extract_plate_text_from_image(image_path)
            status = "ok"
            reason = ""

            if bbox[2] <= bbox[0] or bbox[3] <= bbox[1]:
                status = "invalid"
                reason = "unreadable_image"
            elif not plate_text:
                status = "invalid"
                reason = "missing_plate_text"

            rows.append(
                {
                    "xml_path": "",
                    "object_index": 0,
                    "image_name": image_path.name,
                    "image_path": str(image_path),
                    "xmin": bbox[0],
                    "ymin": bbox[1],
                    "xmax": bbox[2],
                    "ymax": bbox[3],
                    "plate_text": plate_text,
                    "status": status,
                    "reason": reason,
                }
            )

    return pd.DataFrame(rows)


def verify_annotations(annotation_dir: Path, image_dir: Path, data_root: Optional[Path] = None) -> pd.DataFrame:
    xml_files = sorted(annotation_dir.glob("*.xml")) if annotation_dir.exists() else []
    if xml_files:
        return _verify_from_xml(annotation_dir, image_dir)

    fallback_root = data_root if data_root is not None else image_dir.parent
    return _verify_from_folders(fallback_root, image_dir)
