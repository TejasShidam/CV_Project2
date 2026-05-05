from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re
import xml.etree.ElementTree as ET


PLATE_RE = re.compile(r"^[A-Z0-9]{4,12}$")
FILENAME_PLATE_RE = re.compile(r"\b([A-Z]{2}\s*\d{1,2}\s*[A-Z]{1,3}\s*\d{3,4})\b")
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
GENERIC_LABELS = {
    "NUMBER",
    "NUMBERPLATE",
    "PLATE",
    "LICENSE",
    "LICENSEPLATE",
    "REGISTRATION",
    "VEHICLE",
}


@dataclass
class AnnotationRecord:
    xml_path: Path
    image_name: str
    image_path: Path
    bbox: tuple[int, int, int, int]
    plate_text: str
    object_index: int


def _normalize_plate_candidate(value: str) -> str:
    return re.sub(r"[^A-Z0-9]", "", value.upper())


def _is_valid_plate(value: str) -> bool:
    if not PLATE_RE.match(value):
        return False
    if value in GENERIC_LABELS:
        return False
    return any(ch.isalpha() for ch in value) and any(ch.isdigit() for ch in value)


def extract_plate_text_from_filename(file_stem: str) -> str:
    normalized = _normalize_plate_candidate(file_stem)
    if _is_valid_plate(normalized):
        return normalized

    separated = re.sub(r"[_\-]+", " ", file_stem.upper())
    for candidate in FILENAME_PLATE_RE.findall(separated):
        normalized = _normalize_plate_candidate(candidate)
        if _is_valid_plate(normalized):
            return normalized

    return ""


def _extract_bbox(node: ET.Element) -> tuple[int, int, int, int]:
    bnd = node.find("bndbox")
    if bnd is None:
        return (0, 0, 0, 0)

    def _int_or_zero(tag: str) -> int:
        node = bnd.find(tag)
        if node is None or node.text is None:
            return 0
        return int(float(node.text))

    xmin = _int_or_zero("xmin")
    ymin = _int_or_zero("ymin")
    xmax = _int_or_zero("xmax")
    ymax = _int_or_zero("ymax")

    return (xmin, ymin, xmax, ymax)


def _iter_plate_text_candidates(node: ET.Element) -> list[str]:
    candidates: list[str] = []

    for attribute in node.findall(".//attributes/attribute"):
        value_text = attribute.findtext("value")
        if value_text:
            candidates.append(value_text)

    for tag in ["plate", "plate_text", "number", "registration", "license", "text", "name"]:
        node_text = node.findtext(tag)
        if node_text:
            candidates.append(node_text)

    return candidates


READER = None


def get_image_bbox(image_path: Path) -> tuple[int, int, int, int]:
    import cv2

    image = cv2.imread(str(image_path))
    if image is None:
        return (0, 0, 0, 0)

    height, width = image.shape[:2]
    return (0, 0, width, height)


def _ocr_fallback(image_path: Path, bbox: tuple[int, int, int, int]) -> str:
    global READER
    if READER is None:
        try:
            import easyocr
            import logging

            logging.getLogger("easyocr").setLevel(logging.ERROR)
            # Suppress verbose downloads
            READER = easyocr.Reader(["en"], gpu=False, verbose=False)
        except Exception:
            return ""
        
    try:
        import cv2
        img = cv2.imread(str(image_path))
        if img is None:
            return ""
        xmin, ymin, xmax, ymax = bbox
        
        h, w = img.shape[:2]
        pad = 5
        if xmax <= xmin or ymax <= ymin:
            xmin, ymin, xmax, ymax = (0, 0, w, h)
        x1 = max(0, xmin - pad)
        y1 = max(0, ymin - pad)
        x2 = min(w, xmax + pad)
        y2 = min(h, ymax + pad)
        
        crop = img[y1:y2, x1:x2]
        if crop.size == 0:
            return ""
            
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        results = READER.readtext(gray, detail=0)
        
        if results:
            text = "".join(results)
            normalized = _normalize_plate_candidate(text)
            if _is_valid_plate(normalized):
                return normalized
    except Exception:
        return ""
        
    return ""


def extract_plate_text_from_image(image_path: Path) -> str:
    filename_text = extract_plate_text_from_filename(image_path.stem)
    if filename_text:
        return filename_text

    bbox = get_image_bbox(image_path)
    return _ocr_fallback(image_path, bbox)


def _extract_plate_text(node: ET.Element, image_path: Path, bbox: tuple[int, int, int, int]) -> str:
    candidates = _iter_plate_text_candidates(node)

    for candidate in candidates:
        normalized = _normalize_plate_candidate(candidate)
        if _is_valid_plate(normalized):
            return normalized

    return _ocr_fallback(image_path, bbox)


def parse_annotation_xml(xml_path: Path, image_dir: Path) -> list[AnnotationRecord]:
    tree = ET.parse(xml_path)
    root = tree.getroot()

    filename_node = root.find("filename")
    if filename_node is not None and filename_node.text:
        image_name = filename_node.text.strip()
    else:
        image_name = xml_path.stem + ".jpg"

    image_path = image_dir / image_name
    object_nodes = root.findall(".//object")
    if not object_nodes:
        object_nodes = [root]

    records: list[AnnotationRecord] = []
    for index, obj in enumerate(object_nodes):
        bbox = _extract_bbox(obj)
        plate_text = _extract_plate_text(obj, image_path, bbox)
        records.append(
            AnnotationRecord(
                xml_path=xml_path,
                image_name=image_name,
                image_path=image_path,
                bbox=bbox,
                plate_text=plate_text,
                object_index=index,
            )
        )

    return records


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp"}

def get_image_bbox(image_path: Path) -> tuple[int, int, int, int]:
    try:
        import cv2
        img = cv2.imread(str(image_path))
        if img is not None:
            h, w = img.shape[:2]
            return (0, 0, w, h)
    except Exception:
        pass
    return (0, 0, 0, 0)

def extract_plate_text_from_image(image_path: Path) -> str:
    stem = image_path.stem
    normalized = _normalize_plate_candidate(stem)
    if _is_valid_plate(normalized):
        return normalized
    
    bbox = get_image_bbox(image_path)
    if bbox[2] > 0 and bbox[3] > 0:
        return _ocr_fallback(image_path, bbox)
    return ""
