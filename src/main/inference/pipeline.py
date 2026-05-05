from __future__ import annotations

from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
import torch

from anpr.analytics.occupancy import update_occupancy
from anpr.cv.plate_localization import localize_plate_region_with_bbox
from anpr.data.transforms import build_eval_transforms
from anpr.inference.tracker import CentroidTracker
from anpr.models.anpr_model import ANPRModel
from anpr.tokenizer import PlateTokenizer


class ANPRInferenceEngine:
    def __init__(
        self,
        model_path: Path,
        charset: str,
        image_size: int,
        max_label_length: int,
        device: str = "cpu",
    ) -> None:
        self.device = torch.device(device)
        self.tokenizer = PlateTokenizer(charset)
        self.model = ANPRModel(
            vocab_size=self.tokenizer.vocab_size,
            max_label_length=max_label_length,
            sos_id=self.tokenizer.sos_id,
            eos_id=self.tokenizer.eos_id,
        ).to(self.device)

        checkpoint = torch.load(model_path, map_location=self.device)
        self.model.load_state_dict(checkpoint["model_state"])
        self.model.eval()

        self.transforms = build_eval_transforms(image_size)
        self.tracker = CentroidTracker()
        self.zone_counts: dict[str, int] = {}

    def predict_plate_from_image(self, image_bgr: np.ndarray) -> dict[str, object]:
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        roi, bbox = localize_plate_region_with_bbox(image_rgb)

        transformed = self.transforms(image=roi)
        tensor = (
            torch.tensor(transformed["image"], dtype=torch.float32)
            .permute(2, 0, 1)
            .unsqueeze(0)
            .to(self.device)
        )

        with torch.no_grad():
            tokens = self.model.greedy_decode(tensor)

        plate_text = self.tokenizer.decode(tokens[0])
        return {
            "plate_text": plate_text,
            "bbox": bbox,
            "timestamp": datetime.utcnow().isoformat(),
        }

    def process_frame(self, frame_bgr: np.ndarray) -> dict[str, object]:
        pred = self.predict_plate_from_image(frame_bgr)
        bbox = pred["bbox"]
        tracked = self.tracker.update([bbox])

        if tracked:
            track_id = next(iter(tracked))
        else:
            track_id = -1

        zone = self._infer_zone_from_bbox(frame_bgr.shape[1], bbox)
        update_occupancy(self.zone_counts, zone, delta=1)

        pred["track_id"] = track_id
        pred["zone"] = zone
        pred["occupancy"] = dict(self.zone_counts)
        return pred

    def _infer_zone_from_bbox(self, image_width: int, bbox: tuple[int, int, int, int]) -> str:
        x_center = (bbox[0] + bbox[2]) / 2
        if x_center < image_width / 3:
            return "Zone-A"
        if x_center < (2 * image_width) / 3:
            return "Zone-B"
        return "Zone-C"
