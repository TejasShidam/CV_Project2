from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class Track:
    track_id: int
    centroid: tuple[float, float]
    missed: int = 0


class CentroidTracker:
    def __init__(self, max_distance: float = 60.0, max_missed: int = 10) -> None:
        self.max_distance = max_distance
        self.max_missed = max_missed
        self.next_id = 1
        self.tracks: dict[int, Track] = {}

    def _distance(self, a: tuple[float, float], b: tuple[float, float]) -> float:
        va = np.array(a, dtype=np.float32)
        vb = np.array(b, dtype=np.float32)
        return float(np.linalg.norm(va - vb))

    def update(self, detections: list[tuple[int, int, int, int]]) -> dict[int, tuple[int, int, int, int]]:
        if not detections:
            stale_ids = []
            for track_id, track in self.tracks.items():
                track.missed += 1
                if track.missed > self.max_missed:
                    stale_ids.append(track_id)
            for track_id in stale_ids:
                del self.tracks[track_id]
            return {}

        centroids = [((x1 + x2) / 2, (y1 + y2) / 2) for x1, y1, x2, y2 in detections]
        assignments: dict[int, int] = {}

        for det_idx, centroid in enumerate(centroids):
            best_track_id = None
            best_dist = float("inf")

            for track_id, track in self.tracks.items():
                dist = self._distance(track.centroid, centroid)
                if dist < best_dist and dist <= self.max_distance and track_id not in assignments:
                    best_dist = dist
                    best_track_id = track_id

            if best_track_id is None:
                track_id = self.next_id
                self.next_id += 1
                self.tracks[track_id] = Track(track_id=track_id, centroid=centroid, missed=0)
                assignments[track_id] = det_idx
            else:
                self.tracks[best_track_id].centroid = centroid
                self.tracks[best_track_id].missed = 0
                assignments[best_track_id] = det_idx

        matched = set(assignments.keys())
        stale_ids = []
        for track_id, track in self.tracks.items():
            if track_id not in matched:
                track.missed += 1
                if track.missed > self.max_missed:
                    stale_ids.append(track_id)
        for track_id in stale_ids:
            del self.tracks[track_id]

        return {track_id: detections[det_idx] for track_id, det_idx in assignments.items()}
