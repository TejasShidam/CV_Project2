from __future__ import annotations

from collections import defaultdict
from datetime import datetime, timezone
from typing import Optional


# ---------------------------------------------------------------------------
# Simple mutable zone counter (used by the API engine)
# ---------------------------------------------------------------------------

def update_occupancy(zone_counts: dict[str, int], zone: str, delta: int = 1) -> None:
    """Increment (+1) or decrement (-1) a zone counter, floor at 0."""
    zone_counts[zone] = max(0, zone_counts.get(zone, 0) + delta)


def compute_occupancy_summary(records: list[dict]) -> dict[str, int]:
    """Count total detections per zone from a list of detection records."""
    summary: dict[str, int] = defaultdict(int)
    for record in records:
        zone = record.get("zone", "Unknown")
        summary[zone] += 1
    return dict(summary)


# ---------------------------------------------------------------------------
# Timestamped session tracker
# ---------------------------------------------------------------------------

class ParkingSessionTracker:
    """Tracks vehicle entry/exit events per zone with timestamps.

    Usage::

        tracker = ParkingSessionTracker(capacity={"A": 20, "B": 15})
        tracker.enter("MH12AB1234", zone="A")
        tracker.exit("MH12AB1234", zone="A")
        print(tracker.zone_summary())
    """

    def __init__(self, capacity: Optional[dict[str, int]] = None) -> None:
        # zone → {plate → entry_timestamp}
        self._active: dict[str, dict[str, str]] = defaultdict(dict)
        # completed sessions: list of {plate, zone, entry, exit, duration_s}
        self.sessions: list[dict] = []
        self.capacity = capacity or {}

    def enter(self, plate: str, zone: str = "Unknown") -> dict:
        """Record a vehicle entering a zone.

        Returns the entry record dict.
        """
        ts = datetime.now(timezone.utc).isoformat()
        self._active[zone][plate.upper()] = ts
        return {"plate": plate.upper(), "zone": zone, "entry": ts, "event": "enter"}

    def exit(self, plate: str, zone: str = "Unknown") -> dict:
        """Record a vehicle exiting a zone.

        Returns the session record dict (includes duration in seconds).
        """
        plate = plate.upper()
        ts_exit = datetime.now(timezone.utc).isoformat()
        entry_ts = self._active[zone].pop(plate, None)

        session: dict = {"plate": plate, "zone": zone, "exit": ts_exit, "event": "exit"}
        if entry_ts:
            session["entry"] = entry_ts
            try:
                dt_entry = datetime.fromisoformat(entry_ts)
                dt_exit  = datetime.fromisoformat(ts_exit)
                session["duration_s"] = (dt_exit - dt_entry).total_seconds()
            except Exception:
                session["duration_s"] = None
            self.sessions.append(session)

        return session

    def zone_summary(self) -> dict[str, dict]:
        """Return per-zone occupancy and capacity stats.

        Example output::

            {
                "A": {"current": 3, "capacity": 20, "utilisation": 0.15},
                "B": {"current": 7, "capacity": 15, "utilisation": 0.47},
            }
        """
        summary: dict[str, dict] = {}
        all_zones = set(self._active) | set(self.capacity)
        for zone in all_zones:
            current = len(self._active.get(zone, {}))
            cap = self.capacity.get(zone, 0)
            summary[zone] = {
                "current": current,
                "capacity": cap,
                "utilisation": round(current / cap, 4) if cap else None,
                "available": max(0, cap - current) if cap else None,
            }
        return summary

    @property
    def active_plates(self) -> dict[str, list[str]]:
        """Return dict of zone → list of currently-parked plates."""
        return {zone: list(plates) for zone, plates in self._active.items()}

    def total_current(self) -> int:
        """Total vehicles currently parked across all zones."""
        return sum(len(plates) for plates in self._active.values())
