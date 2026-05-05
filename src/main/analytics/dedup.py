from __future__ import annotations

import hashlib
from datetime import datetime, timezone


# ---------------------------------------------------------------------------
# Hashing-based exact dedup
# ---------------------------------------------------------------------------

def plate_hash(plate_text: str) -> str:
    """Return a short SHA-256 hex digest for a normalised plate string."""
    return hashlib.sha256(plate_text.upper().encode()).hexdigest()[:16]


def find_exact_duplicates(plates: list[str]) -> dict[str, list[int]]:
    """Group identical plates by their indices.

    Returns:
        Mapping from plate text → list of indices where it appears.
        Only plates appearing more than once are included.
    """
    seen: dict[str, list[int]] = {}
    for idx, plate in enumerate(plates):
        key = plate.upper()
        seen.setdefault(key, []).append(idx)
    return {k: v for k, v in seen.items() if len(v) > 1}


# ---------------------------------------------------------------------------
# Levenshtein-based fuzzy dedup
# ---------------------------------------------------------------------------

def _levenshtein(a: str, b: str) -> int:
    if a == b:
        return 0
    if len(a) == 0:
        return len(b)
    if len(b) == 0:
        return len(a)

    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, start=1):
        cur = [i]
        for j, cb in enumerate(b, start=1):
            insert_cost = cur[j - 1] + 1
            delete_cost = prev[j] + 1
            replace_cost = prev[j - 1] + (0 if ca == cb else 1)
            cur.append(min(insert_cost, delete_cost, replace_cost))
        prev = cur
    return prev[-1]


def find_duplicates(plates: list[str], max_distance: int = 1) -> list[tuple[str, str, int]]:
    """Find plates that are within *max_distance* edit steps of each other.

    This catches OCR near-misses (e.g. ``MH12AB1234`` vs ``MH12AB1235``).
    For exact duplicates use :func:`find_exact_duplicates`.

    Returns:
        List of ``(plate_a, plate_b, levenshtein_distance)`` tuples.
    """
    duplicates: list[tuple[str, str, int]] = []
    for i in range(len(plates)):
        for j in range(i + 1, len(plates)):
            d = _levenshtein(plates[i], plates[j])
            if d <= max_distance:
                duplicates.append((plates[i], plates[j], d))
    return duplicates


# ---------------------------------------------------------------------------
# Timestamp-aware plate log
# ---------------------------------------------------------------------------

def log_detection(
    plate_text: str,
    zone: str = "Unknown",
    records: list[dict] | None = None,
) -> dict:
    """Create a timestamped detection record and optionally append to a list.

    Args:
        plate_text: Recognised plate string.
        zone:       Parking zone name (e.g. ``"A"``, ``"B"``).
        records:    Optional mutable list to append the new record to.

    Returns:
        The new record dict with keys ``plate``, ``zone``, ``timestamp``, ``hash``.
    """
    record = {
        "plate": plate_text.upper(),
        "zone": zone,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "hash": plate_hash(plate_text),
    }
    if records is not None:
        records.append(record)
    return record
