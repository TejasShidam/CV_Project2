from anpr.analytics.dedup import find_duplicates


def test_find_duplicates() -> None:
    records = ["KA01AB1234", "KA01AB1235", "MH12XY0001"]
    duplicates = find_duplicates(records, max_distance=1)
    assert any(d[0] == "KA01AB1234" and d[1] == "KA01AB1235" for d in duplicates)
