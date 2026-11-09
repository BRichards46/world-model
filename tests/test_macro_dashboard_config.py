from __future__ import annotations

from world_model.config import default_config_path, load_yaml


def test_macro_dashboard_source_manifest_is_ordered_and_unique() -> None:
    config = load_yaml(default_config_path("macro_dashboard.yml"))
    rows = config["rows"]

    assert len(rows) == 14
    assert [row["order"] for row in rows] == list(range(1, 15))
    assert len({row["id"] for row in rows}) == len(rows)
    assert all(row["series_ids"] for row in rows)
    assert all(row["source_mode"] for row in rows)
    assert all(row["frequency"] for row in rows)
    assert all(row["polarity"] for row in rows)
    assert all(row["recipe"] for row in rows)
    assert all(
        context["recipe"]
        for row in rows
        for context in row.get("context", [])
    )


def test_macro_dashboard_defers_ism_only() -> None:
    config = load_yaml(default_config_path("macro_dashboard.yml"))

    assert {item["id"] for item in config["deferred"]} == {"ism_manufacturing_services"}


def test_macro_dashboard_declares_one_fred_datafeed_universe() -> None:
    config = load_yaml(default_config_path("macro_dashboard.yml"))
    configured = set(config["required_fred_series_ids"])
    required = {
        series_id
        for row in config["rows"]
        for series_id in row["series_ids"]
    }
    required.update(
        series_id
        for row in config["rows"]
        for context in row.get("context", [])
        for series_id in context["series_ids"]
    )

    assert required <= configured
    assert len(config["required_fred_series_ids"]) == len(configured)
    assert config["datafeed_additions"] == [
        "PCEC96",
        "ANFCI",
        "MORTGAGE30US",
        "BAMLC0A0CM",
        "BAMLC0A4CBBB",
        "BAMLH0A3HYC",
        "BAA10Y",
    ]
    assert set(config["datafeed_additions"]) <= configured


def test_macro_dashboard_has_freshness_thresholds_for_every_frequency() -> None:
    config = load_yaml(default_config_path("macro_dashboard.yml"))
    thresholds = config["freshness"]["stale_after_days"]
    frequencies = {
        definition["frequency"]
        for row in config["rows"]
        for definition in [row, *row.get("context", [])]
    }

    assert frequencies <= set(thresholds)
    assert all(int(thresholds[frequency]) > 0 for frequency in frequencies)

