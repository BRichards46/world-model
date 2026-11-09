from __future__ import annotations

from world_model.config import default_config_path, load_yaml


def test_directional_momentum_universe_and_horizons() -> None:
    config = load_yaml(default_config_path("directional_momentum.yml"))

    assert [item["id"] for item in config["horizons"]] == [
        "one_month",
        "three_months",
        "six_months",
        "ten_months",
        "twelve_months",
    ]
    assert sum(float(item["weight"]) for item in config["horizons"]) == 1.0
    assert [item["id"] for item in config["futures"]] == [
        "ES", "NQ", "RTY", "ZB", "ZN", "DX", "GC", "SI", "CL", "BO", "FC", "SB", "KC", "KW"
    ]
    assert [item["id"] for item in config["equities"]["assets"]] == [
        "XLC", "XLY", "XLP", "XLE", "XLF", "XLV", "XLI", "XLB", "XLRE", "XLK", "XLU"
    ]
    assert config["equities"]["benchmark"]["id"] == "SPY"
