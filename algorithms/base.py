def common_strategy_config(config: dict) -> dict:
    return {
        "fraction_train": float(config.get("fraction-train", 1.0)),
        "fraction_evaluate": float(config.get("fraction-evaluate", 1.0)),
    }