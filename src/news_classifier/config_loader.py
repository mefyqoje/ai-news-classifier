from pathlib import Path

import yaml


def load_config(config_path: str | Path) -> dict:
    config_path = Path(config_path)

    if not config_path.exists():
        raise FileNotFoundError(f"Конфигурационный файл не найден: {config_path}")

    with open(config_path, encoding="utf-8") as file:
        return yaml.safe_load(file)