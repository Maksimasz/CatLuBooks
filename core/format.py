"""Мелкие форматтеры, общие для интерфейса и модулей."""
from __future__ import annotations


def human_size(size: int) -> str:
    """«1536» -> «1,5 КБ»: размеры в интерфейсе читаются глазом."""
    value = float(size)
    for unit in ("Б", "КБ", "МБ", "ГБ"):
        if value < 1024 or unit == "ГБ":
            return f"{value:.0f} {unit}" if unit == "Б" else f"{value:.1f} {unit}"
        value /= 1024
    return f"{value:.1f} ГБ"
