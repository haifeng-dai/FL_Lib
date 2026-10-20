import unicodedata
from collections.abc import Sequence
from math import ceil, floor
from typing import Any

MAX_CHART_POINTS = 100


def print_table(
    headers: list[str],
    rows: list[list[Any]],
    aligns: Sequence[str] | None = None,
) -> None:
    values = [[str(value) for value in row] for row in rows]
    widths = [display_width(header) for header in headers]
    for row in values:
        for index, value in enumerate(row):
            widths[index] = max(widths[index], display_width(value))
    print(" | ".join(_pad(value, width) for value, width in zip(headers, widths)))
    print("-+-".join("-" * width for width in widths))
    for row in values:
        formatted = [
            _pad(
                value,
                width,
                align=aligns[index] if aligns and index < len(aligns) else "left",
            )
            for index, (value, width) in enumerate(zip(row, widths))
        ]
        print(" | ".join(formatted))


def display_width(value: str) -> int:
    return sum(2 if unicodedata.east_asian_width(char) in "WF" else 1 for char in value)


def _pad(value: str, width: int, align: str = "left") -> str:
    padding = " " * (width - display_width(value))
    if align in ("right", ">"):
        return padding + value
    return value + padding


def render_accuracy_chart(points: list[tuple[int, float]], width: int = 100) -> str:
    if not points:
        return "数据库中没有逐轮 accuracy 数据。"

    rounds = [point[0] for point in points]
    accuracies = [point[1] for point in points]
    min_round, max_round = min(rounds), max(rounds)
    low = max(0.0, floor(min(accuracies) / 5) * 5)
    high = min(100.0, ceil(max(accuracies) / 5) * 5)
    if high == low:
        if high == 100:
            low -= 5
        else:
            high += 5
    height = round((high - low) / 5) + 1

    ordered_points = sorted(points)
    if len(ordered_points) > MAX_CHART_POINTS:
        last_index = len(ordered_points) - 1
        ordered_points = [
            ordered_points[round(index * last_index / (MAX_CHART_POINTS - 1))]
            for index in range(MAX_CHART_POINTS)
        ]

    grid = [[" " for _ in range(width)] for _ in range(height)]

    def coordinates(point: tuple[int, float]) -> tuple[int, int]:
        round_index, accuracy = point
        column = (
            0
            if max_round == min_round
            else round(
                (round_index - min_round) * (width - 1) / (max_round - min_round)
            )
        )
        row = round((high - accuracy) / 5)
        return column, row

    for point in ordered_points:
        column, row = coordinates(point)
        grid[row][column] = "•"

    lines = [
        f"{high - row * 5:6.0f} │" + "".join(values) for row, values in enumerate(grid)
    ]
    lines.append("       └" + "─" * width)
    tick_line = [" " for _ in range(width)]
    for index in range(5):
        tick_value = round(min_round + (max_round - min_round) * index / 4)
        label = str(tick_value)
        column = round((width - 1) * index / 4)
        start = max(0, min(width - len(label), column - len(label) // 2))
        tick_line[start : start + len(label)] = label
    lines.append("        " + "".join(tick_line))
    return "\n".join(lines)
