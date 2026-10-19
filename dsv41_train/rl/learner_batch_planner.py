"""Deterministic response packing using explicitly qualified physical shapes.

This module never infers GPU capacity. Supply only B:T cells validated against
the deployed full model. Uncovered lengths fail before any optimizer mutation.
"""
from __future__ import annotations


def parse_learner_batch_shapes(text):
    if text is None:
        return None
    if not isinstance(text, str) or not text:
        raise ValueError("learner batch shapes require comma-separated integer B:T cells")
    cells = []
    for item in text.split(","):
        fields = item.split(":")
        if len(fields) != 2 or any(not value.isascii() or not value.isdecimal() for value in fields):
            raise ValueError("learner batch shapes require comma-separated integer B:T cells")
        width, tokens = map(int, fields)
        if not 1 <= width <= 32 or not 1 <= tokens <= 2047:
            raise ValueError("learner batch shape bounds must be B1..32:T1..2047")
        cell = {"max_batch_size": width, "max_model_input_tokens": tokens}
        if cell in cells:
            raise ValueError("duplicate learner batch shape cell")
        cells.append(cell)
    return sorted(cells, key=lambda cell: (cell["max_batch_size"], cell["max_model_input_tokens"]))


def plan_learner_microbatches(input_lengths, fixed_size, shapes=None):
    """Cover each original response once, preserving its index and loss weight.

    Optional cells pack a stable length-sorted prefix using the widest eligible
    physical batch. Ties use the tighter token bound, then the smaller capacity
    cell. A partial batch is allowed within its measured B/T ceilings. Missing
    length coverage fails before scoring, backward or optimizer mutation.
    """
    if (type(fixed_size) is not int or not 1 <= fixed_size <= 32
            or not input_lengths or any(type(n) is not int or not 1 <= n <= 2047 for n in input_lengths)):
        raise ValueError("learner planner requires physical B1..32 and nonempty input lengths1..2047")
    if shapes is None:
        return [{"response_indices": list(range(start, min(start + fixed_size, len(input_lengths)))),
                 "qualified_shape_cell": None}
                for start in range(0, len(input_lengths), fixed_size)]
    if not shapes or any(set(cell) != {"max_batch_size", "max_model_input_tokens"}
            or type(cell["max_batch_size"]) is not int or not 1 <= cell["max_batch_size"] <= 32
            or type(cell["max_model_input_tokens"]) is not int or not 1 <= cell["max_model_input_tokens"] <= 2047
            for cell in shapes):
        raise ValueError("learner planner requires explicit qualified B:T cells")
    if any(not any(length <= cell["max_model_input_tokens"] for cell in shapes) for length in input_lengths):
        raise ValueError("qualified learner batch shapes do not cover every actual response length")
    order = sorted(range(len(input_lengths)), key=lambda index: (input_lengths[index], index))
    planned, cursor = [], 0
    while cursor < len(order):
        candidates = []
        for cell in shapes:
            count = 0
            while (count < cell["max_batch_size"] and cursor + count < len(order)
                    and input_lengths[order[cursor + count]] <= cell["max_model_input_tokens"]):
                count += 1
            if count:
                candidates.append((count, cell))
        count, cell = min(candidates, key=lambda row:
            (-row[0], row[1]["max_model_input_tokens"], row[1]["max_batch_size"]))
        planned.append({"response_indices": order[cursor:cursor + count],
                        "qualified_shape_cell": dict(cell)})
        cursor += count
    return planned
