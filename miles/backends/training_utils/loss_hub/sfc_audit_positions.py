"""Deterministic sparse audit selection from original valid response positions."""


def select_positions(metadata, indices, *, count=32, sign_aware=False):
    by_sign = {1: [], -1: [], 0: []}
    for index, sample in zip(indices, metadata, strict=True):
        advantage = sample.get("advantage", 0.0)
        sign = 1 if advantage > 0 else -1 if advantage < 0 else 0
        by_sign[sign].extend((int(index), int(position)) for position in sample["first_valid_positions"])
    all_positions = sorted(position for values in by_sign.values() for position in values)
    if not sign_aware:
        return all_positions[:count]
    selected = sorted(by_sign[1])[:count // 2] + sorted(by_sign[-1])[:count // 2]
    chosen = set(selected)
    selected.extend(position for position in all_positions if position not in chosen)
    return sorted(selected[:count])
