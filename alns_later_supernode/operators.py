from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import random

import numpy as np


@dataclass(frozen=True)
class RouteState:
    order: Tuple[int, ...]
    cost: float

    def objective(self) -> float:
        return float(self.cost)


def calc_cost(order: List[int], id2idx: Dict[int, int], mat: np.ndarray) -> float:
    if len(order) <= 1:
        return 0.0
    idx = [id2idx[int(i)] for i in order]
    return float(sum(mat[idx[k], idx[k + 1]] for k in range(len(idx) - 1)))


def freeze(order: List[int], id2idx: Dict[int, int], mat: np.ndarray) -> RouteState:
    return RouteState(order=tuple(int(x) for x in order), cost=calc_cost(order, id2idx, mat))


# -------- Destroy --------

def destroy_random(state: RouteState, rnd: random.Random, start_id: Optional[int], end_id: Optional[int]) -> Tuple[List[int], List[int]]:
    order = list(state.order)
    protected = set()
    if start_id is not None:
        protected.add(int(start_id))
    if end_id is not None:
        protected.add(int(end_id))

    candidates = [nid for nid in order if nid not in protected]
    if len(candidates) <= 2:
        return order, []

    k = max(1, int(len(candidates) * rnd.uniform(0.10, 0.25)))
    removed = rnd.sample(candidates, k=min(k, len(candidates)))
    partial = [nid for nid in order if nid not in removed]
    return partial, removed


def destroy_worst(
    state: RouteState,
    rnd: random.Random,
    start_id: Optional[int],
    end_id: Optional[int],
    id2idx: Dict[int, int],
    mat: np.ndarray,
) -> Tuple[List[int], List[int]]:
    order = list(state.order)

    protected = set()
    if start_id is not None:
        protected.add(int(start_id))
    if end_id is not None:
        protected.add(int(end_id))

    if len(order) <= 3:
        return order, []

    idx = [id2idx[i] for i in order]
    scores = []
    for pos in range(1, len(order) - 1):
        nid = order[pos]
        if nid in protected:
            continue
        a, b, c = idx[pos - 1], idx[pos], idx[pos + 1]
        saving = (mat[a, b] + mat[b, c]) - mat[a, c]
        scores.append((saving, nid))

    if not scores:
        return order, []

    scores.sort(reverse=True)
    candidates = [nid for _, nid in scores]
    k = max(1, int(len(candidates) * 0.15))
    top = candidates[: max(k * 2, 3)]
    removed = rnd.sample(top, k=min(k, len(top)))

    partial = [nid for nid in order if nid not in removed]
    return partial, removed


def destroy_segment(state: RouteState, rnd: random.Random, start_id: Optional[int], end_id: Optional[int]) -> Tuple[List[int], List[int]]:
    order = list(state.order)

    protected = set()
    if start_id is not None:
        protected.add(int(start_id))
    if end_id is not None:
        protected.add(int(end_id))

    candidates_idx = [i for i, nid in enumerate(order) if nid not in protected]
    if len(candidates_idx) <= 2:
        return order, []

    seg_len = max(1, int(len(candidates_idx) * rnd.uniform(0.10, 0.30)))
    seg_len = min(seg_len, len(candidates_idx))

    start_pos = rnd.choice(candidates_idx)

    removed: List[int] = []
    pos = start_pos
    seen = 0
    max_steps = len(order) * 2

    while len(removed) < seg_len and seen < max_steps:
        nid = order[pos]
        if nid not in protected and nid not in removed:
            removed.append(nid)
        pos = (pos + 1) % len(order)
        seen += 1

    partial = [nid for nid in order if nid not in removed]
    return partial, removed


# -------- Repair --------

def repair_greedy(
    partial: List[int],
    removed: List[int],
    rnd: random.Random,
    start_id: Optional[int],
    end_id: Optional[int],
    id2idx: Dict[int, int],
    mat: np.ndarray,
) -> List[int]:
    order = list(partial)
    pool = list(removed)
    rnd.shuffle(pool)

    for nid in pool:
        best_pos = None
        best_delta = float("inf")

        last = len(order) - 1
        if end_id is not None and order and order[-1] == int(end_id):
            last = len(order) - 2

        for i in range(0, max(0, last) + 1):
            if i == len(order) - 1:
                a = id2idx[order[i]]
                b = id2idx[nid]
                delta = mat[a, b]
            else:
                a = id2idx[order[i]]
                b = id2idx[nid]
                c = id2idx[order[i + 1]]
                delta = (mat[a, b] + mat[b, c]) - mat[a, c]

            if delta < best_delta:
                best_delta = delta
                best_pos = i + 1

        if best_pos is None:
            order.append(nid)
        else:
            order.insert(best_pos, nid)

    if start_id is not None:
        sid = int(start_id)
        if not order or order[0] != sid:
            if sid in order:
                order.remove(sid)
            order.insert(0, sid)

    if end_id is not None:
        eid = int(end_id)
        if not order or order[-1] != eid:
            if eid in order:
                order.remove(eid)
            order.append(eid)

    return order
