from __future__ import annotations

from typing import Any, Dict, List, Optional

import random

import numpy as np

from .payload import build_id_maps


def solve_nn_only(
    address_list: List[Dict[str, Any]],
    mat: np.ndarray,
    *,
    start_id: Optional[int] = None,
    end_id: Optional[int] = None,
) -> List[int]:
    """Deterministic nearest-neighbor (no 2-opt). Directed-safe."""
    id2idx, _ = build_id_maps(address_list)
    ids = [int(r["id"]) for r in address_list]
    if not ids:
        return []
    if len(ids) == 1:
        return [ids[0]]

    sid = int(start_id) if start_id is not None else ids[0]
    eid = int(end_id) if end_id is not None else None

    remaining = [i for i in ids if i != sid and i != eid]
    order = [sid]
    cur = sid
    while remaining:
        cur_idx = id2idx[cur]
        nxt = min(remaining, key=lambda nid: (mat[cur_idx, id2idx[nid]], id2idx[nid]))
        order.append(nxt)
        remaining.remove(nxt)
        cur = nxt

    if eid is not None:
        order.append(eid)
    return order


def randomized_nn_rcl(
    address_list: List[Dict[str, Any]],
    mat: np.ndarray,
    rnd: random.Random,
    *,
    start_id: Optional[int] = None,
    end_id: Optional[int] = None,
    rcl_size: int = 3,
) -> List[int]:
    """Randomized NN using restricted candidate list (top-k)."""
    id2idx, _ = build_id_maps(address_list)
    ids = [int(r["id"]) for r in address_list]
    if not ids:
        return []

    sid = int(start_id) if start_id is not None else ids[0]
    eid = int(end_id) if end_id is not None else None

    remaining = [i for i in ids if i != sid and i != eid]
    order = [sid]
    cur = sid

    while remaining:
        cur_idx = id2idx[cur]
        ranked = sorted(remaining, key=lambda nid: (mat[cur_idx, id2idx[nid]], id2idx[nid]))
        k = min(max(1, int(rcl_size)), len(ranked))
        nxt = rnd.choice(ranked[:k])
        order.append(nxt)
        remaining.remove(nxt)
        cur = nxt

    if eid is not None:
        order.append(eid)
    return order


def _best_insertion_delta_and_pos(
    order: List[int],
    nid: int,
    *,
    end_id: Optional[int],
    id2idx: Dict[int, int],
    mat: np.ndarray,
) -> tuple[float, int]:
    if not order:
        return 0.0, 0

    last = len(order) - 1
    if end_id is not None and order[-1] == int(end_id):
        last = len(order) - 2

    best_delta = float("inf")
    best_pos = len(order)

    for i in range(0, max(0, last) + 1):
        if i == len(order) - 1:
            a = id2idx[order[i]]
            b = id2idx[nid]
            delta = float(mat[a, b])
            pos = i + 1
        else:
            a = id2idx[order[i]]
            b = id2idx[nid]
            c = id2idx[order[i + 1]]
            delta = float((mat[a, b] + mat[b, c]) - mat[a, c])
            pos = i + 1

        if delta < best_delta:
            best_delta = delta
            best_pos = pos

    return best_delta, best_pos


def repair_regret2(
    partial: List[int],
    removed: List[int],
    rnd: random.Random,
    *,
    start_id: Optional[int],
    end_id: Optional[int],
    id2idx: Dict[int, int],
    mat: np.ndarray,
) -> List[int]:
    """Regret-2 insertion repair (directed-safe)."""
    order = list(partial)
    pool = list(removed)

    if start_id is not None:
        sid = int(start_id)
        if sid in pool:
            pool.remove(sid)
        if sid in order:
            order.remove(sid)
        order.insert(0, sid)

    while pool:
        best_choice = None
        for nid in pool:
            best_delta, best_pos = _best_insertion_delta_and_pos(
                order, nid, end_id=end_id, id2idx=id2idx, mat=mat
            )

            # compute second-best delta for regret
            if not order:
                second_delta = best_delta
            else:
                last = len(order) - 1
                if end_id is not None and order[-1] == int(end_id):
                    last = len(order) - 2

                first = (float("inf"), None)
                second = (float("inf"), None)

                for i in range(0, max(0, last) + 1):
                    if i == len(order) - 1:
                        a = id2idx[order[i]]
                        b = id2idx[nid]
                        delta = float(mat[a, b])
                    else:
                        a = id2idx[order[i]]
                        b = id2idx[nid]
                        c = id2idx[order[i + 1]]
                        delta = float((mat[a, b] + mat[b, c]) - mat[a, c])

                    if delta < first[0]:
                        second = first
                        first = (delta, i)
                    elif delta < second[0]:
                        second = (delta, i)

                second_delta = second[0] if second[1] is not None else first[0]

            regret = second_delta - best_delta

            # deterministic tie-break: rnd.random() is deterministic given seed
            score_key = (-regret, best_delta, rnd.random() * 1e-12)
            cand = (score_key, nid, best_pos)
            if best_choice is None or cand[0] < best_choice[0]:
                best_choice = cand

        _, chosen_nid, chosen_pos = best_choice
        order.insert(chosen_pos, chosen_nid)
        pool.remove(chosen_nid)

    if end_id is not None:
        eid = int(end_id)
        if eid in order:
            order.remove(eid)
        order.append(eid)

    return order


def _local_improve_relocate(
    order: List[int],
    *,
    start_id: Optional[int],
    end_id: Optional[int],
    id2idx: Dict[int, int],
    mat: np.ndarray,
    max_rounds: int = 2,
) -> List[int]:
    """Directed-safe local search: 1-node relocate."""
    if len(order) <= 3:
        return order

    sid = int(start_id) if start_id is not None else None
    eid = int(end_id) if end_id is not None else None

    def arc(a: int, b: int) -> float:
        return float(mat[id2idx[a], id2idx[b]])

    best = list(order)
    for _ in range(max_rounds):
        improved = False

        start_i = 1 if sid is not None and best and best[0] == sid else 0
        end_i = (len(best) - 2) if eid is not None and best and best[-1] == eid else (len(best) - 1)

        for i in range(start_i, end_i + 1):
            x = best[i]
            if sid is not None and x == sid:
                continue
            if eid is not None and x == eid:
                continue

            prev = best[i - 1] if i - 1 >= 0 else None
            nxt = best[i + 1] if i + 1 < len(best) else None

            if prev is not None and nxt is not None:
                remove_delta = -arc(prev, x) - arc(x, nxt) + arc(prev, nxt)
            elif prev is not None:
                remove_delta = -arc(prev, x)
            elif nxt is not None:
                remove_delta = -arc(x, nxt)
            else:
                remove_delta = 0.0

            tmp = best[:i] + best[i + 1 :]

            tmp_start = 1 if sid is not None and tmp and tmp[0] == sid else 0
            tmp_end = (len(tmp) - 1) if eid is not None and tmp and tmp[-1] == eid else len(tmp)

            best_move_delta = 0.0
            best_j = None

            for j in range(tmp_start, tmp_end + 1):
                a = tmp[j - 1] if j - 1 >= 0 else None
                b = tmp[j] if j < len(tmp) else None

                if a is not None and b is not None:
                    insert_delta = arc(a, x) + arc(x, b) - arc(a, b)
                elif a is not None:
                    insert_delta = arc(a, x)
                elif b is not None:
                    insert_delta = arc(x, b)
                else:
                    insert_delta = 0.0

                total = remove_delta + insert_delta
                if total < best_move_delta - 1e-12:
                    best_move_delta = total
                    best_j = j

            if best_j is not None:
                best = tmp[:best_j] + [x] + tmp[best_j:]
                improved = True
                break

        if not improved:
            break

    return best


def calc_cost(order: List[int], id2idx: Dict[int, int], mat: np.ndarray) -> float:
    if len(order) <= 1:
        return 0.0
    idx = [id2idx[int(i)] for i in order]
    return float(sum(mat[idx[k], idx[k + 1]] for k in range(len(idx) - 1)))


def build_init_ids_multi_start(
    address_list: List[Dict[str, Any]],
    mat: np.ndarray,
    rnd: random.Random,
    *,
    start_id: Optional[int],
    end_id: Optional[int],
    trials: int = 20,
    rcl_size: int = 3,
    use_regret2: bool = True,
    use_relocate_ls: bool = True,
    relocate_rounds: int = 2,
) -> List[int]:
    """Create several initial solutions and pick the best by cost."""
    id2idx, _ = build_id_maps(address_list)

    candidates: List[List[int]] = []
    candidates.append(solve_nn_only(address_list, mat, start_id=start_id, end_id=end_id))

    for _ in range(max(0, int(trials))):
        candidates.append(
            randomized_nn_rcl(
                address_list,
                mat,
                rnd,
                start_id=start_id,
                end_id=end_id,
                rcl_size=rcl_size,
            )
        )

    if use_regret2 and len(address_list) >= 2:
        ids = [int(r["id"]) for r in address_list]
        sid = int(start_id) if start_id is not None else ids[0]
        eid = int(end_id) if end_id is not None else None

        partial = [sid]
        pool = [i for i in ids if i != sid and i != eid]
        candidates.append(
            repair_regret2(
                partial,
                pool,
                rnd,
                start_id=sid,
                end_id=eid,
                id2idx=id2idx,
                mat=mat,
            )
        )

    best_order: Optional[List[int]] = None
    best_cost = float("inf")

    for o in candidates:
        order = list(o)
        if start_id is not None:
            sid = int(start_id)
            if sid in order:
                order.remove(sid)
            order.insert(0, sid)
        if end_id is not None:
            eid = int(end_id)
            if eid in order:
                order.remove(eid)
            order.append(eid)

        if use_relocate_ls:
            order = _local_improve_relocate(
                order,
                start_id=start_id,
                end_id=end_id,
                id2idx=id2idx,
                mat=mat,
                max_rounds=relocate_rounds,
            )

        c = calc_cost(order, id2idx, mat)
        if c < best_cost:
            best_cost = c
            best_order = order

    return best_order if best_order is not None else candidates[0]
