# alns_pipeline_later_supernode.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Any

import os
import json
import time
import random
import hashlib
import re
from pathlib import Path

import numpy as np
import pandas as pd

from alns import ALNS
from alns.accept import SimulatedAnnealing
from alns.stop import MaxIterations


# =========================================================
# Determinism + Cache Helpers
# =========================================================
DEFAULT_CACHE_DIR = ".alns_cache"
DEFAULT_MATRIX_UNIT = "sec"  # "sec" or "ms"
PROBLEM_SALT = "alns-later-supernode-v1"


def _set_deterministic(seed: int) -> None:
    """
    전체 실행을 최대한 결정적으로 만들기 위한 방어.
    - random / numpy seed 고정
    - BLAS 계열 스레드 1로 고정 (미세한 비결정성 완화)
    """
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
    os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

    random.seed(seed)
    np.random.seed(seed)


def _quantize_matrix(m: np.ndarray, *, unit: str = DEFAULT_MATRIX_UNIT) -> np.ndarray:
    """
    float 미세오차로 인한 tie-break 흔들림 방지:
    - sec: 초 단위 반올림 후 int64
    - ms : 밀리초 단위 반올림 후 int64
    """
    m = np.asarray(m, dtype=float)
    if unit == "sec":
        return np.rint(m).astype(np.int64)
    if unit == "ms":
        return np.rint(m * 1000.0).astype(np.int64)
    raise ValueError("unit must be 'sec' or 'ms'")


def _problem_key(
    m_int: np.ndarray,
    *,
    node_ids: List[int],
    start_id: Optional[int],
    end_id: Optional[int],
    opts: Dict[str, Any],
    salt: str = PROBLEM_SALT,
) -> str:
    """
    '같은 문제'를 식별하기 위한 해시 키:
    - dist_matrix(정수화)
    - node_ids (행/열 매핑; address_list 순서의 id 라벨)
    - start/end
    - opts (max_iters, init 파라미터, SA 스케일 등)
    """
    h = hashlib.sha256()
    h.update(salt.encode("utf-8"))
    h.update(np.ascontiguousarray(m_int).tobytes())
    h.update(("|node_ids:" + ",".join(map(str, node_ids))).encode("utf-8"))
    h.update(("|start_id:" + str(start_id)).encode("utf-8"))
    h.update(("|end_id:" + str(end_id)).encode("utf-8"))
    h.update(("|opts:" + json.dumps(opts, sort_keys=True, ensure_ascii=False)).encode("utf-8"))
    return h.hexdigest()


def _seed_from_key(key_hex: str) -> int:
    """sha256 hex 앞 8자리로 32-bit seed 생성"""
    return int(key_hex[:8], 16)


def _cache_path(cache_dir: str, key_hex: str) -> Path:
    d = Path(cache_dir)
    d.mkdir(parents=True, exist_ok=True)
    return d / f"alns_{key_hex[:16]}.json"


def _load_cache(cache_dir: str, key_hex: str) -> Optional[Dict[str, Any]]:
    p = _cache_path(cache_dir, key_hex)
    if not p.exists():
        return None
    with p.open("r", encoding="utf-8") as f:
        return json.load(f)


def _save_cache(cache_dir: str, key_hex: str, data: Dict[str, Any]) -> None:
    p = _cache_path(cache_dir, key_hex)
    with p.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


# =========================
# Helpers (payload / matrix / id map)
# =========================
def _get_address_list(payload: Dict[str, Any]) -> List[Dict[str, Any]]:
    if "address_list" in payload and payload["address_list"]:
        return payload["address_list"]
    if "address_geocode_list" in payload and payload["address_geocode_list"]:
        return payload["address_geocode_list"]
    raise KeyError("payload must contain 'address_list' or 'address_geocode_list'.")


def _tracking_number_to_id(payload: Dict[str, Any], tracking_number: Optional[str]) -> Optional[int]:
    """
    payload 안에서 tracking_number -> id로 변환.
    - tracking_number가 None이면 None 반환
    - tracking_number 중복이면 에러 (모호)
    """
    if tracking_number is None:
        return None

    tn = str(tracking_number).strip()
    if not tn:
        return None

    address_list = _get_address_list(payload)

    found_ids = []
    for rec in address_list:
        rec_tn = str(rec.get("tracking_number") or "").strip()
        if rec_tn == tn:
            found_ids.append(int(rec["id"]))

    if not found_ids:
        raise ValueError(f"tracking_number='{tn}' not found in payload.")

    if len(found_ids) > 1:
        raise ValueError(f"tracking_number='{tn}' is not unique in payload. matched ids={found_ids}")

    return found_ids[0]


def _get_matrix(payload: Dict[str, Any], matrix_key: str) -> np.ndarray:
    if matrix_key not in payload:
        raise KeyError(f"payload must contain '{matrix_key}'.")
    m = np.array(payload[matrix_key], dtype=float)
    if m.ndim != 2 or m.shape[0] != m.shape[1]:
        raise ValueError(f"{matrix_key} must be a square 2D matrix. got shape={m.shape}")
    return m


def _build_id_maps(address_list: List[Dict[str, Any]]) -> Tuple[Dict[int, int], Dict[int, Dict[str, Any]]]:
    # id -> matrix index (address_list 순서)
    id2idx = {int(rec["id"]): i for i, rec in enumerate(address_list)}
    id2rec = {int(rec["id"]): rec for rec in address_list}
    return id2idx, id2rec


def _stable_seed_from_payload(payload: Dict[str, Any], matrix_key: str, *, salt: str = "alns-v2") -> int:
    """
    (레거시) 동일 입력(payload 주소들)이면 항상 동일 seed.
    지금은 기본 정책을 "dist_matrix 기반"으로 바꿨지만,
    seed를 강제로 주거나 fallback이 필요할 때를 위해 남겨둠.
    """
    addr = _get_address_list(payload)
    parts = []
    for r in sorted(addr, key=lambda x: int(x["id"])):
        parts.append(f'{int(r["id"])}|{r.get("tracking_number","")}|{r.get("lat","")}|{r.get("lng","")}')
    raw = salt + "|" + matrix_key + "|" + "||".join(parts)
    h = hashlib.sha256(raw.encode("utf-8")).hexdigest()
    return int(h[:8], 16)  # 32-bit seed


def _get_area(rec: Dict[str, Any]) -> Any:
    return rec.get("Area") or rec.get("area") or rec.get("region") or rec.get("code_base")


# =========================
# Unit remap: unit_original_id -> 0 (id label only)
# =========================
def _remap_unit_to_zero(
    payload: Dict[str, Any],
    *,
    unit_original_id: int,
    matrix_key: str = "dist_matrix",
) -> Tuple[Dict[str, Any], Dict[int, int]]:
    """
    unit_original_id → 0 으로 'id 라벨만' 재매핑.
    - address_list 순서(=matrix index 기준)는 그대로 유지
    - matrix는 재정렬하지 않음
    """
    address_list = _get_address_list(payload)
    old_ids = [int(r["id"]) for r in address_list]
    if int(unit_original_id) not in old_ids:
        raise ValueError(f"unit_original_id={unit_original_id} not found in payload ids.")

    new_to_old: Dict[int, int] = {}
    used = set()
    next_id = 1

    new_address_list = []
    for rec in address_list:
        old_id = int(rec["id"])
        if old_id == int(unit_original_id):
            new_id = 0
        else:
            while next_id in used or next_id == 0:
                next_id += 1
            new_id = next_id
            next_id += 1

        used.add(new_id)
        new_to_old[new_id] = old_id

        new_rec = dict(rec)
        new_rec["id"] = new_id
        new_address_list.append(new_rec)

    new_payload = dict(payload)
    if "address_list" in payload and payload["address_list"]:
        new_payload["address_list"] = new_address_list
    else:
        new_payload["address_geocode_list"] = new_address_list

    new_payload.setdefault("meta", {})
    new_payload["meta"].update(
        {
            "unit_remapped_to_zero": True,
            "unit_original_id": int(unit_original_id),
            "matrix_key": matrix_key,
        }
    )

    return new_payload, new_to_old


# =========================
# ALNS op_select (compat)
# =========================
class RandomOpSelectCompat:
    def __init__(self, num_destroy: int, num_repair: int):
        if num_destroy <= 0:
            raise ValueError("No destroy operators registered.")
        if num_repair <= 0:
            raise ValueError("No repair operators registered.")
        self.num_destroy = num_destroy
        self.num_repair = num_repair

    def __call__(self, *args, **kwargs):
        rng = args[0] if args else None
        if hasattr(rng, "randrange"):
            d_idx = rng.randrange(self.num_destroy)
            r_idx = rng.randrange(self.num_repair)
        else:
            d_idx = random.randrange(self.num_destroy)
            r_idx = random.randrange(self.num_repair)
        return d_idx, r_idx

    def update(self, *args, **kwargs):
        return None

    def select(self, *args, **kwargs):
        return self.__call__(*args, **kwargs)


# =========================
# RouteState (objective)
# =========================
@dataclass(frozen=True)
class RouteState:
    order: Tuple[int, ...]
    cost: float

    def objective(self) -> float:
        return float(self.cost)


def _calc_cost(order: List[int], id2idx: Dict[int, int], mat: np.ndarray) -> float:
    if len(order) <= 1:
        return 0.0
    idx = [id2idx[int(i)] for i in order]
    return float(sum(mat[idx[k], idx[k + 1]] for k in range(len(idx) - 1)))


def _freeze(order: List[int], id2idx: Dict[int, int], mat: np.ndarray) -> RouteState:
    return RouteState(order=tuple(int(x) for x in order), cost=_calc_cost(order, id2idx, mat))


# =========================
# Init improvement: Multi-start (RCL NN + Regret2) + relocate local search
# =========================
def solve_nn_only(
    address_list: List[Dict[str, Any]],
    mat: np.ndarray,
    *,
    start_id: int | None = None,
    end_id: int | None = None,
) -> List[int]:
    """Deterministic nearest-neighbor only (NO 2-opt). Safe for directed matrices."""
    id2idx, _ = _build_id_maps(address_list)
    n = len(address_list)
    if n == 0:
        return []
    if n == 1:
        return [int(address_list[0]["id"])]

    start_idx = id2idx[int(start_id)] if start_id is not None else 0
    end_idx = id2idx[int(end_id)] if end_id is not None else None

    unvisited = list(range(n))
    unvisited.remove(start_idx)

    tour = [start_idx]
    cur = start_idx
    while unvisited:
        nxt = min(unvisited, key=lambda j: (mat[cur, j], j))
        tour.append(nxt)
        unvisited.remove(nxt)
        cur = nxt

    if end_idx is not None:
        if end_idx in tour:
            tour.remove(end_idx)
        tour.append(end_idx)

    return [int(address_list[i]["id"]) for i in tour]


def randomized_nn_rcl(
    address_list: List[Dict[str, Any]],
    mat: np.ndarray,
    rnd: random.Random,
    *,
    start_id: int | None = None,
    end_id: int | None = None,
    rcl_size: int = 3,
) -> List[int]:
    """Randomized NN using Restricted Candidate List (top-k nearest)."""
    id2idx, _ = _build_id_maps(address_list)
    ids = [int(r["id"]) for r in address_list]
    if not ids:
        return []

    start = int(start_id) if start_id is not None else ids[0]
    eid = int(end_id) if end_id is not None else None

    remaining = [i for i in ids if i != start and i != eid]
    order = [start]
    cur = start

    while remaining:
        cur_idx = id2idx[cur]
        ranked = sorted(remaining, key=lambda nid: mat[cur_idx, id2idx[nid]])
        k = min(max(1, int(rcl_size)), len(ranked))
        nxt = rnd.choice(ranked[:k])
        order.append(nxt)
        remaining.remove(nxt)
        cur = nxt

    if eid is not None:
        order.append(eid)

    return order


def _local_improve_relocate(
    order: List[int],
    *,
    start_id: int | None,
    end_id: int | None,
    id2idx: Dict[int, int],
    mat: np.ndarray,
    max_rounds: int = 3,
) -> List[int]:
    """
    Directed-safe local search: relocate(1-node move).
    """
    if len(order) <= 3:
        return order

    sid = int(start_id) if start_id is not None else None
    eid = int(end_id) if end_id is not None else None

    def arc(a: int, b: int) -> float:
        return float(mat[id2idx[a], id2idx[b]])

    best_order = list(order)

    for _ in range(max_rounds):
        improved = False

        start_i = 0
        end_i = len(best_order) - 1

        if sid is not None and best_order[0] == sid:
            start_i = 1
        if eid is not None and best_order[-1] == eid:
            end_i = len(best_order) - 2

        for i in range(start_i, end_i + 1):
            x = best_order[i]
            if sid is not None and x == sid:
                continue
            if eid is not None and x == eid:
                continue

            prev = best_order[i - 1] if i - 1 >= 0 else None
            nxt = best_order[i + 1] if i + 1 < len(best_order) else None

            remove_delta = 0.0
            if prev is not None and nxt is not None:
                remove_delta = -arc(prev, x) - arc(x, nxt) + arc(prev, nxt)
            elif prev is not None and nxt is None:
                remove_delta = -arc(prev, x)
            elif prev is None and nxt is not None:
                remove_delta = -arc(x, nxt)

            tmp = best_order[:i] + best_order[i + 1 :]

            tmp_start = 0
            tmp_end = len(tmp)

            if sid is not None and tmp and tmp[0] == sid:
                tmp_start = 1
            if eid is not None and tmp and tmp[-1] == eid:
                tmp_end = len(tmp) - 1

            best_move_delta = 0.0
            best_j = None

            for j in range(tmp_start, tmp_end + 1):
                a = tmp[j - 1] if j - 1 >= 0 else None
                b = tmp[j] if j < len(tmp) else None

                insert_delta = 0.0
                if a is not None and b is not None:
                    insert_delta = arc(a, x) + arc(x, b) - arc(a, b)
                elif a is not None and b is None:
                    insert_delta = arc(a, x)
                elif a is None and b is not None:
                    insert_delta = arc(x, b)

                total_delta = remove_delta + insert_delta
                if total_delta < best_move_delta - 1e-12:
                    best_move_delta = total_delta
                    best_j = j

            if best_j is not None:
                best_order = tmp[:best_j] + [x] + tmp[best_j:]
                improved = True
                break

        if not improved:
            break

    return best_order


def build_init_ids_multi_start(
    address_list: List[Dict[str, Any]],
    mat: np.ndarray,
    rnd: random.Random,
    *,
    start_id: int | None,
    end_id: int | None,
    trials: int = 20,
    rcl_size: int = 3,
    use_regret2: bool = True,
    use_relocate_ls: bool = True,
    relocate_rounds: int = 2,
) -> List[int]:
    """
    초기해 후보 여러 개를 만들고(cost 최소) best 선택.
    """
    id2idx, _ = _build_id_maps(address_list)

    def cost(order_ids: List[int]) -> float:
        return _calc_cost(order_ids, id2idx, mat)

    candidates: List[List[int]] = []

    candidates.append(solve_nn_only(address_list, mat, start_id=start_id, end_id=end_id))

    for _ in range(max(0, int(trials))):
        candidates.append(
            randomized_nn_rcl(address_list, mat, rnd, start_id=start_id, end_id=end_id, rcl_size=rcl_size)
        )

    if use_regret2 and len(address_list) >= 2:
        ids = [int(r["id"]) for r in address_list]
        sid = int(start_id) if start_id is not None else ids[0]
        eid = int(end_id) if end_id is not None else None

        partial = [sid]
        pool = [i for i in ids if i != sid and i != eid]
        regret_order = repair_regret2(partial, pool, rnd, start_id=sid, end_id=eid, id2idx=id2idx, mat=mat)
        candidates.append(regret_order)

    best_order = None
    best_cost = float("inf")

    for ord0 in candidates:
        ord1 = list(ord0)

        if start_id is not None:
            sid = int(start_id)
            if sid in ord1:
                ord1.remove(sid)
            ord1.insert(0, sid)

        if end_id is not None:
            eid = int(end_id)
            if eid in ord1:
                ord1.remove(eid)
            ord1.append(eid)

        if use_relocate_ls:
            ord1 = _local_improve_relocate(
                ord1,
                start_id=start_id,
                end_id=end_id,
                id2idx=id2idx,
                mat=mat,
                max_rounds=relocate_rounds,
            )

        c = cost(ord1)
        if c < best_cost:
            best_cost = c
            best_order = ord1

    return best_order if best_order is not None else candidates[0]


# =========================
# Destroy / Repair
# =========================
def destroy_random(
    state: RouteState, rnd: random.Random, start_id: int | None, end_id: int | None
) -> Tuple[List[int], List[int]]:
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
    start_id: int | None,
    end_id: int | None,
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


def destroy_segment(
    state: RouteState,
    rnd: random.Random,
    start_id: int | None,
    end_id: int | None,
) -> Tuple[List[int], List[int]]:
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

    removed = []
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


def repair_greedy(
    partial: List[int],
    removed: List[int],
    rnd: random.Random,
    start_id: int | None,
    end_id: int | None,
    id2idx: Dict[int, int],
    mat: np.ndarray,
) -> List[int]:
    order = list(partial)
    removed = list(removed)
    rnd.shuffle(removed)

    for nid in removed:
        best_pos = None
        best_delta = float("inf")

        last = len(order) - 1
        if end_id is not None and len(order) > 0 and order[-1] == int(end_id):
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
        if len(order) == 0 or order[0] != sid:
            if sid in order:
                order.remove(sid)
            order.insert(0, sid)

    if end_id is not None:
        eid = int(end_id)
        if len(order) == 0 or order[-1] != eid:
            if eid in order:
                order.remove(eid)
            order.append(eid)

    return order


def _best_insertion_delta_and_pos(
    order: List[int],
    nid: int,
    *,
    end_id: int | None,
    id2idx: Dict[int, int],
    mat: np.ndarray,
) -> Tuple[float, int]:
    if not order:
        return 0.0, 0

    last = len(order) - 1
    if end_id is not None and len(order) > 0 and order[-1] == int(end_id):
        last = len(order) - 2

    best_delta = float("inf")
    best_pos = len(order)

    for i in range(0, max(0, last) + 1):
        if i == len(order) - 1:
            a = id2idx[order[i]]
            b = id2idx[nid]
            delta = mat[a, b]
            pos = i + 1
        else:
            a = id2idx[order[i]]
            b = id2idx[nid]
            c = id2idx[order[i + 1]]
            delta = (mat[a, b] + mat[b, c]) - mat[a, c]
            pos = i + 1

        if delta < best_delta:
            best_delta = delta
            best_pos = pos

    return best_delta, best_pos


def repair_regret2(
    partial: List[int],
    removed: List[int],
    rnd: random.Random,
    start_id: int | None,
    end_id: int | None,
    id2idx: Dict[int, int],
    mat: np.ndarray,
) -> List[int]:
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
            best_delta, best_pos = _best_insertion_delta_and_pos(order, nid, end_id=end_id, id2idx=id2idx, mat=mat)

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
                        delta = mat[a, b]
                    else:
                        a = id2idx[order[i]]
                        b = id2idx[nid]
                        c = id2idx[order[i + 1]]
                        delta = (mat[a, b] + mat[b, c]) - mat[a, c]

                    if delta < first[0]:
                        second = first
                        first = (delta, i)
                    elif delta < second[0]:
                        second = (delta, i)

                second_delta = second[0] if second[1] is not None else first[0]

            regret = second_delta - best_delta

            # rnd는 seed로 결정되므로 완전 재현 가능
            score_key = (-regret, best_delta, rnd.random() * 1e-12)

            cand = (score_key, nid, best_pos, best_delta)
            if best_choice is None or cand[0] < best_choice[0]:
                best_choice = cand

        _, chosen_nid, chosen_pos, _ = best_choice
        order.insert(chosen_pos, chosen_nid)
        pool.remove(chosen_nid)

    if end_id is not None:
        eid = int(end_id)
        if eid in order:
            order.remove(eid)
        order.append(eid)

    return order


# =========================================================
# Internal: ALNS solve (FULL) with cache + (MISS only) best-of-k
# =========================================================
def _solve_alns_full_cached(
    payload: Dict[str, Any],
    *,
    matrix_key: str = "dist_matrix",
    start_id: Optional[int] = 0,
    end_id: Optional[int] = None,
    seed: Optional[int] = None,
    max_iters: Optional[int] = None,
    # cache/determinism
    use_cache: bool = True,
    cache_dir: str = DEFAULT_CACHE_DIR,
    matrix_unit: str = DEFAULT_MATRIX_UNIT,
    verbose: bool = True,
    # init params (키에 포함)
    init_trials: int = 20,
    init_rcl_size: int = 3,
    init_use_regret2: bool = True,
    init_use_relocate_ls: bool = True,
    init_relocate_rounds: int = 2,
    # SA scale (키에 포함)
    sa_start_mult: float = 5.0,
    sa_end_mult: float = 0.01,
    # =================================================
    # best-of-k (MISS only) options
    # =================================================
    miss_enable_bestof_k: bool = False,   # 기본 OFF
    miss_bestof_k: int = 8,
    miss_short_iters: int = 1500,
    miss_seed_stride: int = 10007,
    miss_refine: bool = True,
) -> Dict[str, Any]:
    address_list = _get_address_list(payload)
    mat_raw = _get_matrix(payload, matrix_key)

    # 결정성: 비용/비교 안정화 위해 정수화된 행렬로 수행
    mat = _quantize_matrix(mat_raw, unit=matrix_unit)

    id2idx, _ = _build_id_maps(address_list)

    if start_id is not None and int(start_id) not in id2idx:
        raise ValueError(f"start_id={start_id} not found in address_list ids.")
    if end_id is not None and int(end_id) not in id2idx:
        raise ValueError(f"end_id={end_id} not found in address_list ids.")

    max_iters_eff = int(max_iters) if max_iters is not None else 2000
    node_ids = [int(r["id"]) for r in address_list]

    # opts는 "결과에 영향 주는 것"을 모두 담기 (캐시 키 구성요소)
    opts: Dict[str, Any] = {
        "matrix_key": matrix_key,
        "matrix_unit": matrix_unit,
        "max_iters": max_iters_eff,
        "init_trials": int(init_trials),
        "init_rcl_size": int(init_rcl_size),
        "init_use_regret2": bool(init_use_regret2),
        "init_use_relocate_ls": bool(init_use_relocate_ls),
        "init_relocate_rounds": int(init_relocate_rounds),
        "sa_start_mult": float(sa_start_mult),
        "sa_end_mult": float(sa_end_mult),
        "miss_enable_bestof_k": bool(miss_enable_bestof_k),
        "miss_bestof_k": int(miss_bestof_k),
        "miss_short_iters": int(miss_short_iters),
        "miss_seed_stride": int(miss_seed_stride),
        "miss_refine": bool(miss_refine),
    }

    if seed is not None:
        opts["seed_override"] = int(seed)

    key_hex = _problem_key(mat, node_ids=node_ids, start_id=start_id, end_id=end_id, opts=opts)

    cache_miss = False
    if use_cache and cache_dir:
        cached = _load_cache(cache_dir, key_hex)
        if cached is not None:
            cached["cache_hit"] = True
            if verbose:
                print(f"[ALNS] dist_matrix unchanged → cache HIT (key={key_hex[:16]})")
            return cached
        cache_miss = True
        if verbose:
            print(f"[ALNS] cache MISS (key={key_hex[:16]})")

    seed_eff = int(seed) if seed is not None else _seed_from_key(key_hex)

    def _run_once(seed_run: int, iters_run: int, tag: str = "") -> Dict[str, Any]:
        _set_deterministic(seed_run)
        rnd = random.Random(seed_run)

        init_ids = build_init_ids_multi_start(
            address_list,
            mat,
            rnd,
            start_id=start_id,
            end_id=end_id,
            trials=init_trials,
            rcl_size=init_rcl_size,
            use_regret2=init_use_regret2,
            use_relocate_ls=init_use_relocate_ls,
            relocate_rounds=init_relocate_rounds,
        )
        init_cost = _calc_cost(init_ids, id2idx, mat)
        init_state = _freeze(init_ids, id2idx, mat)

        t0 = time.perf_counter()

        alns = ALNS(rnd)
        alns.add_destroy_operator(lambda s, r: destroy_random(s, r, start_id, end_id))
        alns.add_destroy_operator(lambda s, r: destroy_worst(s, r, start_id, end_id, id2idx, mat))
        alns.add_destroy_operator(lambda s, r: destroy_segment(s, r, start_id, end_id))

        def _repair_greedy_wrapper(destroyed, r):
            partial, removed = destroyed
            new_order = repair_greedy(partial, removed, r, start_id, end_id, id2idx, mat)
            return _freeze(new_order, id2idx, mat)

        def _repair_regret2_wrapper(destroyed, r):
            partial, removed = destroyed
            new_order = repair_regret2(partial, removed, r, start_id, end_id, id2idx, mat)
            return _freeze(new_order, id2idx, mat)

        alns.add_repair_operator(_repair_greedy_wrapper)
        alns.add_repair_operator(_repair_regret2_wrapper)

        # SA temperature scaling
        n_edges = max(1, len(init_state.order) - 1)
        avg_edge = init_state.cost / n_edges

        start_temp = max(1.0, avg_edge * sa_start_mult)
        end_temp = max(1e-3, avg_edge * sa_end_mult)
        step = (end_temp / start_temp) ** (1.0 / max(1, iters_run))

        accept = SimulatedAnnealing(
            start_temperature=start_temp,
            end_temperature=end_temp,
            step=step,
        )
        stop = MaxIterations(int(iters_run))

        op_select = RandomOpSelectCompat(
            num_destroy=len(alns.destroy_operators),
            num_repair=len(alns.repair_operators),
        )

        result = alns.iterate(init_state, op_select, accept, stop)
        best_order = list(result.best_state.order)
        best_cost = _calc_cost(best_order, id2idx, mat)

        runtime_ms = (time.perf_counter() - t0) * 1000.0

        if verbose and tag:
            improve_abs = init_cost - best_cost
            improve_pct = (improve_abs / init_cost * 100.0) if init_cost > 0 else 0.0
            print(
                f"[ALNS]{tag} seed={seed_run} iters={iters_run} "
                f"init={init_cost:.3f} best={best_cost:.3f} improve={improve_pct:.2f}% "
                f"runtime={runtime_ms:.1f}ms"
            )

        return {
            "seed": int(seed_run),
            "iters": int(iters_run),
            "init_order": [int(x) for x in init_ids],
            "best_order": [int(x) for x in best_order],
            "init_cost": float(init_cost),
            "best_cost": float(best_cost),
            "runtime_ms": float(runtime_ms),
        }

    # MISS에서만 best-of-k → refine
    do_bestofk = bool(miss_enable_bestof_k) and bool(cache_miss)
    picked: Optional[Dict[str, Any]] = None
    bestofk_trials: List[Dict[str, Any]] = []

    if do_bestofk:
        k = max(1, int(miss_bestof_k))
        short_iters = max(1, int(min(max_iters_eff, int(miss_short_iters))))

        if verbose:
            print(f"[ALNS] best-of-k enabled on MISS: k={k}, short_iters={short_iters}, refine={miss_refine}")

        for i in range(k):
            s = (seed_eff + i * int(miss_seed_stride)) & 0xFFFFFFFF
            trial = _run_once(int(s), int(short_iters), tag=f"[trial {i+1}/{k}]")
            bestofk_trials.append(
                {"seed": trial["seed"], "best_cost": trial["best_cost"], "runtime_ms": trial["runtime_ms"]}
            )
            if picked is None:
                picked = trial
            else:
                if (trial["best_cost"] < picked["best_cost"] - 1e-12) or (
                    abs(trial["best_cost"] - picked["best_cost"]) <= 1e-12 and trial["seed"] < picked["seed"]
                ):
                    picked = trial

        picked_seed = int(picked["seed"])
        if bool(miss_refine) and max_iters_eff > short_iters:
            final_run = _run_once(picked_seed, int(max_iters_eff), tag="[refine]")
        else:
            final_run = picked

        seed_final = int(final_run["seed"])
        init_ids = final_run["init_order"]
        best_order = final_run["best_order"]
        init_cost = float(final_run["init_cost"])
        best_cost = float(final_run["best_cost"])
        runtime_ms = float(final_run["runtime_ms"])

        bestofk_meta = {
            "enabled": True,
            "k": int(k),
            "short_iters": int(short_iters),
            "picked_seed": int(picked_seed),
            "trials": bestofk_trials,
        }
    else:
        one = _run_once(int(seed_eff), int(max_iters_eff), tag="[single]")
        seed_final = int(one["seed"])
        init_ids = one["init_order"]
        best_order = one["best_order"]
        init_cost = float(one["init_cost"])
        best_cost = float(one["best_cost"])
        runtime_ms = float(one["runtime_ms"])
        bestofk_meta = {"enabled": False}

    out = {
        "key": key_hex[:16],
        "key_full": key_hex,
        "seed": int(seed_final),
        "cache_hit": False,
        "n_nodes": int(len(address_list)),
        "max_iters": int(max_iters_eff),
        "opts": opts,
        "init_order": [int(x) for x in init_ids],
        "best_order": [int(x) for x in best_order],
        "init_cost": float(init_cost),
        "best_cost": float(best_cost),
        "runtime_ms": float(runtime_ms),
        "bestofk": bestofk_meta,  # 디버깅용
    }

    if use_cache and cache_dir:
        _save_cache(cache_dir, key_hex, out)

    return out


# =========================
# Internal: ALNS solve -> best id order (NO grouping)
# =========================
def _solve_alns_ids(
    payload: Dict[str, Any],
    *,
    matrix_key: str = "dist_matrix",
    start_id: Optional[int] = 0,
    end_id: Optional[int] = None,
    seed: Optional[int] = None,
    max_iters: int = None,
    use_cache: bool = True,
    cache_dir: str = DEFAULT_CACHE_DIR,
    matrix_unit: str = DEFAULT_MATRIX_UNIT,
    # pass-through
    miss_enable_bestof_k: bool = False,
    miss_bestof_k: int = 8,
    miss_short_iters: int = 1500,
    miss_seed_stride: int = 10007,
    miss_refine: bool = True,
) -> List[int]:
    res = _solve_alns_full_cached(
        payload,
        matrix_key=matrix_key,
        start_id=start_id,
        end_id=end_id,
        seed=seed,
        max_iters=max_iters,
        use_cache=use_cache,
        cache_dir=cache_dir,
        matrix_unit=matrix_unit,
        miss_enable_bestof_k=miss_enable_bestof_k,
        miss_bestof_k=miss_bestof_k,
        miss_short_iters=miss_short_iters,
        miss_seed_stride=miss_seed_stride,
        miss_refine=miss_refine,
    )
    return list(res["best_order"])


# =========================
# Post-grouping (AFTER ALNS): make same ordering for same group
#  - start_id, end_id 둘 다 "보호"해서
#    (1) 그룹 union에서 제외
#    (2) start 그룹은 맨 앞, end 그룹은 맨 뒤로 정렬
# =========================
def _post_group_ordering_and_suborder(
    payload: Dict[str, Any],
    ordered_ids: List[int],
    *,
    start_id: Optional[int] = 0,
    end_id: Optional[int] = None,  # ✅ NEW
    round_ndigits: int = 6,
    inner_sort_key: str = "tracking_number",
    enable_same_coords: bool = True,
    enable_same_road_addr2: bool = True,
    enable_apartment_road: bool = True,
    enable_same_road: bool = True,
    sub_order_start: int = 1,
) -> pd.DataFrame:
    address_list = _get_address_list(payload)
    _, id2rec = _build_id_maps(address_list)

    start_id_int = int(start_id) if start_id is not None else None
    end_id_int = int(end_id) if end_id is not None else None

    def _norm_ws(s: Any) -> str:
        s = str(s or "").strip()
        s = re.sub(r"\s+", " ", s).strip()
        return s

    def _norm_road(s: Any) -> str:
        s = _norm_ws(s)
        s = re.sub(r"\([^)]*\)", " ", s)
        s = re.sub(r"\s+", " ", s).strip()
        return s

    def _coord_key(rec: Dict[str, Any]) -> Tuple[float, float]:
        return (round(float(rec["lat"]), round_ndigits), round(float(rec["lng"]), round_ndigits))

    def _road_addr2_key(rec: Dict[str, Any]) -> str:
        road = _norm_road(rec.get("address_road"))
        addr2 = _norm_ws(rec.get("address2"))
        if not road or not addr2:
            return ""
        return f"{road}||{addr2}"

    def _is_protected(nid: int) -> bool:
        if start_id_int is not None and int(nid) == start_id_int:
            return True
        if end_id_int is not None and int(nid) == end_id_int:
            return True
        return False

    def _is_unit_id(nid: int) -> bool:
        return (start_id_int is not None) and (int(nid) == start_id_int)

    # base ordering (ALNS order 그대로)
    base_ordering: Dict[int, int] = {}
    counter = 1
    for nid in ordered_ids:
        nid = int(nid)
        if _is_unit_id(nid):
            base_ordering[nid] = 0
        else:
            base_ordering[nid] = counter
            counter += 1

    all_ids = [int(x) for x in ordered_ids]
    idx_of_id = {nid: i for i, nid in enumerate(all_ids)}
    parent = list(range(len(all_ids)))

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a: int, b: int):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra

    # ---- union: protected(start/end)는 아예 그룹에 묶지 않음 ----
    if enable_same_coords:
        coord_map: Dict[Tuple[float, float], int] = {}
        for i, nid in enumerate(all_ids):
            if _is_protected(nid):
                continue
            rec = id2rec[nid]
            ck = _coord_key(rec)
            if ck in coord_map:
                union(i, coord_map[ck])
            else:
                coord_map[ck] = i

    if enable_same_road_addr2:
        addr_map: Dict[str, int] = {}
        for i, nid in enumerate(all_ids):
            if _is_protected(nid):
                continue
            rec = id2rec[nid]
            ak = _road_addr2_key(rec)
            if not ak:
                continue
            if ak in addr_map:
                union(i, addr_map[ak])
            else:
                addr_map[ak] = i

    if enable_apartment_road:
        apt_road_map: Dict[str, int] = {}
        for i, nid in enumerate(all_ids):
            if _is_protected(nid):
                continue
            rec = id2rec[nid]
            road = _norm_road(rec.get("address_road"))
            if not road:
                continue
            apt_flag = int(rec.get("apartment_flag") or 0)
            if apt_flag != 1:
                continue
            k = f"APT::{road}"
            if k in apt_road_map:
                union(i, apt_road_map[k])
            else:
                apt_road_map[k] = i

    if enable_same_road:
        road_map: Dict[str, int] = {}
        for i, nid in enumerate(all_ids):
            if _is_protected(nid):
                continue
            rec = id2rec[nid]
            road = _norm_road(rec.get("address_road"))
            if not road:
                continue
            k = f"ROAD::{road}"
            if k in road_map:
                union(i, road_map[k])
            else:
                road_map[k] = i

    # (기존) unit과 같은 root로 잘못 묶이면 분리해주는 안전장치 유지
    if start_id_int is not None and start_id_int in idx_of_id:
        unit_i = idx_of_id[start_id_int]
        unit_root = find(unit_i)
        for i, nid in enumerate(all_ids):
            if i == unit_i:
                continue
            if find(i) == unit_root:
                parent[i] = i

    # end도 혹시 동일 root로 억지로 섞인 경우 대비 (이미 protected라 거의 안 생김)
    if end_id_int is not None and end_id_int in idx_of_id:
        end_i = idx_of_id[end_id_int]
        end_root = find(end_i)
        for i, nid in enumerate(all_ids):
            if i == end_i:
                continue
            if find(i) == end_root and nid != end_id_int:
                parent[i] = i

    root_to_members: Dict[int, List[int]] = {}
    for i, nid in enumerate(all_ids):
        root_to_members.setdefault(find(i), []).append(nid)

    group_min_order: Dict[int, int] = {}
    for root, members in root_to_members.items():
        non_unit_orders = [base_ordering.get(int(mid), 0) for mid in members if base_ordering.get(int(mid), 0) != 0]
        if not non_unit_orders:
            group_min_order[root] = 0
        else:
            group_min_order[root] = min(non_unit_orders)

    unified_ordering_raw: Dict[int, int] = {}
    for root, members in root_to_members.items():
        o = group_min_order[root]
        for mid in members:
            unified_ordering_raw[mid] = o

    uniq = sorted({o for o in unified_ordering_raw.values() if o != 0})
    compress = {o: i + 1 for i, o in enumerate(uniq)}
    unified_ordering: Dict[int, int] = {}
    for mid, o in unified_ordering_raw.items():
        unified_ordering[mid] = 0 if o == 0 else compress[o]

    rows: List[Dict[str, Any]] = []

    def _root_sort_key(root: int):
        members = root_to_members[root]
        # start 그룹은 맨 앞
        if start_id_int is not None and start_id_int in members:
            return (0, 0)
        # end 그룹은 맨 뒤
        if end_id_int is not None and end_id_int in members:
            return (2, 0)
        # 나머지는 기존 ordering 기준
        o = group_min_order[root]
        if o == 0:
            return (1, 0)
        return (1, o)

    ordered_roots = sorted(root_to_members.keys(), key=_root_sort_key)

    for root in ordered_roots:
        members = root_to_members[root]

        unit_members = [mid for mid in members if base_ordering.get(mid, -1) == 0]
        non_unit_members = [mid for mid in members if base_ordering.get(mid, -1) != 0]

        # unit(start) row (ordering=0)
        for mid in unit_members:
            rec = id2rec[mid]
            rows.append(
                {
                    "id": int(rec.get("id")),
                    "Area": _get_area(rec),
                    "tracking_number": rec.get("tracking_number"),
                    "address_road": rec.get("address_road"),
                    "address2": rec.get("address2"),
                    "lat": float(rec.get("lat")),
                    "lng": float(rec.get("lng")),
                    "apartment_flag": int(rec.get("apartment_flag") or 0),
                    "ordering": 0,
                    "sub_order": 0,
                }
            )

        # 그룹 내부 정렬: 기본은 tracking_number, BUT end는 항상 그룹 내 마지막
        def _member_sort_key(mid: int):
            if end_id_int is not None and int(mid) == end_id_int:
                return (1, "", int(mid))
            return (0, str(id2rec[mid].get(inner_sort_key, "")), int(mid))

        non_unit_sorted = sorted(non_unit_members, key=_member_sort_key)

        sub = sub_order_start
        for mid in non_unit_sorted:
            rec = id2rec[mid]
            rows.append(
                {
                    "id": int(rec.get("id")),
                    "Area": _get_area(rec),
                    "tracking_number": rec.get("tracking_number"),
                    "address_road": rec.get("address_road"),
                    "address2": rec.get("address2"),
                    "lat": float(rec.get("lat")),
                    "lng": float(rec.get("lng")),
                    "apartment_flag": int(rec.get("apartment_flag") or 0),
                    "ordering": unified_ordering[mid],
                    "sub_order": sub,
                }
            )
            sub += 1

    return pd.DataFrame(rows)


# =========================
# Public API
# =========================
def solve_alns_to_df_later_supernode(
    payload: Dict[str, Any],
    *,
    matrix_key: str = "dist_matrix",
    start_id: Optional[int] = 1,
    end_id: Optional[int] = None,

    # tracking_number 선택
    selected_start_tracking_number: Optional[str] = None,
    selected_end_tracking_number: Optional[str] = None,

    # id 선택 (옵션)
    selected_start_id: Optional[int] = None,
    selected_end_id: Optional[int] = None,

    # MISS best-of-k
    miss_enable_bestof_k: bool = False,
    miss_bestof_k: int = 8,
    miss_short_iters: int = 1500,
    miss_seed_stride: int = 10007,
    miss_refine: bool = True,

    seed: Optional[int] = None,
    max_iters: int = None,
    use_cache: bool = True,
    cache_dir: str = DEFAULT_CACHE_DIR,
    matrix_unit: str = DEFAULT_MATRIX_UNIT,
    group_same_coords: bool = True,
    group_same_road_addr2: bool = True,
    group_same_road_apartment: bool = True,
    group_same_road: bool = True,
    round_ndigits: int = 6,
    inner_sort_key: str = "tracking_number",
) -> pd.DataFrame:
    """
    1) start(=unit) id -> 0 리라벨
    2) 전체 노드로 ALNS 실행하여 순서 획득 (start/end 제약 반영)
    3) ALNS 결과를 후처리 그룹핑으로 ordering만 동일하게 묶음
       - start/end 노드는 그룹핑에서 보호 (ordering/끝 보장)
    4) 그룹 내부 sub_order 부여
    """

    # tracking_number -> id
    start_id_from_tn = _tracking_number_to_id(payload, selected_start_tracking_number)
    end_id_from_tn = _tracking_number_to_id(payload, selected_end_tracking_number)

    # 우선순위: tn > selected_id > 기본 start_id/end_id
    start_old = start_id_from_tn if start_id_from_tn is not None else (selected_start_id if selected_start_id is not None else start_id)
    end_old = end_id_from_tn if end_id_from_tn is not None else (selected_end_id if selected_end_id is not None else end_id)

    payload_eff = payload
    start_eff = start_old
    end_eff = end_old

    if start_old is not None and end_old is not None and int(start_old) == int(end_old):
        raise ValueError("start and end cannot be the same node.")

    # start가 있으면 remap: start_old -> 0
    if start_eff is not None:
        payload_eff, new_to_old = _remap_unit_to_zero(payload_eff, unit_original_id=int(start_eff), matrix_key=matrix_key)
        old_to_new = {int(old): int(new) for new, old in new_to_old.items()}
        start_eff = 0

        # end도 old id 기준이므로 new id로 변환
        if end_eff is not None:
            if int(end_eff) not in old_to_new:
                raise ValueError(f"end_id={end_eff} not found in payload ids.")
            end_eff = old_to_new[int(end_eff)]

    ordered_ids = _solve_alns_ids(
        payload_eff,
        matrix_key=matrix_key,
        start_id=start_eff,
        end_id=end_eff,
        seed=seed,
        max_iters=max_iters,
        use_cache=use_cache,
        cache_dir=cache_dir,
        matrix_unit=matrix_unit,
        miss_enable_bestof_k=miss_enable_bestof_k,
        miss_bestof_k=miss_bestof_k,
        miss_short_iters=miss_short_iters,
        miss_seed_stride=miss_seed_stride,
        miss_refine=miss_refine,
    )

    # custom start 여부 (UI에서 start를 따로 선택하면 df에서 ordering=0(유닛) 숨기고 싶었던 정책 유지)
    is_custom_start = (start_id_from_tn is not None) or (selected_start_id is not None)

    df = _post_group_ordering_and_suborder(
        payload_eff,
        ordered_ids,
        start_id=(start_eff if not is_custom_start else None),
        end_id=end_eff,  # ✅ NEW: end도 보호/끝 보장
        round_ndigits=round_ndigits,
        inner_sort_key=inner_sort_key,
        enable_same_coords=bool(group_same_coords),
        enable_same_road_addr2=bool(group_same_road_addr2),
        enable_apartment_road=bool(group_same_road_apartment),
        enable_same_road=bool(group_same_road),
        sub_order_start=1,
    )

    df = df.sort_values(["ordering", "sub_order"], kind="stable").reset_index(drop=True)
    return df


# =========================
# Evaluation: ALNS metrics (Route Cost / runtime / improvement)
#  - solve_alns_to_df와 옵션이 같아야 비교가 정확하니 miss_*도 pass-through
# =========================
def _compute_route_cost_from_payload(
    payload: Dict[str, Any],
    order: List[int],
    *,
    matrix_key: str,
) -> float:
    address_list = _get_address_list(payload)
    mat = _get_matrix(payload, matrix_key)
    id2idx, _ = _build_id_maps(address_list)
    return _calc_cost(order, id2idx, mat)


def eval_alns_metrics(
    payload: Dict[str, Any],
    *,
    matrix_key: str = "dist_matrix",
    start_id: Optional[int] = 1,
    end_id: Optional[int] = None,
    seed: Optional[int] = None,
    max_iters: int = None,
    use_cache: bool = True,
    cache_dir: str = DEFAULT_CACHE_DIR,
    matrix_unit: str = DEFAULT_MATRIX_UNIT,
    # ✅ NEW: best-of-k 옵션도 동일하게
    miss_enable_bestof_k: bool = False,
    miss_bestof_k: int = 8,
    miss_short_iters: int = 1500,
    miss_seed_stride: int = 10007,
    miss_refine: bool = True,
) -> Dict[str, Any]:
    start_id_eff = start_id
    end_id_eff = end_id
    payload_eff = payload

    if start_id_eff is not None:
        payload_eff, new_to_old = _remap_unit_to_zero(payload_eff, unit_original_id=int(start_id_eff), matrix_key=matrix_key)
        old_to_new = {int(old): int(new) for new, old in new_to_old.items()}
        start_id_eff = 0

        if end_id_eff is not None:
            if int(end_id_eff) not in old_to_new:
                raise ValueError(f"end_id={end_id_eff} not found in payload ids.")
            end_id_eff = old_to_new[int(end_id_eff)]

    res = _solve_alns_full_cached(
        payload_eff,
        matrix_key=matrix_key,
        start_id=start_id_eff,
        end_id=end_id_eff,
        seed=seed,
        max_iters=max_iters,
        use_cache=use_cache,
        cache_dir=cache_dir,
        matrix_unit=matrix_unit,
        miss_enable_bestof_k=miss_enable_bestof_k,
        miss_bestof_k=miss_bestof_k,
        miss_short_iters=miss_short_iters,
        miss_seed_stride=miss_seed_stride,
        miss_refine=miss_refine,
    )

    best_order = [int(x) for x in res["best_order"]]
    init_order = [int(x) for x in res["init_order"]]

    start_ok = True if start_id_eff is None else (len(best_order) > 0 and int(best_order[0]) == int(start_id_eff))
    end_ok = True if end_id_eff is None else (len(best_order) > 0 and int(best_order[-1]) == int(end_id_eff))

    init_cost = float(res["init_cost"])
    best_cost = float(res["best_cost"])

    improve_abs = float(init_cost - best_cost)
    improve_pct = float((improve_abs / init_cost) * 100.0) if init_cost > 0 else 0.0

    return {
        "n_nodes": int(res["n_nodes"]),
        "seed": int(res["seed"]),
        "key": str(res.get("key", "")),
        "cache_hit": bool(res.get("cache_hit", False)),
        "max_iters": int(res["max_iters"]),
        "runtime_ms": float(res["runtime_ms"]),
        "init_cost": float(init_cost),
        "best_cost": float(best_cost),
        "improve_abs": float(improve_abs),
        "improve_pct": float(improve_pct),
        "start_ok": bool(start_ok),
        "end_ok": bool(end_ok),
        "init_order": init_order,
        "best_order": best_order,
        "bestofk": res.get("bestofk", {"enabled": False}),
    }


def eval_alns_metrics_batch(
    payloads: List[Dict[str, Any]],
    *,
    matrix_key: str = "dist_matrix",
    start_id: Optional[int] = 1,
    end_id: Optional[int] = None,
    seed: Optional[int] = None,
    max_iters: int = None,
    use_cache: bool = True,
    cache_dir: str = DEFAULT_CACHE_DIR,
    matrix_unit: str = DEFAULT_MATRIX_UNIT,
    # ✅ NEW
    miss_enable_bestof_k: bool = False,
    miss_bestof_k: int = 8,
    miss_short_iters: int = 1500,
    miss_seed_stride: int = 10007,
    miss_refine: bool = True,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    rows = []
    for i, payload in enumerate(payloads):
        m = eval_alns_metrics(
            payload,
            matrix_key=matrix_key,
            start_id=start_id,
            end_id=end_id,
            seed=seed,
            max_iters=max_iters,
            use_cache=use_cache,
            cache_dir=cache_dir,
            matrix_unit=matrix_unit,
            miss_enable_bestof_k=miss_enable_bestof_k,
            miss_bestof_k=miss_bestof_k,
            miss_short_iters=miss_short_iters,
            miss_seed_stride=miss_seed_stride,
            miss_refine=miss_refine,
        )
        m["case_idx"] = i
        rows.append(m)

    df = pd.DataFrame(rows)

    def _pctl(s: pd.Series, q: float) -> float:
        if len(s) == 0:
            return 0.0
        return float(np.percentile(s.astype(float).to_numpy(), q))

    summary = {
        "n_cases": int(len(df)),
        "max_iters": int(df["max_iters"].iloc[0]) if len(df) else int(max_iters or 0),
        "best_cost_mean": float(df["best_cost"].mean()) if len(df) else 0.0,
        "best_cost_median": float(df["best_cost"].median()) if len(df) else 0.0,
        "best_cost_p95": _pctl(df["best_cost"], 95),
        "improve_pct_mean": float(df["improve_pct"].mean()) if len(df) else 0.0,
        "improve_pct_median": float(df["improve_pct"].median()) if len(df) else 0.0,
        "improve_pct_p95": _pctl(df["improve_pct"], 95),
        "runtime_ms_mean": float(df["runtime_ms"].mean()) if len(df) else 0.0,
        "runtime_ms_p95": _pctl(df["runtime_ms"], 95),
        "start_ok_rate": float(df["start_ok"].mean()) if len(df) else 0.0,
        "end_ok_rate": float(df["end_ok"].mean()) if len(df) else 0.0,
        "cache_hit_rate": float(df["cache_hit"].mean()) if len(df) else 0.0,
        "bestofk_enabled_rate": float(df["bestofk"].apply(lambda x: bool(x.get("enabled", False))).mean())
        if len(df)
        else 0.0,
    }

    return df, summary