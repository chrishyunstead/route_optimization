# alns_pipeline.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Any
import random
import numpy as np
import pandas as pd
import hashlib
import re

from alns import ALNS
from alns.accept import SimulatedAnnealing
from alns.stop import MaxIterations


# =========================
# Helpers (payload / matrix / id map)
# =========================
def _get_address_list(payload: Dict[str, Any]) -> List[Dict[str, Any]]:
    if "address_list" in payload and payload["address_list"]:
        return payload["address_list"]
    if "address_geocode_list" in payload and payload["address_geocode_list"]:
        return payload["address_geocode_list"]
    raise KeyError("payload must contain 'address_list' or 'address_geocode_list'.")


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
    동일 입력(payload 주소들)이면 항상 동일 seed.
    """
    addr = _get_address_list(payload)
    parts = []
    for r in sorted(addr, key=lambda x: int(x["id"])):
        parts.append(
            f'{int(r["id"])}|{r.get("tracking_number","")}|{r.get("lat","")}|{r.get("lng","")}'
        )
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


def _find_super_start_id(super_id_to_member_ids: Dict[int, List[int]], start_id: Optional[int]) -> Optional[int]:
    if start_id is None:
        return None
    sid = int(start_id)
    for super_id, members in super_id_to_member_ids.items():
        if sid in members:
            return int(super_id)
    raise ValueError(f"start_id={sid} not found in any super node members.")


# =========================
# Super node grouping (DSU / OR conditions)
# =========================
def _group_supernodes_payload(
    payload: Dict[str, Any],
    *,
    matrix_key: str = "dist_matrix",
    round_ndigits: int = 6,
    inner_sort_key: str = "tracking_number",
    start_id: Optional[int] = 0,
    # OR 조건 토글
    enable_same_coords: bool = True,
    enable_same_road_addr2: bool = True,
    enable_apartment_road: bool = True,  # group_same_road 옵션 켜면 True
) -> Tuple[Dict[str, Any], Dict[int, List[int]]]:
    """
    super node 그룹화 (OR 조건, DSU):

    기본(항상 권장):
    1) (lat,lng) 동일 (round_ndigits)
    2) (address_road, address2) 동일 (정규화 후)  - road/addr2 둘 다 있어야 적용

    옵션:
    3) apartment_flag == 1 이면서 address_road 동일 (정규화 후)

    ✅ 중요한 규칙:
    - unit(start_id)은 어떤 union에도 포함하지 않고 "항상 단독"으로 둔다.
      (간접적으로라도 섞일 여지를 원천 차단)
    """
    address_list = _get_address_list(payload)
    mat = _get_matrix(payload, matrix_key)
    id2idx, _ = _build_id_maps(address_list)

    start_id_int = int(start_id) if start_id is not None else None

    def _coord_key(rec: Dict[str, Any]) -> Tuple[float, float]:
        return (round(float(rec["lat"]), round_ndigits), round(float(rec["lng"]), round_ndigits))

    def _norm_ws(s: Any) -> str:
        s = str(s or "").strip()
        s = re.sub(r"\s+", " ", s).strip()
        return s

    def _norm_road(s: Any) -> str:
        s = _norm_ws(s)
        s = re.sub(r"\([^)]*\)", " ", s)
        s = re.sub(r"\s+", " ", s).strip()
        return s

    def _road_addr2_key(rec: Dict[str, Any]) -> str:
        road = _norm_road(rec.get("address_road"))
        addr2 = _norm_ws(rec.get("address2"))
        if not road or not addr2:
            return ""
        return f"{road}||{addr2}"

    # ----- DSU -----
    n = len(address_list)
    parent = list(range(n))

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a: int, b: int):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra

    def _is_unit(rec: Dict[str, Any]) -> bool:
        return (start_id_int is not None) and (int(rec["id"]) == start_id_int)

    # 1) same coords union
    if enable_same_coords:
        coord_map: Dict[Tuple[float, float], int] = {}
        for i, rec in enumerate(address_list):
            if _is_unit(rec):
                continue
            ck = _coord_key(rec)
            if ck in coord_map:
                union(i, coord_map[ck])
            else:
                coord_map[ck] = i

    # 2) same (road, addr2) union
    if enable_same_road_addr2:
        addr_map: Dict[str, int] = {}
        for i, rec in enumerate(address_list):
            if _is_unit(rec):
                continue
            ak = _road_addr2_key(rec)
            if not ak:
                continue
            if ak in addr_map:
                union(i, addr_map[ak])
            else:
                addr_map[ak] = i

    # 3) apartment_flag==1 & same road union
    if enable_apartment_road:
        apt_road_map: Dict[str, int] = {}
        for i, rec in enumerate(address_list):
            if _is_unit(rec):
                continue

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

    # ✅ unit이 혹시라도 섞일 여지를 완전 차단(방어)
    # - unit 인덱스를 찾아서, unit 루트와 같은 루트로 들어간 노드가 있으면 강제로 분리
    #   (정상적으로는 발생하지 않지만, 데이터 이상/향후 코드 변경 대비)
    if start_id_int is not None:
        unit_idx = None
        for i, rec in enumerate(address_list):
            if int(rec["id"]) == start_id_int:
                unit_idx = i
                break
        if unit_idx is not None:
            unit_root = find(unit_idx)
            for i, rec in enumerate(address_list):
                if i == unit_idx:
                    continue
                if find(i) == unit_root:
                    parent[i] = i  # 강제 분리

    # 루트별 멤버 수집
    root_to_members: Dict[int, List[Dict[str, Any]]] = {}
    for i, rec in enumerate(address_list):
        root = find(i)
        root_to_members.setdefault(root, []).append(rec)

    # 안정적 순서 정렬 키
    def _group_sort_key(members: List[Dict[str, Any]]):
        # unit 그룹 최우선
        if start_id_int is not None and any(int(r["id"]) == start_id_int for r in members):
            return (0, "UNIT", 0)

        rep = sorted(members, key=lambda r: int(r["id"]))[0]
        ak = _road_addr2_key(rep)
        ck = _coord_key(rep)
        base = ak if ak else f"COORD::{ck[0]},{ck[1]}"
        min_id = min(int(r["id"]) for r in members)
        return (1, base, min_id)

    grouped_items = sorted(root_to_members.items(), key=lambda kv: _group_sort_key(kv[1]))

    super_nodes: List[Dict[str, Any]] = []
    super_id_to_members: Dict[int, List[int]] = {}

    for super_id, (_root, members) in enumerate(grouped_items, start=1):
        members_sorted = sorted(members, key=lambda r: (str(r.get(inner_sort_key, "")), int(r["id"])))
        member_ids = [int(r["id"]) for r in members_sorted]
        super_id_to_members[super_id] = member_ids

        rep = members_sorted[0]
        super_nodes.append(
            {
                "id": super_id,
                "lat": float(rep["lat"]),
                "lng": float(rep["lng"]),
                "tracking_number": str(rep.get("tracking_number", "")),
                "Area": _get_area(rep),
                "member_count": len(member_ids),
                "member_ids": member_ids,
                # 디버깅 편하게(필요 없으면 빼도 됨)
                "address_road": rep.get("address_road"),
                "address2": rep.get("address2"),
                "apartment_flag": int(rep.get("apartment_flag") or 0),
            }
        )

    # 축소 매트릭스
    K = len(super_nodes)
    grouped_mat = np.zeros((K, K), dtype=float)
    for i in range(1, K + 1):
        for j in range(1, K + 1):
            if i == j:
                grouped_mat[i - 1, j - 1] = 0.0
            else:
                a_id = super_id_to_members[i][0]
                b_id = super_id_to_members[j][0]
                grouped_mat[i - 1, j - 1] = mat[id2idx[a_id], id2idx[b_id]]

    grouped_payload = {
        "address_list": super_nodes,
        matrix_key: grouped_mat.tolist(),
        "meta": {
            "grouped": True,
            "grouped_by": "OR: (lat,lng) OR (address_road,address2) OR (apt==1 & address_road)",
            "round_ndigits": round_ndigits,
            "inner_sort_key": inner_sort_key,
            "n_original": len(address_list),
            "n_grouped": K,
            "enable_same_coords": enable_same_coords,
            "enable_same_road_addr2": enable_same_road_addr2,
            "enable_apartment_road": enable_apartment_road,
        },
    }
    return grouped_payload, super_id_to_members


# =========================
# Expand super order -> final DF (ordering + sub_order)
# =========================
def _expand_super_order_to_df(
    original_payload: Dict[str, Any],
    super_order_ids: List[int],
    super_id_to_member_ids: Dict[int, List[int]],
    *,
    original_start_id: Optional[int] = 0,
    same_order_for_same_group: bool = True,
    add_sub_order: bool = True,
    sub_order_start: int = 1,
) -> pd.DataFrame:
    """
    - ordering: 그룹 단위 순서 (unit은 0)
    - sub_order: 그룹 내부 순서 (unit은 0)
    """
    address_list = _get_address_list(original_payload)
    _, id2rec = _build_id_maps(address_list)

    start_id_int = int(original_start_id) if original_start_id is not None else None

    rows = []
    ordering_counter = 1

    for super_id in super_order_ids:
        member_ids = super_id_to_member_ids[super_id]

        has_non_unit = any(
            (start_id_int is None) or (int(mid) != start_id_int)
            for mid in member_ids
        )

        group_ordering = ordering_counter
        sub_counter = sub_order_start

        for mid in member_ids:
            mid_int = int(mid)
            rec = id2rec[mid_int]

            if start_id_int is not None and mid_int == start_id_int:
                ordering = 0
                sub_order = 0
            else:
                ordering = group_ordering if same_order_for_same_group else ordering_counter
                sub_order = sub_counter if add_sub_order else None

            row = {
                "Area": _get_area(rec),
                "tracking_number": rec.get("tracking_number"),
                "address_road": rec.get("address_road"),
                "address2": rec.get("address2"),
                "lat": float(rec.get("lat")),
                "lng": float(rec.get("lng")),
                "apartment_flag": int(rec.get("apartment_flag") or 0),
                "ordering": ordering,
            }
            if add_sub_order:
                row["sub_order"] = sub_order
            rows.append(row)

            if not (start_id_int is not None and mid_int == start_id_int):
                if add_sub_order:
                    sub_counter += 1

            if not same_order_for_same_group and ordering != 0:
                ordering_counter += 1

        if same_order_for_same_group:
            if has_non_unit:
                ordering_counter += 1

    return pd.DataFrame(rows)


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
# Deterministic init (NN + 2-opt)
# =========================
def solve_group_deterministic(
    address_list: List[Dict[str, Any]],
    matrix: np.ndarray,
    *,
    start_id: int | None = None,
    end_id: int | None = None,
) -> Tuple[List[int], float]:
    dist = matrix
    id2idx, _ = _build_id_maps(address_list)
    n = len(address_list)
    if n == 0:
        return [], 0.0
    if n == 1:
        return [int(address_list[0]["id"])], 0.0

    fixed_start = id2idx[int(start_id)] if start_id is not None else 0
    fixed_end = id2idx[int(end_id)] if end_id is not None else None

    unvisited = list(range(n))
    unvisited.remove(fixed_start)

    tour = [fixed_start]
    cur = fixed_start
    while unvisited:
        nxt = min(unvisited, key=lambda j: (dist[cur, j], j))
        tour.append(nxt)
        unvisited.remove(nxt)
        cur = nxt

    def route_cost(path_idx: List[int]) -> float:
        return float(sum(dist[path_idx[i], path_idx[i + 1]] for i in range(len(path_idx) - 1)))

    improved = True
    while improved:
        improved = False
        last_index = len(tour) - (1 if fixed_end is not None else 0)
        for i in range(0, last_index - 2):
            for j in range(i + 2, last_index):
                before = dist[tour[i], tour[i + 1]] + dist[tour[j], tour[j + 1 if j + 1 < len(tour) else j]]
                after = dist[tour[i], tour[j]] + dist[tour[i + 1], tour[j + 1 if j + 1 < len(tour) else j]]
                if after + 1e-12 < before:
                    tour[i + 1 : j + 1] = reversed(tour[i + 1 : j + 1])
                    improved = True

    if fixed_end is not None:
        if tour[-1] != fixed_end:
            tour.remove(fixed_end)
            tour.append(fixed_end)

    ordered_ids = [int(address_list[i]["id"]) for i in tour]
    return ordered_ids, route_cost(tour)


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
# Destroy / Repair
# =========================
def destroy_random(state: RouteState, rnd: random.Random, start_id: int | None, end_id: int | None) -> Tuple[List[int], List[int]]:
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

    # start/end 강제 고정
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


# =========================
# Internal: ALNS solve -> best id order
# =========================
def _solve_alns_ids(
    payload: Dict[str, Any],
    *,
    matrix_key: str = "dist_matrix",
    start_id: Optional[int] = 0,
    end_id: Optional[int] = None,
    seed: Optional[int] = None,
    max_iters: int = 1000,
) -> List[int]:
    address_list = _get_address_list(payload)
    mat = _get_matrix(payload, matrix_key)
    id2idx, _ = _build_id_maps(address_list)

    if start_id is not None and int(start_id) not in id2idx:
        raise ValueError(f"start_id={start_id} not found in address_list ids.")
    if end_id is not None and int(end_id) not in id2idx:
        raise ValueError(f"end_id={end_id} not found in address_list ids.")

    if seed is None:
        seed = _stable_seed_from_payload(payload, matrix_key)

    rnd = random.Random(seed)

    init_ids, _ = solve_group_deterministic(address_list, mat, start_id=start_id, end_id=end_id)
    init_state = _freeze(init_ids, id2idx, mat)

    alns = ALNS(rnd)
    alns.add_destroy_operator(lambda s, r: destroy_random(s, r, start_id, end_id))
    alns.add_destroy_operator(lambda s, r: destroy_worst(s, r, start_id, end_id, id2idx, mat))

    def _repair_wrapper(destroyed, r):
        partial, removed = destroyed
        new_order = repair_greedy(partial, removed, r, start_id, end_id, id2idx, mat)
        return _freeze(new_order, id2idx, mat)

    alns.add_repair_operator(_repair_wrapper)

    accept = SimulatedAnnealing(start_temperature=1.0, end_temperature=0.01, step=0.995)
    stop = MaxIterations(max_iters)

    op_select = RandomOpSelectCompat(
        num_destroy=len(alns.destroy_operators),
        num_repair=len(alns.repair_operators),
    )

    result = alns.iterate(init_state, op_select, accept, stop)
    return list(result.best_state.order)


# =========================
# Public API
# =========================
def solve_alns_to_df(
    payload: Dict[str, Any],
    *,
    matrix_key: str = "dist_matrix",
    start_id: Optional[int] = 1,
    end_id: Optional[int] = None,
    seed: Optional[int] = None, # 입력 기반 자동 seed
    max_iters: int = 1000, # 파괴(destroy)→복구(repair)→채택(accept) 반복 횟수
    # 그룹화 옵션
    group_same_coords: bool = True,
    group_same_road: bool = True,   # apartment_flag==1 & address_road 동일 그룹화까지 켜기
    round_ndigits: int = 6,
    inner_sort_key: str = "tracking_number",
) -> pd.DataFrame:
    """
    반환 DF 컬럼:
    Area, tracking_number, address_road, address2, lat, lng, apartment_flag, ordering, sub_order

    규칙:
    - unit은 id=0으로 재라벨링 후 ALNS start 고정
    - grouping은 OR 조건 DSU로 통합:
      (lat,lng 동일) OR (address_road,address2 동일) OR (apt==1 & road 동일[옵션])
    - ordering: 그룹 단위 순서 (unit=0)
    - sub_order: 그룹 내부 순서 (unit=0)
    """

    # 1) unit id -> 0
    if start_id is not None:
        payload, _new_to_old = _remap_unit_to_zero(
            payload,
            unit_original_id=start_id,
            matrix_key=matrix_key,
        )
        start_id = 0

    # 2) 그룹화 (통합 DSU)
    # ✅ (road,address2)는 기본 중복 제거로 항상 켠다(원래 니 요구사항)
    grouped_payload, super_map = _group_supernodes_payload(
        payload,
        matrix_key=matrix_key,
        round_ndigits=round_ndigits,
        inner_sort_key=inner_sort_key,
        start_id=start_id,
        enable_same_coords=bool(group_same_coords),
        enable_same_road_addr2=True,               # ✅ 항상 ON
        enable_apartment_road=bool(group_same_road),
    )

    super_start_id = _find_super_start_id(super_map, start_id) if start_id is not None else None

    super_order = _solve_alns_ids(
        grouped_payload,
        matrix_key=matrix_key,
        start_id=super_start_id,
        end_id=None,
        seed=seed,
        max_iters=max_iters,
    )

    return _expand_super_order_to_df(
        payload,
        super_order,
        super_map,
        original_start_id=start_id,
        same_order_for_same_group=True,
        add_sub_order=True,
        sub_order_start=1,
    )