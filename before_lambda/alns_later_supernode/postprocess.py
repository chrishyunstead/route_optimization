from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import re

import pandas as pd

from .payload import build_id_maps, get_address_list, get_area


def post_group_ordering_and_suborder(
    payload: Dict[str, Any],
    ordered_ids: List[int],
    *,
    start_id: Optional[int] = 0,
    end_id: Optional[int] = None,
    round_ndigits: int = 6,
    inner_sort_key: str = "tracking_number",
    enable_same_coords: bool = True,
    enable_same_road_addr2: bool = True,
    enable_apartment_road: bool = True,
    enable_same_road: bool = True,
    sub_order_start: int = 1,
) -> pd.DataFrame:
    """After-ALNS grouping.

    - ALNS is run on all nodes.
    - Post step merges nodes into groups (same coord / same road+addr2 / apt+road / same road).
    - All members in the same group share the same 'ordering'.
    - 'sub_order' gives an intra-group deterministic order.
    """

    address_list = get_address_list(payload)
    _, id2rec = build_id_maps(address_list)

    start_id_int = int(start_id) if start_id is not None else None
    end_id_int = int(end_id) if end_id is not None else None

    def norm_ws(s: Any) -> str:
        s = str(s or "").strip()
        s = re.sub(r"\s+", " ", s).strip()
        return s

    def norm_road(s: Any) -> str:
        s = norm_ws(s)
        s = re.sub(r"\([^)]*\)", " ", s)
        s = re.sub(r"\s+", " ", s).strip()
        return s

    def coord_key(rec: Dict[str, Any]) -> Tuple[float, float]:
        return (round(float(rec["lat"]), round_ndigits), round(float(rec["lng"]), round_ndigits))

    def road_addr2_key(rec: Dict[str, Any]) -> str:
        road = norm_road(rec.get("address_road"))
        addr2 = norm_ws(rec.get("address2"))
        if not road or not addr2:
            return ""
        return f"{road}||{addr2}"

    def is_unit(nid: int) -> bool:
        return start_id_int is not None and int(nid) == start_id_int

    def is_end(nid: int) -> bool:
        return end_id_int is not None and int(nid) == end_id_int

    # base ordering from ALNS sequence (unit -> 0, others -> 1..)
    base_order: Dict[int, int] = {}
    counter = 1
    for nid in ordered_ids:
        nid = int(nid)
        if is_unit(nid):
            base_order[nid] = 0
        else:
            base_order[nid] = counter
            counter += 1

    all_ids = [int(x) for x in ordered_ids]
    idx_of_id = {nid: i for i, nid in enumerate(all_ids)}

    # union-find over indices
    parent = list(range(len(all_ids)))

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a: int, b: int) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra

    if enable_same_coords:
        seen: Dict[Tuple[float, float], int] = {}
        for i, nid in enumerate(all_ids):
            if is_unit(nid) or is_end(nid):
                continue
            ck = coord_key(id2rec[nid])
            if ck in seen:
                union(i, seen[ck])
            else:
                seen[ck] = i

    if enable_same_road_addr2:
        seen: Dict[str, int] = {}
        for i, nid in enumerate(all_ids):
            if is_unit(nid) or is_end(nid):
                continue
            key = road_addr2_key(id2rec[nid])
            if not key:
                continue
            if key in seen:
                union(i, seen[key])
            else:
                seen[key] = i

    if enable_apartment_road:
        seen: Dict[str, int] = {}
        for i, nid in enumerate(all_ids):
            if is_unit(nid) or is_end(nid):
                continue
            rec = id2rec[nid]
            road = norm_road(rec.get("address_road"))
            if not road:
                continue
            if int(rec.get("apartment_flag") or 0) != 1:
                continue
            key = f"APT::{road}"
            if key in seen:
                union(i, seen[key])
            else:
                seen[key] = i

    if enable_same_road:
        seen: Dict[str, int] = {}
        for i, nid in enumerate(all_ids):
            if is_unit(nid) or is_end(nid):
                continue
            road = norm_road(id2rec[nid].get("address_road"))
            if not road:
                continue
            key = f"ROAD::{road}"
            if key in seen:
                union(i, seen[key])
            else:
                seen[key] = i

    # never merge unit with others
    if start_id_int is not None and start_id_int in idx_of_id:
        unit_i = idx_of_id[start_id_int]
        unit_root = find(unit_i)
        for i in range(len(all_ids)):
            if i != unit_i and find(i) == unit_root:
                parent[i] = i

    # groups
    root_to_members: Dict[int, List[int]] = {}
    for i, nid in enumerate(all_ids):
        root_to_members.setdefault(find(i), []).append(nid)

    group_min_order: Dict[int, int] = {}
    for root, members in root_to_members.items():
        non_unit = [base_order[m] for m in members if base_order[m] != 0]
        group_min_order[root] = min(non_unit) if non_unit else 0

    # unify ordering
    raw: Dict[int, int] = {}
    for root, members in root_to_members.items():
        o = group_min_order[root]
        for m in members:
            raw[m] = o

    uniq = sorted({o for o in raw.values() if o != 0})
    compress = {o: i + 1 for i, o in enumerate(uniq)}
    unified = {m: (0 if o == 0 else compress[o]) for m, o in raw.items()}

    # output rows
    rows: List[Dict[str, Any]] = []

    def root_sort_key(root: int):
        o = group_min_order[root]
        if o == 0:
            return (0, 0)
        # Put the user-chosen end node's group at the end (so it stays last after sorting).
        if end_id_int is not None and end_id_int in root_to_members[root]:
            return (2, o)
        return (1, o)

    for root in sorted(root_to_members.keys(), key=root_sort_key):
        members = root_to_members[root]
        unit_members = [m for m in members if base_order[m] == 0]
        non_unit_members = [m for m in members if base_order[m] != 0]

        for mid in unit_members:
            rec = id2rec[mid]
            rows.append(
                {
                    "id": int(rec.get("id")),
                    "Area": get_area(rec),
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

        non_unit_sorted = sorted(
            non_unit_members,
            key=lambda mid: (str(id2rec[mid].get(inner_sort_key, "")), int(mid)),
        )

        sub = sub_order_start
        for mid in non_unit_sorted:
            rec = id2rec[mid]
            rows.append(
                {
                    "id": int(rec.get("id")),
                    "Area": get_area(rec),
                    "tracking_number": rec.get("tracking_number"),
                    "address_road": rec.get("address_road"),
                    "address2": rec.get("address2"),
                    "lat": float(rec.get("lat")),
                    "lng": float(rec.get("lng")),
                    "apartment_flag": int(rec.get("apartment_flag") or 0),
                    "ordering": int(unified[mid]),
                    "sub_order": int(sub),
                }
            )
            sub += 1

    return pd.DataFrame(rows)
