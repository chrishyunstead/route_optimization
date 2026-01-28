from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from .cache import DEFAULT_CACHE_DIR, DEFAULT_MATRIX_UNIT
from .payload import tracking_number_to_id, drop_node_from_payload, get_address_list, get_area
from .remap import remap_unit_to_zero
from .solver import solve_alns_full_cached, solve_alns_ids
from .postprocess import post_group_ordering_and_suborder


def solve_alns_to_df_later_supernode(
    payload: Dict[str, Any],
    *,
    matrix_key: str = "dist_matrix",
    start_id: Optional[int] = 1,
    end_id: Optional[int] = None,
    # NEW: tracking_number based selection
    selected_start_tracking_number: Optional[str] = None,
    selected_end_tracking_number: Optional[str] = None,
    selected_start_id: Optional[int] = None,
    selected_end_id: Optional[int] = None,
    # best-of-k on cache MISS
    miss_enable_bestof_k: bool = False,
    miss_bestof_k: int = 8,
    miss_short_iters: int = 1500,
    miss_seed_stride: int = 10007,
    miss_refine: bool = True,
    # solve params
    seed: Optional[int] = None,
    max_iters: Optional[int] = None,
    use_cache: bool = True,
    cache_dir: str = DEFAULT_CACHE_DIR,
    matrix_unit: str = DEFAULT_MATRIX_UNIT,
    # grouping
    group_same_coords: bool = True,
    group_same_road_addr2: bool = True,
    group_same_road_apartment: bool = True,
    group_same_road: bool = True,
    round_ndigits: int = 6,
    inner_sort_key: str = "tracking_number",
) -> pd.DataFrame:
    """Main entry.

    Flow
    1) Convert tracking_number -> id (optional)
    2) Remap start node id to 0 (label only)
    3) Run ALNS (no grouping)
    4) Post-group ordering + sub_order
    """
    
    # start_id는 "기본 유닛 id"로 취급
    unit_id = int(start_id) if start_id is not None else None

    start_from_tn = tracking_number_to_id(payload, selected_start_tracking_number)
    end_from_tn = tracking_number_to_id(payload, selected_end_tracking_number)

    user_start = (start_from_tn is not None) or (selected_start_id is not None)
    user_end   = (end_from_tn is not None) or (selected_end_id is not None)
    has_custom_endpoint = user_start or user_end

    # 사용자 지정 우선순위 반영
    if user_start:
        start_old = start_from_tn if start_from_tn is not None else selected_start_id
    elif user_end:
        # end만 지정된 케이스: start를 유닛으로 두지 말고 "자유 시작"
        start_old = None
    else:
        # 아무 지정 없을 때만 유닛을 start로 사용
        start_old = start_id

    if user_end:
        end_old = end_from_tn if end_from_tn is not None else selected_end_id
    else:
        end_old = end_id

    payload_eff = payload

    # 0) 유닛 레코드 저장(시각화용)
    unit_rec = None
    if unit_id is not None:
        for r in get_address_list(payload):
            if int(r["id"]) == int(unit_id):
                unit_rec = dict(r)
                break

    # start/end가 하나라도 지정되면 유닛은 경로에 포함되면 안됨 → payload에서 제거
    if has_custom_endpoint and unit_id is not None:
        payload_eff = drop_node_from_payload(payload_eff, drop_id=unit_id, matrix_key=matrix_key)

    start_eff = start_old
    end_eff = end_old

    # remap start -> 0
    if start_eff is not None:
        payload_eff, new_to_old = remap_unit_to_zero(payload_eff, unit_original_id=int(start_eff), matrix_key=matrix_key)
        old_to_new = {int(old): int(new) for new, old in new_to_old.items()}
        start_eff = 0
        if end_eff is not None:
            if int(end_eff) not in old_to_new:
                raise ValueError(f"end_id={end_eff} not found in payload ids.")
            end_eff = old_to_new[int(end_eff)]

    ordered_ids = solve_alns_ids(
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

    # if user explicitly selected a start (via tn or selected_start_id), do NOT create ordering==0.
    is_custom_start = (start_from_tn is not None) or (selected_start_id is not None)

    df = post_group_ordering_and_suborder(
        payload_eff,
        ordered_ids,
        start_id=(start_eff if not is_custom_start else None),
        end_id=end_eff,
        round_ndigits=round_ndigits,
        inner_sort_key=inner_sort_key,
        enable_same_coords=bool(group_same_coords),
        enable_same_road_addr2=bool(group_same_road_addr2),
        enable_apartment_road=bool(group_same_road_apartment),
        enable_same_road=bool(group_same_road),
        sub_order_start=1,
    )

    df.sort_values(["ordering", "sub_order"], kind="stable").reset_index(drop=True)

    # 출발/도착지가 있을 경우, 마지막에 유닛 다시 붙이기 (경로에는 포함 안 되지만 지도에는 찍히도록)
    if has_custom_endpoint and unit_rec is not None:
        unit_row = {
            "id": int(unit_rec.get("id")),
            "Area": get_area(unit_rec),
            "tracking_number": unit_rec.get("tracking_number"),
            "address_road": unit_rec.get("address_road"),
            "address2": unit_rec.get("address2"),
            "lat": float(unit_rec.get("lat")),
            "lng": float(unit_rec.get("lng")),
            "apartment_flag": int(unit_rec.get("apartment_flag") or 0),
            # 중요: 경로 ordering에 섞이지 않게 별도 값 부여
            "ordering": -1,
            "sub_order": 0,
        }
        df = pd.concat([pd.DataFrame([unit_row]), df], ignore_index=True)

    return df


def eval_alns_metrics(
    payload: Dict[str, Any],
    *,
    matrix_key: str = "dist_matrix",
    start_id: Optional[int] = 1,
    end_id: Optional[int] = None,
    seed: Optional[int] = None,
    max_iters: Optional[int] = None,
    use_cache: bool = True,
    cache_dir: str = DEFAULT_CACHE_DIR,
    matrix_unit: str = DEFAULT_MATRIX_UNIT,
    # allow best-of-k in metric runs too
    miss_enable_bestof_k: bool = False,
    miss_bestof_k: int = 8,
    miss_short_iters: int = 1500,
    miss_seed_stride: int = 10007,
    miss_refine: bool = True,
) -> Dict[str, Any]:

    start_eff = start_id
    end_eff = end_id
    payload_eff = payload

    if start_eff is not None:
        payload_eff, new_to_old = remap_unit_to_zero(payload_eff, unit_original_id=int(start_eff), matrix_key=matrix_key)
        old_to_new = {int(old): int(new) for new, old in new_to_old.items()}
        start_eff = 0
        if end_eff is not None:
            if int(end_eff) not in old_to_new:
                raise ValueError(f"end_id={end_eff} not found in payload ids.")
            end_eff = old_to_new[int(end_eff)]

    res = solve_alns_full_cached(
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
        verbose=False,
    )

    best_order = [int(x) for x in res["best_order"]]
    init_order = [int(x) for x in res["init_order"]]

    start_ok = True if start_eff is None else (len(best_order) > 0 and int(best_order[0]) == int(start_eff))
    end_ok = True if end_eff is None else (len(best_order) > 0 and int(best_order[-1]) == int(end_eff))

    init_cost = float(res["init_cost"])
    best_cost = float(res["best_cost"])

    improve_abs = init_cost - best_cost
    improve_pct = (improve_abs / init_cost * 100.0) if init_cost > 0 else 0.0

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
        "bestofk": res.get("bestofk", {}),
    }


def eval_alns_metrics_batch(
    payloads: List[Dict[str, Any]],
    *,
    matrix_key: str = "dist_matrix",
    start_id: Optional[int] = 1,
    end_id: Optional[int] = None,
    seed: Optional[int] = None,
    max_iters: Optional[int] = None,
    use_cache: bool = True,
    cache_dir: str = DEFAULT_CACHE_DIR,
    matrix_unit: str = DEFAULT_MATRIX_UNIT,
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

    def pctl(s: pd.Series, q: float) -> float:
        if len(s) == 0:
            return 0.0
        return float(np.percentile(s.astype(float).to_numpy(), q))

    summary = {
        "n_cases": int(len(df)),
        "max_iters": int(df["max_iters"].iloc[0]) if len(df) else int(max_iters or 0),
        "best_cost_mean": float(df["best_cost"].mean()) if len(df) else 0.0,
        "best_cost_median": float(df["best_cost"].median()) if len(df) else 0.0,
        "best_cost_p95": pctl(df["best_cost"], 95),
        "improve_pct_mean": float(df["improve_pct"].mean()) if len(df) else 0.0,
        "improve_pct_median": float(df["improve_pct"].median()) if len(df) else 0.0,
        "improve_pct_p95": pctl(df["improve_pct"], 95),
        "runtime_ms_mean": float(df["runtime_ms"].mean()) if len(df) else 0.0,
        "runtime_ms_p95": pctl(df["runtime_ms"], 95),
        "start_ok_rate": float(df["start_ok"].mean()) if len(df) else 0.0,
        "end_ok_rate": float(df["end_ok"].mean()) if len(df) else 0.0,
        "cache_hit_rate": float(df["cache_hit"].mean()) if len(df) else 0.0,
    }

    return df, summary
