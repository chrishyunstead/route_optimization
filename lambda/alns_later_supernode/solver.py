from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple, Union  

import random
import time

import numpy as np

from alns import ALNS
from alns.accept import SimulatedAnnealing
from alns.stop import MaxIterations

from .cache import (
    DEFAULT_CACHE_DIR,
    DEFAULT_MATRIX_UNIT,
    load_cache,
    problem_key,
    quantize_matrix,
    save_cache,
    seed_from_key,
    set_deterministic,
)
from .init_methods import build_init_ids_multi_start, repair_regret2
from .operators import (
    RouteState,
    calc_cost,
    destroy_random,
    destroy_segment,
    destroy_worst,
    freeze,
    repair_greedy,
)
from .payload import build_id_maps, get_address_list, get_matrix


class RandomOpSelectCompat:
    """Compatibility layer for operator selection across alns versions."""

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


def solve_alns_full_cached(
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
    # init
    init_trials: int = 6,
    init_rcl_size: int = 2,
    init_use_regret2: bool = True,
    init_use_relocate_ls: bool = True,
    init_relocate_rounds: int = 1,
    # SA
    sa_start_mult: float = 5.0,
    sa_end_mult: float = 0.01,
    # best-of-k (MISS only)
    miss_enable_bestof_k: bool = False,
    miss_bestof_k: int = 8,
    miss_short_iters: int = 1500,
    miss_seed_stride: int = 10007,
    miss_refine: bool = True,
) -> Dict[str, Any]:
    """Run ALNS and return rich metrics, with deterministic cache key.

    NOTE on `miss_seed_stride`:
    - We generate candidate seeds as: base + i*stride.
    - stride is chosen as a large odd number (good bit mixing on 32-bit wrap),
      and not a multiple of common moduli, to avoid correlated RNG streams.
    """

    address_list = get_address_list(payload)
    mat_raw = get_matrix(payload, matrix_key)
    mat = quantize_matrix(mat_raw, unit=matrix_unit)

    id2idx, _ = build_id_maps(address_list)

    if start_id is not None and int(start_id) not in id2idx:
        raise ValueError(f"start_id={start_id} not found in address_list ids.")
    if end_id is not None and int(end_id) not in id2idx:
        raise ValueError(f"end_id={end_id} not found in address_list ids.")

    max_iters_eff = int(max_iters) if max_iters is not None else 2000
    node_ids = [int(r["id"]) for r in address_list]

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

    key_hex = problem_key(mat, node_ids=node_ids, start_id=start_id, end_id=end_id, opts=opts)

    cache_miss = False
    if use_cache and cache_dir:
        cached = load_cache(cache_dir, key_hex)
        if cached is not None:
            cached["cache_hit"] = True
            if verbose:
                print(f"[ALNS] dist_matrix unchanged → cache HIT (key={key_hex[:16]})")
            return cached
        cache_miss = True
        if verbose:
            print(f"[ALNS] cache MISS (key={key_hex[:16]})")

    seed_base = int(seed) if seed is not None else seed_from_key(key_hex)

    def run_once(seed_run: int, iters_run: int, tag: str = "") -> Dict[str, Any]:
        set_deterministic(seed_run)
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
        init_cost = calc_cost(init_ids, id2idx, mat)
        init_state = freeze(init_ids, id2idx, mat)

        t0 = time.perf_counter()

        alns = ALNS(rnd)
        alns.add_destroy_operator(lambda s, r: destroy_random(s, r, start_id, end_id))
        alns.add_destroy_operator(lambda s, r: destroy_worst(s, r, start_id, end_id, id2idx, mat))
        alns.add_destroy_operator(lambda s, r: destroy_segment(s, r, start_id, end_id))

        def repair_greedy_wrap(destroyed, r):
            partial, removed = destroyed
            new_order = repair_greedy(partial, removed, r, start_id, end_id, id2idx, mat)
            return freeze(new_order, id2idx, mat)

        def repair_regret2_wrap(destroyed, r):
            partial, removed = destroyed
            new_order = repair_regret2(
                partial,
                removed,
                r,
                start_id=start_id,
                end_id=end_id,
                id2idx=id2idx,
                mat=mat,
            )
            return freeze(new_order, id2idx, mat)

        alns.add_repair_operator(repair_greedy_wrap)
        alns.add_repair_operator(repair_regret2_wrap)

        n_edges = max(1, len(init_state.order) - 1)
        avg_edge = init_state.cost / n_edges
        start_temp = max(1.0, avg_edge * sa_start_mult)
        end_temp = max(1e-3, avg_edge * sa_end_mult)
        step = (end_temp / start_temp) ** (1.0 / max(1, iters_run))

        accept = SimulatedAnnealing(start_temperature=start_temp, end_temperature=end_temp, step=step)
        stop = MaxIterations(int(iters_run))
        op_select = RandomOpSelectCompat(len(alns.destroy_operators), len(alns.repair_operators))

        result = alns.iterate(init_state, op_select, accept, stop)
        best_order = list(result.best_state.order)
        best_cost = calc_cost(best_order, id2idx, mat)

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

    # best-of-k on MISS
    do_bestofk = bool(miss_enable_bestof_k) and bool(cache_miss)
    picked: Optional[Dict[str, Any]] = None
    bestofk_trials: List[Dict[str, Any]] = []

    if do_bestofk:
        k = max(1, int(miss_bestof_k))
        short_iters = max(1, int(min(max_iters_eff, int(miss_short_iters))))

        if verbose:
            print(f"[ALNS] best-of-k enabled on MISS: k={k}, short_iters={short_iters}, refine={miss_refine}")

        for i in range(k):
            s = (seed_base + i * int(miss_seed_stride)) & 0xFFFFFFFF
            trial = run_once(int(s), int(short_iters), tag=f"[trial {i+1}/{k}]")
            bestofk_trials.append({"seed": trial["seed"], "best_cost": trial["best_cost"], "runtime_ms": trial["runtime_ms"]})
            if picked is None:
                picked = trial
            else:
                if (trial["best_cost"] < picked["best_cost"] - 1e-12) or (
                    abs(trial["best_cost"] - picked["best_cost"]) <= 1e-12 and trial["seed"] < picked["seed"]
                ):
                    picked = trial

        assert picked is not None
        picked_seed = int(picked["seed"])
        if bool(miss_refine) and max_iters_eff > short_iters:
            final_run = run_once(picked_seed, int(max_iters_eff), tag="[refine]")
        else:
            final_run = picked

        bestofk_meta = {"enabled": True, "k": int(k), "short_iters": int(short_iters), "picked_seed": int(picked_seed), "trials": bestofk_trials}

    else:
        final_run = run_once(int(seed_base), int(max_iters_eff), tag="[single]")
        bestofk_meta = {"enabled": False}

    out = {
        "key": key_hex[:16],
        "key_full": key_hex,
        "seed": int(final_run["seed"]),
        "cache_hit": False,
        "n_nodes": int(len(address_list)),
        "max_iters": int(max_iters_eff),
        "opts": opts,
        "init_order": final_run["init_order"],
        "best_order": final_run["best_order"],
        "init_cost": float(final_run["init_cost"]),
        "best_cost": float(final_run["best_cost"]),
        "runtime_ms": float(final_run["runtime_ms"]),
        "bestofk": bestofk_meta,
    }

    if use_cache and cache_dir:
        save_cache(cache_dir, key_hex, out)

    return out


def solve_alns_ids(
    payload: Dict[str, Any],
    *,
    matrix_key: str = "dist_matrix",
    start_id: Optional[int] = 0,
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
    return_meta: bool = False,
) -> Union[List[int], Tuple[List[int], Dict[str, Any]]]:
    res = solve_alns_full_cached(
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
    best = list(res["best_order"])
    if return_meta:
        return best, res
    return best
