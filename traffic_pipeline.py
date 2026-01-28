import os, json, hashlib
from datetime import datetime
from dateutil import tz

from traffic_api import generate_its_grid_snapshot, get_5min_bucket_str
from speed_segment import generate_speed_segment, OUTPUT_CSV

KST = tz.gettz("Asia/Seoul")
CACHE_META = "./cache/last_osrm_speed_key.json"

def _file_sig(path: str) -> str:
    try:
        st = os.stat(path)
        return f"{int(st.st_mtime)}:{st.st_size}"
    except FileNotFoundError:
        return "missing"

def _pipe_ver() -> str:
    parts = [
        _file_sig("./grid/W_grid_link_100m.npz"),
        _file_sig("./grid/link_index_100m.json"),
        _file_sig("./traffic_api.py"),
        _file_sig("./speed_segment.py"),
    ]
    return hashlib.sha1("|".join(parts).encode("utf-8")).hexdigest()[:8]

def _make_cache_key(now_str: str) -> str:
    return f"osrm_speed:v1:{_pipe_ver()}:{now_str}"

def _load_last_key() -> str | None:
    try:
        with open(CACHE_META, "r", encoding="utf-8") as f:
            return json.load(f).get("cache_key")
    except Exception:
        return None

def _save_last_key(key: str):
    os.makedirs(os.path.dirname(CACHE_META), exist_ok=True)
    with open(CACHE_META, "w", encoding="utf-8") as f:
        json.dump(
            {"cache_key": key, "created_at": datetime.now(tz=KST).isoformat()},
            f,
            ensure_ascii=False,
            indent=2,
        )

def run_traffic_to_osrm_csv(debug: bool = True):
    now_str = get_5min_bucket_str()
    cache_key = _make_cache_key(now_str)
    last_key = _load_last_key()

    if last_key == cache_key and os.path.exists(OUTPUT_CSV):
        if debug:
            print(f"[CACHE HIT] {cache_key}")
            print(f"[CACHE HIT] reuse: {OUTPUT_CSV}")
        meta = {"mode": "cache_hit", "cache_key": cache_key}
        return None, OUTPUT_CSV, meta, now_str

    grid_df, snapshot_path, meta, now_str = generate_its_grid_snapshot(
        now_str=now_str,
        debug=debug,
        save_snapshot=False,   # ✅ 중간 CSV 저장 X
    )

    if grid_df is None:
        meta = meta or {}
        meta["cache_key"] = cache_key
        return None, None, meta, now_str

    segment_csv_path = generate_speed_segment(
        now_str=now_str,
        debug=debug,
        grid_snapshot_df=grid_df,  # ✅ 메모리 전달
    )

    _save_last_key(cache_key)
    meta = meta or {}
    meta["cache_key"] = cache_key
    return snapshot_path, segment_csv_path, meta, now_str