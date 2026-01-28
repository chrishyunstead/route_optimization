# speed_segment.py

import os
import hashlib
import time
from datetime import datetime
from typing import Optional, Dict, Any, Tuple

import pandas as pd
import geopandas as gpd
import numpy as np
import shapely
from shapely.geometry import Point, LineString, MultiLineString
from pyrosm import OSM
from dateutil import tz


# ----------------------------
# 설정
# ----------------------------
PBF_PATH = "./total_poly/total_region.osm.pbf"
TOTAL_SHP = "./total_poly/total_region.shp"

OUTPUT_DIR = "./tsp-route-optimization/osrm-data"
OUTPUT_CSV = f"{OUTPUT_DIR}/updates_total_midpoint_100m.csv"
HISTORY_DIR = f"{OUTPUT_DIR}/history"

MIDPOINT_CACHE_PARQUET = "./cache/total_edges_midpoints_5179.parquet"  # [u,v,x,y,lon,lat]
MIDPOINT_CACHE_META_JSON = MIDPOINT_CACHE_PARQUET.replace(".parquet", ".meta.json")

ENABLE_HISTORY_ENV = "ENABLE_OSRM_HISTORY"

# 권역 폴리곤 엄밀 필터 토글 (기본 False)
STRICT_REGION_FILTER = os.getenv("STRICT_REGION_FILTER", "0").strip().lower() in ("1", "true", "t", "yes", "y", "on")
WITHIN_CHUNK_SIZE = int(os.getenv("WITHIN_CHUNK_SIZE", "500000"))

# CRS
CRS_WGS84 = "EPSG:4326"
CRS_KOREA = "EPSG:5179"

# ----------------------------
# ✅ grid_id 산술 매핑 상수 (100m 격자)
#  - grid_id = KR_{row}_{col}
#  - row = floor((y - origin_y)/100)
#  - col = floor((x - origin_x)/100)
#  - origin_x/origin_y: KR_0_0 셀의 좌하단 (EPSG:5179 meters)
# ----------------------------
GRID_CELL_M = 100.0
GRID_ORIGIN_X = 746110.0
GRID_ORIGIN_Y = 1458754.0


# ----------------------------
# 공용 유틸
# ----------------------------
def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}")


def _env_bool(name: str, default: bool = False) -> bool:
    v = os.getenv(name)
    if v is None:
        return default
    return v.strip().lower() in ("1", "true", "t", "yes", "y", "on")


def _safe_makedirs(path: str) -> None:
    if path and not os.path.exists(path):
        os.makedirs(path, exist_ok=True)


def _quick_file_fingerprint(path: str, head_bytes: int = 1024 * 1024, tail_bytes: int = 1024 * 1024) -> str:
    st = os.stat(path)
    size = st.st_size
    mtime_ns = getattr(st, "st_mtime_ns", int(st.st_mtime * 1e9))

    h = hashlib.sha256()
    h.update(str(size).encode("utf-8"))
    h.update(str(mtime_ns).encode("utf-8"))

    with open(path, "rb") as f:
        head = f.read(head_bytes)
        h.update(head)
        if size > tail_bytes:
            f.seek(max(size - tail_bytes, 0))
            tail = f.read(tail_bytes)
            h.update(tail)
    return h.hexdigest()


def _shapefile_fingerprint(shp_path: str) -> str:
    base, _ = os.path.splitext(shp_path)
    candidates = [
        f"{base}.shp",
        f"{base}.shx",
        f"{base}.dbf",
        f"{base}.prj",
        f"{base}.cpg",
        f"{base}.sbn",
        f"{base}.sbx",
        f"{base}.qix",
        f"{base}.fix",
    ]
    parts = []
    for p in candidates:
        if os.path.exists(p):
            parts.append(_quick_file_fingerprint(p, head_bytes=128 * 1024, tail_bytes=128 * 1024))
        else:
            parts.append("MISSING")

    h = hashlib.sha256()
    h.update("|".join(parts).encode("utf-8"))
    return h.hexdigest()


def _is_oneway(val) -> bool:
    if isinstance(val, str):
        return val.strip().lower() in ("yes", "true", "1", "t", "y")
    return bool(val)


def normalize_uv(edges: pd.DataFrame, nodes_gdf: gpd.GeoDataFrame | None = None) -> pd.DataFrame:
    candidates = [
        ("u", "v"),
        ("from", "to"),
        ("from_node", "to_node"),
        ("src", "dst"),
        ("source", "target"),
    ]
    for a, b in candidates:
        if a in edges.columns and b in edges.columns:
            if a != "u" or b != "v":
                edges = edges.rename(columns={a: "u", b: "v"})
            return edges

    if nodes_gdf is not None:
        df = edges.copy()

        def endpoints(geom):
            if isinstance(geom, LineString):
                c = geom.coords
                return Point(c[0]), Point(c[-1])
            elif isinstance(geom, MultiLineString):
                f, l = geom.geoms[0], geom.geoms[-1]
                return Point(f.coords[0]), Point(l.coords[-1])
            return None, None

        s, t = zip(*df.geometry.apply(endpoints))
        start_gdf = gpd.GeoDataFrame(geometry=gpd.GeoSeries(s, crs=CRS_WGS84))
        end_gdf = gpd.GeoDataFrame(geometry=gpd.GeoSeries(t, crs=CRS_WGS84))

        if nodes_gdf.crs is None:
            nodes_gdf = nodes_gdf.set_crs(CRS_WGS84, allow_override=True)
        elif nodes_gdf.crs.to_string() != CRS_WGS84:
            nodes_gdf = nodes_gdf.to_crs(CRS_WGS84)

        _ = nodes_gdf.sindex
        hit_s = gpd.sjoin_nearest(start_gdf, nodes_gdf[["id", "geometry"]], how="left", distance_col="d")
        hit_t = gpd.sjoin_nearest(end_gdf, nodes_gdf[["id", "geometry"]], how="left", distance_col="d")

        df["u"] = hit_s["id"].astype("int64")
        df["v"] = hit_t["id"].astype("int64")
        return df

    raise KeyError(f"u/v 칼럼을 찾지 못했습니다. 현재 칼럼: {list(edges.columns)}")


def _read_meta(meta_path: str) -> Dict[str, Any]:
    try:
        if os.path.exists(meta_path):
            return pd.read_json(meta_path, typ="series").to_dict()
    except Exception:
        pass
    return {}


def _write_meta(meta_path: str, meta: Dict[str, Any]) -> None:
    _safe_makedirs(os.path.dirname(meta_path))
    pd.Series(meta).to_json(meta_path)


def _cache_valid(meta: Dict[str, Any], *, pbf_sig: str, shp_sig: str) -> bool:
    return (
        meta.get("pbf_sig") == pbf_sig
        and meta.get("shp_sig") == shp_sig
        and meta.get("pbf_path") == PBF_PATH
        and meta.get("shp_path") == TOTAL_SHP
    )


def xy_to_grid_id_vec(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """
    EPSG:5179(meters) x,y -> grid_id('KR_{row}_{col}') (벡터화).
    """
    col = np.floor((x - GRID_ORIGIN_X) / GRID_CELL_M).astype(np.int64)
    row = np.floor((y - GRID_ORIGIN_Y) / GRID_CELL_M).astype(np.int64)
    return np.char.add(np.char.add("KR_", row.astype(str)), np.char.add("_", col.astype(str)))


# ----------------------------
# 권역 union/bounds
# ----------------------------
def _load_region_union_and_bounds() -> Tuple[Any, Any, Tuple[float, float, float, float]]:
    region = gpd.read_file(TOTAL_SHP)
    if region.crs is None:
        region = region.set_crs(CRS_WGS84, allow_override=True)
    else:
        region = region.to_crs(CRS_WGS84)

    union_4326 = region.unary_union
    if not union_4326.is_valid:
        union_4326 = union_4326.buffer(0)

    minx, miny, maxx, maxy = region.total_bounds

    union_5179 = None
    if STRICT_REGION_FILTER:
        union_5179 = gpd.GeoSeries([union_4326], crs=CRS_WGS84).to_crs(CRS_KOREA).iloc[0]
        if not union_5179.is_valid:
            union_5179 = union_5179.buffer(0)

    return union_4326, union_5179, (minx, miny, maxx, maxy)


def _filter_points_in_single_polygon_chunked(
    points_gdf_5179: gpd.GeoDataFrame,
    polygon_5179,
    *,
    chunk_size: int = WITHIN_CHUNK_SIZE,
    debug: bool = True,
) -> gpd.GeoDataFrame:
    t0 = time.perf_counter()
    minx, miny, maxx, maxy = polygon_5179.bounds

    x = points_gdf_5179.geometry.x.to_numpy()
    y = points_gdf_5179.geometry.y.to_numpy()
    bbox_mask = (x >= minx) & (x <= maxx) & (y >= miny) & (y <= maxy)
    cand_idx = np.flatnonzero(bbox_mask)

    if debug:
        log(f"[STRICT] bbox prefilter: cand={cand_idx.size:,} / total={len(points_gdf_5179):,}")

    if cand_idx.size == 0:
        if debug:
            log(f"[STRICT] contains(chunked): kept=0 (bbox empty), {time.perf_counter()-t0:.1f}s")
        return points_gdf_5179.iloc[0:0].copy()

    geom = points_gdf_5179.geometry.to_numpy()
    keep = np.zeros(len(points_gdf_5179), dtype=bool)

    t1 = time.perf_counter()
    for i in range(0, cand_idx.size, chunk_size):
        idx = cand_idx[i:i + chunk_size]
        keep[idx] = shapely.contains(polygon_5179, geom[idx])

    out = points_gdf_5179.loc[keep].copy()
    if debug:
        log(
            f"[STRICT] contains(chunked): kept={len(out):,} / cand={cand_idx.size:,}, "
            f"chunk={chunk_size:,}, {time.perf_counter()-t1:.1f}s (total {time.perf_counter()-t0:.1f}s)"
        )
    return out


# ----------------------------
# 미드포인트 캐시 (라인 to_crs 제거 버전)
# ----------------------------
def _load_or_build_midpoints(debug: bool = True) -> pd.DataFrame:
    if not os.path.exists(PBF_PATH):
        raise FileNotFoundError(f"PBF 파일을 찾을 수 없습니다: {PBF_PATH}")
    if not os.path.exists(TOTAL_SHP):
        raise FileNotFoundError(f"SHP 파일을 찾을 수 없습니다: {TOTAL_SHP}")

    pbf_sig = _quick_file_fingerprint(PBF_PATH, head_bytes=512 * 1024, tail_bytes=512 * 1024)
    shp_sig = _shapefile_fingerprint(TOTAL_SHP)
    meta = _read_meta(MIDPOINT_CACHE_META_JSON)

    if os.path.exists(MIDPOINT_CACHE_PARQUET) and _cache_valid(meta, pbf_sig=pbf_sig, shp_sig=shp_sig):
        mids_df = pd.read_parquet(MIDPOINT_CACHE_PARQUET)
        if debug:
            log(f"Loaded midpoint cache: {MIDPOINT_CACHE_PARQUET}, rows={len(mids_df):,}")
        return mids_df

    if debug:
        why = []
        if not os.path.exists(MIDPOINT_CACHE_PARQUET):
            why.append("cache_missing")
        if meta.get("pbf_sig") != pbf_sig:
            why.append("pbf_changed")
        if meta.get("shp_sig") != shp_sig:
            why.append("shp_changed")
        if meta.get("pbf_path") != PBF_PATH:
            why.append("pbf_path_changed")
        if meta.get("shp_path") != TOTAL_SHP:
            why.append("shp_path_changed")
        log(f"Rebuilding midpoints cache ({', '.join(why) if why else 'unknown'})")
        log(f"STRICT_REGION_FILTER={STRICT_REGION_FILTER}")

    # region bounds 준비
    _, union_5179, (minx, miny, maxx, maxy) = _load_region_union_and_bounds()

    # 도로 네트워크 로드
    t0 = time.perf_counter()
    osm = OSM(PBF_PATH)
    net = osm.get_network(network_type="driving", nodes=True, extra_attributes=["oneway"])
    if debug:
        log(f"Loaded network: {time.perf_counter() - t0:.1f}s")

    if isinstance(net, tuple):
        nodes_gdf, edges = net
    else:
        nodes_gdf = None
        edges = osm.get_network(network_type="driving", extra_attributes=["oneway"])

    edges = gpd.GeoDataFrame(edges, geometry="geometry", crs=CRS_WGS84)
    edges = edges[edges.geometry.notnull()].copy()

    # BBOX 선필터 (4326)
    t0 = time.perf_counter()
    edges = edges.cx[minx:maxx, miny:maxy]
    if debug:
        log(f"BBOX slice edges: {time.perf_counter() - t0:.1f}s, rows={len(edges):,}")

    # u/v, oneway 정규화
    t0 = time.perf_counter()
    edges = normalize_uv(edges, nodes_gdf=nodes_gdf)
    if "oneway" not in edges.columns:
        edges["oneway"] = False
    edges["oneway"] = edges["oneway"].apply(_is_oneway)
    if debug:
        log(f"normalize u/v & oneway: {time.perf_counter() - t0:.1f}s")

    # base edges (복제 전)
    one = edges[edges["oneway"]][["u", "v", "geometry"]].copy()
    bi = edges[~edges["oneway"]][["u", "v", "geometry"]].copy()
    one["bidir"] = False
    bi["bidir"] = True

    edges_base = pd.concat([one, bi], ignore_index=True)
    edges_base = gpd.GeoDataFrame(edges_base, geometry="geometry", crs=CRS_WGS84)
    if debug:
        log(f"edges_base built: rows={len(edges_base):,}")

    # midpoint 4326에서 계산
    t0 = time.perf_counter()
    mids_geom_4326 = edges_base.geometry.interpolate(0.5, normalized=True)
    mids_base_4326 = gpd.GeoDataFrame(
        edges_base[["u", "v", "bidir"]].reset_index(drop=True),
        geometry=mids_geom_4326.reset_index(drop=True),
        crs=CRS_WGS84,
    )
    if debug:
        log(f"interpolate(midpoint) in 4326: {time.perf_counter() - t0:.1f}s, rows={len(mids_base_4326):,}")

    # midpoint 포인트만 5179로 변환
    t0 = time.perf_counter()
    mids_base_5179 = mids_base_4326.to_crs(CRS_KOREA)
    if debug:
        log(f"to_crs(5179) midpoints(points only): {time.perf_counter() - t0:.1f}s")

    # (선택) 엄밀 권역 필터
    if STRICT_REGION_FILTER:
        if union_5179 is None:
            raise RuntimeError("STRICT_REGION_FILTER=True 이지만 union_5179가 준비되지 않았습니다.")
        mids_base_5179 = _filter_points_in_single_polygon_chunked(
            mids_base_5179,
            union_5179,
            chunk_size=WITHIN_CHUNK_SIZE,
            debug=debug,
        )
        mids_base_4326 = mids_base_4326.loc[mids_base_5179.index].copy()
    else:
        if debug:
            log("STRICT_REGION_FILTER=False (bbox only). skip polygon contains.")

    # base_df 구성
    base_df = pd.DataFrame(
        {
            "u": mids_base_5179["u"].astype(np.int64).values,
            "v": mids_base_5179["v"].astype(np.int64).values,
            "bidir": mids_base_5179["bidir"].astype(bool).values,
            "x": mids_base_5179.geometry.x.values.astype(np.float32),
            "y": mids_base_5179.geometry.y.values.astype(np.float32),
            "lon": mids_base_4326.geometry.x.values.astype(np.float64),
            "lat": mids_base_4326.geometry.y.values.astype(np.float64),
        }
    )

    # bidir 복제(pandas)
    df_one = base_df[~base_df["bidir"]].drop(columns=["bidir"])
    df_bi = base_df[base_df["bidir"]].drop(columns=["bidir"])
    df_bi_rev = df_bi.rename(columns={"u": "v", "v": "u"})

    mids_df = pd.concat([df_one, df_bi, df_bi_rev], ignore_index=True)

    # dtype 확정
    mids_df["u"] = mids_df["u"].astype(np.int64)
    mids_df["v"] = mids_df["v"].astype(np.int64)
    mids_df["x"] = mids_df["x"].astype(np.float32)
    mids_df["y"] = mids_df["y"].astype(np.float32)
    mids_df["lon"] = mids_df["lon"].astype(np.float64)
    mids_df["lat"] = mids_df["lat"].astype(np.float64)

    # 캐시 저장
    _safe_makedirs(os.path.dirname(MIDPOINT_CACHE_PARQUET))
    mids_df.to_parquet(MIDPOINT_CACHE_PARQUET, index=False)

    new_meta = {
        "pbf_path": PBF_PATH,
        "shp_path": TOTAL_SHP,
        "pbf_sig": pbf_sig,
        "shp_sig": shp_sig,
        "created_at": datetime.utcnow().isoformat(timespec="seconds") + "Z",
        "rows": int(len(mids_df)),
        "strict_region_filter": bool(STRICT_REGION_FILTER),
    }
    _write_meta(MIDPOINT_CACHE_META_JSON, new_meta)

    log(f"Saved midpoint cache: {MIDPOINT_CACHE_PARQUET}, rows={len(mids_df):,}")
    log(f"Saved midpoint cache meta: {MIDPOINT_CACHE_META_JSON}")

    return mids_df


# ----------------------------
# 메인
# ----------------------------
def generate_speed_segment(now_str: str | None = None, debug: bool = True, grid_snapshot_df: pd.DataFrame | None = None) -> str:
    if now_str is None: 
        KST = tz.gettz("Asia/Seoul")
        now = datetime.now(tz=KST)
        bucket_minute = (now.minute // 5) * 5
        bucket_time = now.replace(minute=bucket_minute, second=0, microsecond=0)
        now_str = bucket_time.strftime("%Y-%m-%d_%H-%M-00")

    mids_df = _load_or_build_midpoints(debug=debug)
    if debug:
        log(f"Directed edges(midpoint cached): {len(mids_df):,}")

    # --------------------------------------------------------
    # snapshot 로드 (grid_id -> metric)
    # --------------------------------------------------------
    t0 = time.perf_counter()
    if grid_snapshot_df is None:
        snapshot_csv = f"./grid_traffic_output/its_grid_speed_snapshot_grid100m_{now_str}.csv"
        if not os.path.exists(snapshot_csv):
            raise FileNotFoundError(
                f"격자 스냅샷 파일을 찾을 수 없습니다: {snapshot_csv}\n"
                f"generate_its_grid_snapshot()에서 생성된 시각(now_str)이랑 맞는지 확인하세요."
            )
        if debug:
            print(f"[INFO] Using snapshot CSV: {snapshot_csv}")
        snap = pd.read_csv(snapshot_csv, usecols=["grid_id", "metric"], encoding="utf-8")
    else:
        if debug:
            print("[INFO] Using grid_snapshot_df (in-memory). skip reading snapshot CSV.")
        snap = grid_snapshot_df[["grid_id", "metric"]].copy()
        
    snap = snap.dropna(subset=["grid_id", "metric"]).copy()
    snap["grid_id"] = snap["grid_id"].astype(str)
    snap["metric"] = pd.to_numeric(snap["metric"], errors="coerce").astype(np.float32)
    snap = snap.dropna(subset=["metric"])

    # 중복 grid_id가 있으면 마지막 값 우선
    snap = snap.drop_duplicates(subset=["grid_id"], keep="last")
    metric_series = snap.set_index("grid_id")["metric"]

    if debug:
        log(f"Loaded snapshot map: rows={len(metric_series):,}, {time.perf_counter()-t0:.1f}s")

    # --------------------------------------------------------
    # ✅ sjoin 제거: mids_df(x,y) -> grid_id 산술 매핑 -> metric map
    # --------------------------------------------------------
    t0 = time.perf_counter()

    grid_ids = xy_to_grid_id_vec(
        mids_df["x"].to_numpy(dtype=np.float64),
        mids_df["y"].to_numpy(dtype=np.float64),
    ).astype(str)

    # pandas Index.map은 큰 merge보다 가벼운 편
    speed = pd.Index(grid_ids).map(metric_series).to_numpy(dtype=np.float32)

    mask = np.isfinite(speed)
    matched = int(mask.sum())

    if debug:
        log(
            f"grid_id map: {time.perf_counter()-t0:.1f}s, "
            f"matched={matched:,}/{len(speed):,} ({matched/len(speed)*100:.2f}%)"
        )

    # --------------------------------------------------------
    # 저장 포맷 구성
    # --------------------------------------------------------
    out = pd.DataFrame(
        {
            "from_osm_node_id": mids_df.loc[mask, "u"].to_numpy(dtype=np.int64),
            "to_osm_node_id": mids_df.loc[mask, "v"].to_numpy(dtype=np.int64),
            "speed_kph": speed[mask].astype(np.float32),
        }
    )

    _safe_makedirs(OUTPUT_DIR)
    _safe_makedirs(HISTORY_DIR)

    out.to_csv(OUTPUT_CSV, index=False, header=False, encoding="utf-8", lineterminator="\n")
    if debug:
        log(f"Saved for OSRM: {OUTPUT_CSV}, rows={len(out):,}")

    if _env_bool(ENABLE_HISTORY_ENV, default=False):
        safe_now_str = now_str.replace(":", "-").replace(" ", "_")
        history_csv = f"{HISTORY_DIR}/updates_total_midpoint_100m_{safe_now_str}.csv"
        out.to_csv(history_csv, index=False, header=False, encoding="utf-8", lineterminator="\n")
        if debug:
            log(f"Saved history: {history_csv}")
    else:
        if debug:
            log(f"History disabled ({ENABLE_HISTORY_ENV}=0). skip.")

    return OUTPUT_CSV