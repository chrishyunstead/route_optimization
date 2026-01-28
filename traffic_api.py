import os
import json
import asyncio
from datetime import datetime
import aiohttp
import pandas as pd
import geopandas as gpd
import numpy as np
from dateutil import tz
import xmltodict
import scipy.sparse as sp
from dotenv import load_dotenv
from datetime import timedelta
# .env 로드
load_dotenv()

# ===== 환경/설정 =====
KST = tz.gettz("Asia/Seoul")
BASE_URL   = "https://openapi.its.go.kr:9443/trafficInfo"
API_KEY    = os.getenv("ITS_API_KEY")
ROAD_TYPE  = "all"   # "all"|"ex"|"its"
DRC_TYPE   = "all"   # "all"|"up"|"down"|...

# 사전준비 산출물 로드 (모듈 import 시 1회 실행)
W = sp.load_npz("./grid/W_grid_link_100m.npz")          # (G x L)
grid_index = json.load(open("./grid/grid_index_100m.json"))
link_index = json.load(open("./grid/link_index_100m.json"))
grid_geom  = gpd.read_feather("./grid/grid_geom_100m.feather")  # [grid_id, geometry] (5179로 저장됨)


# 유틸 함수(5분 버킷 함수)
def get_5min_bucket_str(dt: datetime | None = None) -> str:
    """
    dt를 5분 단위로 내림한 "%Y-%m-%d_%H-%M-00" 문자열 반환
    """
    if dt is None:
        dt = datetime.now(tz=KST)
    bucket_minute = (dt.minute // 5) * 5
    bucket_time = dt.replace(minute=bucket_minute, second=0, microsecond=0)
    return bucket_time.strftime("%Y-%m-%d_%H-%M-00")


# ===== 유틸 =====
def _pick(d, *keys):
    for k in keys:
        if isinstance(d, dict) and k in d and d[k] not in (None, ""):
            return d[k]
    return None


def _parse_items(txt):
    """ITS 응답(JSON/XML 모두)에서 items 배열만 뽑기"""
    data = None
    try:
        data = json.loads(txt)
    except Exception:
        try:
            data = xmltodict.parse(txt)
            if "response" in data:
                data = data["response"]
        except Exception:
            return [], {}
    header = data.get("header") if isinstance(data, dict) else {}
    body   = data.get("body") if isinstance(data, dict) else None
    items = None
    if body and isinstance(body, dict):
        items = (body.get("items") or body.get("itemList") or
                 body.get("data")  or body.get("list"))
    if items is None and isinstance(data, dict):
        items = data.get("items")
    if isinstance(items, dict):     # xmltodict 케이스: {"items":{"item":[...]}}
        items = items.get("item") or items.get("items")
    if items is None:
        items = []
    return items, header


async def _one_call(params, timeout=aiohttp.ClientTimeout(total=60), debug=False):
    async with aiohttp.ClientSession(
        connector=aiohttp.TCPConnector(limit=2, ttl_dns_cache=60)
    ) as session:
        async with session.get(BASE_URL, params=params, timeout=timeout) as r:
            txt = await r.text()
            if r.status != 200:
                if debug:
                    print(f"[HTTP {r.status}] {r.url}\n{txt[:400]}")
                return pd.DataFrame([]), {}
            items, header = _parse_items(txt)
            rows = []
            for rec in items or []:
                if not isinstance(rec, dict):
                    continue
                link_id = _pick(rec, "linkId", "linkID", "LINK_ID", "link_id", "LINKID")
                speed   = _pick(rec, "speed", "spd", "SPEED")
                created = _pick(rec, "createdDate", "created_at", "time", "timestamp")
                if link_id is None or speed in (None, ""):
                    continue
                try:
                    speed = float(speed)
                except Exception:
                    continue
                if isinstance(created, str) and len(created) == 14 and created.isdigit():
                    ts = datetime.strptime(created, "%Y%m%d%H%M%S").replace(tzinfo=KST)
                else:
                    ts = datetime.now(tz=KST)
                rows.append({"link_id": str(link_id), "speed": speed, "timestamp": ts})
            df = pd.DataFrame(rows)
            if not df.empty:
                df = (
                    df.sort_values(["link_id", "timestamp"])
                      .drop_duplicates(["link_id", "timestamp"], keep="last")
                      .reset_index(drop=True)
                )
            return df, header


def national_bbox_from_grid_geom(grid_geom_gdf, margin_deg=0.02):
    """준비단계에서 저장된 grid_geom(5179)을 4326으로 변환해 bbox 산출"""
    g = grid_geom_gdf.copy()
    if g.crs is None or g.crs.to_epsg() != 4326:
        g = g.set_crs("EPSG:5179", allow_override=True).to_crs(4326)
    minx, miny, maxx, maxy = g.total_bounds
    return {
        "minX": float(minx - margin_deg),
        "maxX": float(maxx + margin_deg),
        "minY": float(miny - margin_deg),
        "maxY": float(maxy + margin_deg),
    }


async def poll_single_shot_with_fallback(api_key, road_type="all", drc_type="all", debug=True):
    base = {"apiKey": api_key, "apikey": api_key, "type": road_type, "drcType": drc_type}

    # 1) no-bbox 시도
    df1, h1 = await _one_call(base, debug=debug)
    if not df1.empty:
        if debug:
            print(f"✅ no-bbox rows={len(df1)}")
        return df1, {"mode": "no_bbox", "header": h1}

    # 2) bbox fallback 시도
    bbox = national_bbox_from_grid_geom(grid_geom)
    params2 = base | bbox
    df2, h2 = await _one_call(params2, debug=debug)
    if not df2.empty:
        if debug:
            print(f"✅ bbox-fallback rows={len(df2)}  bbox={bbox}")
        return df2, {"mode": "bbox_fallback", "header": h2}

    if debug:
        print("⚠️ ITS 응답 비어있음. header 미리보기:", (h1 or h2))
        print("  - 키/파라미터 재확인: apiKey/type/drcType/bbox")
    return pd.DataFrame([]), {"mode": "failed"}


# ===== 커버리지/스냅샷 =====
def diagnose_mapping_coverage(df_links, link_index, W):
    api_links = set(df_links["link_id"].astype(str).unique())
    map_links = set(link_index.keys())
    mapped = len(api_links & map_links)
    total = len(api_links)
    print(f"[coverage] API link_ids: {total:,} | mapped in W: {mapped:,} ({(mapped/total*100 if total else 0):.1f}%)")
    print(f"[W] shape={W.shape}, nnz={W.nnz:,}")


def compute_grid_snapshot_fast(
    df_links,
    W,
    grid_geom,
    link_index,
    timestamp,
    return_inner_only=True,
    to_wgs84=True,
):
    # 1) 커버리지 확인
    diagnose_mapping_coverage(df_links, link_index, W)

    # 2) 링크 속도 벡터화
    L = len(link_index)
    s = np.zeros(L, dtype=np.float32)
    hits = 0
    for lid, spd in zip(df_links["link_id"].astype(str), df_links["speed"]):
        j = link_index.get(lid)
        if j is not None and np.isfinite(spd):
            s[j] = float(spd)
            hits += 1
    # print(f"[debug] matched link speeds: {hits:,}")

    # 3) 가중 평균 (관측된 링크만 분모 카운트)
    mask = (s > 0).astype(np.float32)
    num = W.dot(s)      # (G,)
    den = W.dot(mask)   # (G,)
    with np.errstate(divide="ignore", invalid="ignore"):
        metric = np.where(den > 0, num / den, np.nan).astype(np.float32)

    out = grid_geom.copy()
    if out.crs is None:
        out = out.set_crs("EPSG:5179", allow_override=True)  # feather CRS 방어
    out["metric"] = metric
    out["timestamp"] = timestamp

    # 4) 관측된 격자만 반환 (예전처럼 inner)
    if return_inner_only:
        out = out[np.isfinite(out["metric"])].reset_index(drop=True)

    # 5) 위경도 변환
    if to_wgs84:
        out = out.to_crs(4326)

    return out[["grid_id", "timestamp", "metric", "geometry"]]


# =========================
#  비동기 실행 본체
# =========================
async def _generate_its_grid_snapshot_async(now_str: str | None = None, debug: bool = True, save_snapshot: bool = True):
    """
    ITS API를 한 번 호출해서 링크 속도 → 격자 속도 스냅샷으로 변환하고
    CSV로 저장까지 한 뒤, (grid_snapshot, csv_path, meta)를 반환하는 비동기 함수
    """
    if not API_KEY:
        raise RuntimeError("ITS_API_KEY가 .env에 설정되어 있지 않습니다.")

    # 1) ITS에서 링크별 속도 스냅샷 가져오기
    df_links, meta = await poll_single_shot_with_fallback(
        api_key=API_KEY,
        road_type=ROAD_TYPE,
        drc_type=DRC_TYPE,
        debug=debug,
    )
    print("=== META ===", meta)

    if df_links.empty:
        print("❌ API 응답이 비었습니다. (no-bbox, bbox 둘 다 실패)")
        if now_str is None:
            now_str = get_5min_bucket_str()
        return None, None, meta, now_str

    # 2) 가장 최신 타임스탬프 기준으로 격자 속도 계산
    tstamp = df_links["timestamp"].max()
    grid_snapshot = compute_grid_snapshot_fast(
        df_links=df_links,
        W=W,
        grid_geom=grid_geom,
        link_index=link_index,
        timestamp=tstamp,
        return_inner_only=True,   # 관측 격자만 (행수 ~ 기존 수준)
        to_wgs84=True             # geometry를 4326(127.x, 37.x)로
    )

    # 3) 파일명용 현재 시각 (5분 버킷)
    if now_str is None:
        now_str = get_5min_bucket_str()

    # 4) 디렉토리 확보 + 저장
    csv_path = None
    if save_snapshot:
        os.makedirs("grid_traffic_output", exist_ok=True)
        csv_path = f"grid_traffic_output/its_grid_speed_snapshot_grid100m_{now_str}.csv"
        grid_snapshot.to_csv(csv_path, index=False, encoding="utf-8")
        if debug:
            print("📁 saved to:", csv_path)
    else:
        if debug:
            print("📁 saved to: (skip) save_snapshot=False")

    return grid_snapshot, csv_path, meta, now_str


# =========================
#  동기 wrapper (app.py에서 이거만 부르면 됨)
#  동기 wrapper도 now_str 받기/반환
# =========================
def generate_its_grid_snapshot(now_str: str | None = None, debug: bool = True, save_snapshot: bool = True):
    """
    app.py 등에서 바로 호출할 수 있는 동기 함수.

        from traffic_api import generate_its_grid_snapshot
        df, path, meta = generate_its_grid_snapshot()

    이런 식으로 사용.
    """
    return asyncio.run(_generate_its_grid_snapshot_async(now_str=now_str, debug=debug, save_snapshot=save_snapshot))