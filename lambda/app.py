import json
import os
import base64
import time
from datetime import datetime, timezone
import hashlib
import boto3
import pandas as pd
import traceback
from utils.db_handler import DBHandler
from queries.item import ItemDatasetQuery
from queries.unit import UnitDatasetQuery
from utils.preprocess.transform_matix import transform_input_data_with_osrm_matrix
from alns_later_supernode.api import solve_alns_to_df_later_supernode

def log_perf(step, start_ts, request_id=None, **extra):
    elapsed_ms = (time.perf_counter() - start_ts) * 1000
    payload = {
        "tag": "perf",
        "step": step,
        "elapsed_ms": round(elapsed_ms, 1),
    }
    if request_id:
        payload["request_id"] = request_id

    for k, v in extra.items():
        if v is not None:
            payload[k] = v

    print(json.dumps(payload, ensure_ascii=False))

def ensure_tmp_dirs():
    for k in ("ALNS_CACHE_DIR", "MPLCONFIGDIR"):
        p = os.environ.get(k)
        if p:
            os.makedirs(p, exist_ok=True)

# 전역 초기화: warm invoke 재사용
ensure_tmp_dirs()

S3_CLIENT = boto3.client("s3")
LAMBDA_CLIENT = boto3.client("lambda")
DB_HANDLER = DBHandler()
ITEM_QUERY = ItemDatasetQuery(DB_HANDLER)
UNIT_QUERY = UnitDatasetQuery(DB_HANDLER)

_UNIT_CACHE = {
    "data": None,
    "loaded_at": 0,
}
_UNIT_CACHE_TTL = 300  # 5분


def get_cached_unit_df(unit_query, request_id=None):
    total_t0 = time.perf_counter()
    now = time.time()

    if (
        _UNIT_CACHE["data"] is not None
        and (now - _UNIT_CACHE["loaded_at"] < _UNIT_CACHE_TTL)
    ):
        df_unit = _UNIT_CACHE["data"]
        print("[cache] unit cache hit")
        log_perf(
            "unit_fetch_or_cache_total",
            total_t0,
            request_id=request_id,
            cache_hit=True,
            rows=int(len(df_unit)) if isinstance(df_unit, pd.DataFrame) else None,
        )
        return df_unit

    print("[cache] unit cache miss")

    t0 = time.perf_counter()
    df_unit = unit_query.unit_dataset_df()
    log_perf(
        "db_unit_fetch",
        t0,
        request_id=request_id,
        rows=int(len(df_unit)) if isinstance(df_unit, pd.DataFrame) else None,
    )

    if isinstance(df_unit, pd.DataFrame) and not df_unit.empty:
        _UNIT_CACHE["data"] = df_unit
        _UNIT_CACHE["loaded_at"] = now
    else:
        print("[cache] unit cache skip (empty or invalid df)")

    log_perf(
        "unit_fetch_or_cache_total",
        total_t0,
        request_id=request_id,
        cache_hit=False,
        rows=int(len(df_unit)) if isinstance(df_unit, pd.DataFrame) else None,
    )
    return df_unit

def pack_df(df: pd.DataFrame, sample_n: int = 500):
    """DF를 columns/data로 패킹 (응답 크기 폭발 방지)"""
    if df is None or not isinstance(df, pd.DataFrame):
        return {"columns": [], "data": []}

    if sample_n is not None:
        df2 = df.head(sample_n).copy()
    else:
        df2 = df.copy()

    df2 = df2.where(df2.notna(), None)  # NaN -> None

    return {
        "columns": list(df2.columns),
        "data": df2.to_dict(orient="records"),
    }

def save_result_to_s3(
    request_id,
    user_id,
    user_selected_start_tn,
    user_selected_end_tn,
    method,
    path,
    reason_code,
    reason,
    packed,
    alns_meta,
):
    bucket = os.environ.get("RESULT_S3_BUCKET")
    prefix = os.environ.get("RESULT_S3_PREFIX", "route-optimization").rstrip("/")

    if not bucket:
        print("[s3] RESULT_S3_BUCKET not set, skip save")
        return None

    now = datetime.now(timezone.utc)
    dt = now.strftime("%Y-%m-%d")
    ts = now.strftime("%Y%m%dT%H%M%SZ")
    req = request_id or f"user-{user_id}-{ts}"

    key = f"{prefix}/dt={dt}/user_id={user_id}/request_id={req}.json"

    payload = {
        "dt": dt,
        "user_id": user_id,
        "saved_at_utc": ts,
        "request_id": request_id,
        "meta": {
            "input": {
                "user_id": user_id,
                "user_selected_start_tn": user_selected_start_tn,
                "user_selected_end_tn": user_selected_end_tn,
            },
            "api": {
                "httpMethod": method,
                "path": path,
            },
            "cache": {
                "hit": bool((alns_meta or {}).get("cache_hit", False)),
                "key16": str((alns_meta or {}).get("key", ""))[:16],
            },
            "status": "OK",
            "reason_code": reason_code,
            "reason": reason,
        },
        "result": {
            "df_ordered": packed,
        },
    }

    body_bytes = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    body_size = len(body_bytes)
    body_md5_16 = hashlib.md5(body_bytes).hexdigest()[:16]

    put_resp = S3_CLIENT.put_object(
        Bucket=bucket,
        Key=key,
        Body=body_bytes,
        ContentType="application/json",
    )

    s3_request_id = (
        (put_resp.get("ResponseMetadata") or {}).get("RequestId")
    )
    etag = (put_resp.get("ETag") or "").replace('"', "")
    version_id = put_resp.get("VersionId")

    print(json.dumps({
        "tag": "s3_verify",
        "step": "put_object_ok",
        "request_id": request_id,
        "user_id": user_id,
        "bucket": bucket,
        "key": key,
        "body_size": body_size,
        "body_md5_16": body_md5_16,
        "etag": etag,
        "s3_request_id": s3_request_id,
        "version_id": version_id,
    }, ensure_ascii=False))

    return {
        "bucket": bucket,
        "key": key,
        "body_size": body_size,
        "body_md5_16": body_md5_16,
        "etag": etag,
        "s3_request_id": s3_request_id,
        "version_id": version_id,
    }

def invoke_eta_calculate_async(
    user_id,
    request_id=None,
    user_selected_start_tn=None,
    user_selected_end_tn=None,
    s3_saved=None,
    reason_code=None,
    source="route_optimization_completed",
):
    """
    경로최적화 완료 후 ETA 계산 Lambda를 비동기로 호출한다.

    - InvocationType="Event" 이므로 Route Optimization 응답을 기다리지 않는다.
    - 호출 실패는 경로최적화 성공 여부에 영향을 주지 않고 로그/응답 meta에만 남긴다.
    - ETA Lambda는 user_id를 기준으로 DAAS 미배송 송장, 최신 route ordering, GIS apartment_flag를 다시 조인한다.
    """
    function_name = os.environ.get("ETA_CALCULATE_FUNCTION_NAME", "").strip()
    enabled = os.environ.get("ETA_INVOKE_ENABLED", "true").strip().lower() in {"1", "true", "yes", "y"}

    if not enabled:
        return {
            "enabled": False,
            "invoked": False,
            "reason": "ETA_INVOKE_DISABLED",
        }

    if not function_name:
        return {
            "enabled": True,
            "invoked": False,
            "reason": "ETA_CALCULATE_FUNCTION_NAME_EMPTY",
        }

    payload = {
        "user_id": int(user_id),
        "source": source,
        "route_request_id": request_id,
        "user_selected_start_tn": user_selected_start_tn,
        "user_selected_end_tn": user_selected_end_tn,
        "route_reason_code": reason_code,
        "route_s3": {
            "bucket": (s3_saved or {}).get("bucket"),
            "key": (s3_saved or {}).get("key"),
            "etag": (s3_saved or {}).get("etag"),
        } if s3_saved else None,
    }

    payload = {k: v for k, v in payload.items() if v is not None}

    resp = LAMBDA_CLIENT.invoke(
        FunctionName=function_name,
        InvocationType="Event",
        Payload=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
    )

    status_code = int(resp.get("StatusCode", 0) or 0)
    lambda_request_id = ((resp.get("ResponseMetadata") or {}).get("RequestId"))

    return {
        "enabled": True,
        "invoked": status_code in (202, 200),
        "function_name": function_name,
        "status_code": status_code,
        "lambda_request_id": lambda_request_id,
        "payload": payload,
    }


def _json_response(status_code: int, payload: dict, request_id: str | None = None):
    """
    모든 응답에 meta.request_id를 포함시키기 위한 wrapper
    - payload에 meta가 없으면 생성
    - meta가 있어도 request_id가 없으면 자동 삽입
    """
    if request_id:
        meta = payload.get("meta")
        if not isinstance(meta, dict):
            meta = {}
        meta.setdefault("request_id", request_id)
        payload["meta"] = meta

    # Postman 테스트 용도: CORS 헤더 없음
    return {
        "statusCode": status_code,
        "headers": {"Content-Type": "application/json"},
        "body": json.dumps(payload, ensure_ascii=False),
    }


def _bad_request(code: str, message: str, method: str, path: str, request_id: str | None):
    return _json_response(400, {
        "error": {"code": code, "message": message},
        "meta": {"api": {"httpMethod": method, "path": path}}
    }, request_id=request_id)

def _bad_request_with_input(code: str, message: str, method: str, path: str, request_id: str | None, input_obj: dict):
    return _json_response(400, {
        "error": {"code": code, "message": message},
        "meta": {
            "input": input_obj,
            "api": {"httpMethod": method, "path": path}
        }
    }, request_id=request_id)

def _bad_request_data(code: str, message: str, method: str, path: str, request_id: str | None, input_obj: dict, details: dict | None = None):
    payload = {
        "error": {"code": code, "message": message},
        "meta": {
            "input": input_obj,
            "api": {"httpMethod": method, "path": path},
        },
    }
    if details is not None:
        payload["error"]["details"] = details
    return _json_response(400, payload, request_id=request_id)


def _is_empty_df(df) -> bool:
    return (df is None) or (not isinstance(df, pd.DataFrame)) or df.empty

def _to_float_or_none(x):
    try:
        return float(x)
    except Exception:
        return None


def _collect_coordinate_issues(df: pd.DataFrame, sample_n: int = 20) -> list[dict]:
    """
    OSRM 호출 전에 좌표 이상 여부를 검사한다.

    검사 대상:
    - item 좌표: lat, lng
    - unit 좌표: unit_lat, unit_lng

    감지 유형:
    - NON_NUMERIC_COORD: 숫자로 변환 불가
    - OUT_OF_RANGE_COORD: 위경도 허용 범위 초과
    - LIKELY_SWAPPED_LAT_LNG: 한국 좌표 기준 lat/lng 뒤집힘 의심
      예) lat=127.08, lng=37.59
    """
    if df is None or not isinstance(df, pd.DataFrame) or df.empty:
        return []

    issues = []
    seen = set()

    coord_specs = [
        ("item", "lat", "lng"),
        ("unit", "unit_lat", "unit_lng"),
    ]

    for coord_type, lat_col, lng_col in coord_specs:
        if lat_col not in df.columns or lng_col not in df.columns:
            continue

        for idx, row in df.iterrows():
            lat = _to_float_or_none(row.get(lat_col))
            lng = _to_float_or_none(row.get(lng_col))
            reason = None

            if lat is None or lng is None:
                reason = "NON_NUMERIC_COORD"
            elif not (-90 <= lat <= 90) or not (-180 <= lng <= 180):
                reason = "OUT_OF_RANGE_COORD"
            # 한국 서비스 기준에서 뒤집힌 좌표 패턴 감지
            # 정상: lat 약 33~39, lng 약 124~132
            # 뒤집힘: lat 약 124~132, lng 약 33~39
            elif 124 <= lat <= 132 and 33 <= lng <= 39:
                reason = "LIKELY_SWAPPED_LAT_LNG"

            if reason is None:
                continue

            rec = {
                "row_index": int(idx),
                "coord_type": coord_type,
                "lat_col": lat_col,
                "lng_col": lng_col,
                "lat": lat,
                "lng": lng,
                "reason": reason,
            }

            if "tracking_number" in df.columns:
                rec["tracking_number"] = str(row.get("tracking_number"))
            if "Area" in df.columns:
                rec["Area"] = str(row.get("Area"))

            dedup_key = (
                coord_type,
                rec.get("tracking_number"),
                rec.get("Area"),
                lat,
                lng,
                reason,
            )
            if dedup_key in seen:
                continue

            seen.add(dedup_key)
            issues.append(rec)

            if len(issues) >= sample_n:
                return issues

    return issues

def _parse_body_only_rest_api_event(event: dict, method: str, path: str, request_id: str | None):
    """
    REST API(v1) 기준: body 파싱
    - body 없음/빈값/JSON 문법 오류를 명확히 400으로 반환
    """
    event = event or {}
    body = event.get("body")

    # body가 아예 없는 경우
    if body is None:
        return None, _bad_request(
            "MISSING_BODY",
            "request body is required",
            method, path, request_id
        )

    # base64 body 대응
    if event.get("isBase64Encoded") and isinstance(body, str) and body:
        try:
            body = base64.b64decode(body).decode("utf-8")
        except Exception:
            return None, _bad_request(
                "INVALID_BODY_ENCODING",
                "body is base64-encoded but could not be decoded",
                method, path, request_id
            )

    # 문자열인데 비어있거나 공백뿐인 경우
    if isinstance(body, str):
        if not body.strip():
            return None, _bad_request(
                "MISSING_BODY",
                "request body is required",
                method, path, request_id
            )
        try:
            obj = json.loads(body)
        except json.JSONDecodeError:
            return None, _bad_request(
                "INVALID_JSON_BODY",
                "Invalid JSON. Use null or \"\" for user_selected_start_tn/end_tn.",
                method, path, request_id
            )
        if not isinstance(obj, dict):
            return None, _bad_request(
                "INVALID_JSON_BODY",
                "JSON body must be an object",
                method, path, request_id
            )
        return obj, None

    # 콘솔 테스트 등에서 body가 dict로 들어오는 경우
    if isinstance(body, dict):
        return body, None

    # 그 외 타입은 비정상
    return None, _bad_request(
        "INVALID_JSON_BODY",
        "Invalid request body type",
        method, path, request_id
    )

def _get_method_path(event: dict):
    event = event or {}

    # REST API(v1)
    method = event.get("httpMethod")
    path = event.get("path")

    # HTTP API(v2) / Function URL
    if not method:
        method = (((event.get("requestContext") or {}).get("http") or {}).get("method")
                  or event.get("requestContext", {}).get("httpMethod"))
    if not path:
        path = (((event.get("requestContext") or {}).get("http") or {}).get("path")
                or event.get("rawPath")
                or event.get("path"))

    return (method or ""), (path or "")


def lambda_handler(event, context):
    req_t0 = time.perf_counter()

    method, path = _get_method_path(event)
    print(f"[lambda] method={method}, path={path}")

    ACCOUNT_ENV = os.environ.get("ACCOUNT_ENV", "beta")
    print(f"[lambda] ACCOUNT_ENV: {ACCOUNT_ENV}")

    request_id = None
    try:
        request_id = getattr(context, "aws_request_id", None)
    except Exception:
        request_id = None

    lambda_status = "started"
    user_id = None
    item_count = None
    n_nodes = None
    cache_hit = None
    s3_saved_ok = None
    eta_trigger = None

    try:
        payload, parse_err = _parse_body_only_rest_api_event(event, method, path, request_id)
        if parse_err:
            lambda_status = "bad_request"
            return parse_err

        def _norm_tn(x):
            if x is None:
                return None
            s = str(x).strip()
            return s if s else None

        user_id = payload.get("user_id")
        if user_id is None or str(user_id).strip() == "":
            lambda_status = "bad_request"
            return _bad_request("MISSING_USER_ID", "user_id is required", method, path, request_id)

        user_selected_start_tn = _norm_tn(payload.get("user_selected_start_tn"))
        user_selected_end_tn = _norm_tn(payload.get("user_selected_end_tn"))

        try:
            user_id = int(user_id)
        except Exception:
            lambda_status = "bad_request"
            return _bad_request("INVALID_USER_ID", "user_id must be an integer", method, path, request_id)

        if user_selected_start_tn is not None and user_selected_end_tn is not None:
            s = str(user_selected_start_tn).strip()
            e = str(user_selected_end_tn).strip()
            if s and e and s == e:
                lambda_status = "bad_request"
                return _bad_request_with_input(
                    "INVALID_SAME_START_END",
                    "Start and end tracking numbers must be different.",
                    method, path, request_id,
                    {
                        "user_id": user_id,
                        "user_selected_start_tn": user_selected_start_tn,
                        "user_selected_end_tn": user_selected_end_tn,
                    }
                )

        print("[lambda] payload parsed:", {
            "user_id": user_id,
            "user_selected_start_tn": user_selected_start_tn,
            "user_selected_end_tn": user_selected_end_tn,
        })

        print("[lambda] fetch_data begin")

        t0 = time.perf_counter()
        df_shipping = ITEM_QUERY.item_dataset_df(user_id)
        log_perf(
            "db_shipping_fetch",
            t0,
            request_id=request_id,
            user_id=user_id,
            rows=int(len(df_shipping)) if isinstance(df_shipping, pd.DataFrame) else None,
        )

        df_unit = get_cached_unit_df(UNIT_QUERY, request_id=request_id)

        input_obj = {
            "user_id": user_id,
            "user_selected_start_tn": user_selected_start_tn,
            "user_selected_end_tn": user_selected_end_tn,
        }

        if _is_empty_df(df_shipping):
            lambda_status = "bad_request"
            return _bad_request_data(
                code="NO_SHIPPING_DATA",
                message="No shipping items found for this user_id (df_shipping is empty).",
                method=method, path=path, request_id=request_id,
                input_obj=input_obj,
                details={"df_shipping_rows": 0}
            )

        tn_set = set(
            df_shipping["tracking_number"]
            .dropna()
            .astype(str)
            .str.strip()
        )
        tn_set.discard("")

        missing_start = (user_selected_start_tn is not None) and (str(user_selected_start_tn).strip() not in tn_set)
        missing_end = (user_selected_end_tn is not None) and (str(user_selected_end_tn).strip() not in tn_set)

        if missing_start and missing_end:
            lambda_status = "bad_request"
            return _bad_request_data(
                code="START_END_TN_NOT_FOUND",
                message="Both user_selected_start_tn and user_selected_end_tn were not found in df_shipping.tracking_number.",
                method=method, path=path, request_id=request_id,
                input_obj=input_obj,
                details={
                    "user_selected_start_tn": user_selected_start_tn,
                    "user_selected_end_tn": user_selected_end_tn,
                },
            )

        if missing_start:
            lambda_status = "bad_request"
            return _bad_request_data(
                code="START_TN_NOT_FOUND",
                message="user_selected_start_tn was not found in df_shipping.tracking_number.",
                method=method, path=path, request_id=request_id,
                input_obj=input_obj,
                details={"user_selected_start_tn": user_selected_start_tn},
            )

        if missing_end:
            lambda_status = "bad_request"
            return _bad_request_data(
                code="END_TN_NOT_FOUND",
                message="user_selected_end_tn was not found in df_shipping.tracking_number.",
                method=method, path=path, request_id=request_id,
                input_obj=input_obj,
                details={"user_selected_end_tn": user_selected_end_tn},
            )

        if _is_empty_df(df_unit):
            lambda_status = "bad_request"
            return _bad_request_data(
                code="NO_UNIT_DATA",
                message="No unit found (df_unit is empty).",
                method=method, path=path, request_id=request_id,
                input_obj=input_obj,
                details={"df_unit_rows": 0}
            )

        print("[lambda] fetch_data complete")
        print("[lambda] df_shipping rows:", int(len(df_shipping)))
        print("[lambda] df_unit rows:", int(len(df_unit)))

        t0 = time.perf_counter()
        df = pd.merge(df_shipping, df_unit, on="Area", how="inner")
        log_perf(
            "merge_shipping_unit",
            t0,
            request_id=request_id,
            user_id=user_id,
            rows=int(len(df)) if isinstance(df, pd.DataFrame) else None,
        )

        if _is_empty_df(df):
            left_areas = set(df_shipping["Area"].dropna().astype(str).unique()) if "Area" in df_shipping.columns else set()
            right_areas = set(df_unit["Area"].dropna().astype(str).unique()) if "Area" in df_unit.columns else set()
            common = len(left_areas.intersection(right_areas)) if left_areas and right_areas else 0

            lambda_status = "bad_request"
            return _bad_request_data(
                code="MERGE_EMPTY",
                message="Merge result is empty. Check join key mismatch on 'Area'.",
                method=method, path=path, request_id=request_id,
                input_obj=input_obj,
                details={
                    "df_shipping_rows": int(len(df_shipping)),
                    "df_unit_rows": int(len(df_unit)),
                    "join_key": "Area",
                    "shipping_unique_area": int(len(left_areas)) if left_areas else None,
                    "unit_unique_area": int(len(right_areas)) if right_areas else None,
                    "common_area_count": int(common),
                }
            )
        
        coord_issues = _collect_coordinate_issues(df, sample_n=20)
        if coord_issues:
            issue_reasons = {x["reason"] for x in coord_issues}

            if "LIKELY_SWAPPED_LAT_LNG" in issue_reasons:
                code = "SWAPPED_COORDINATES_DETECTED"
                message = "Detected likely swapped lat/lng coordinates before OSRM call."
            else:
                code = "INVALID_COORDINATES"
                message = "Invalid coordinates detected before OSRM call."

            print(f"[lambda] coordinate validation failed: {coord_issues}")

            lambda_status = "bad_request"
            return _bad_request_data(
                code=code,
                message=message,
                method=method,
                path=path,
                request_id=request_id,
                input_obj=input_obj,
                details={
                    "issue_count": len(coord_issues),
                    "issues": coord_issues,
                },
            )

        print(f"[lambda] merged df rows={int(len(df))}")

        t0 = time.perf_counter()
        matrix_json = transform_input_data_with_osrm_matrix(df, request_id=request_id)
        n_nodes = len(matrix_json["address_list"])
        log_perf(
            "osrm_matrix_total",
            t0,
            request_id=request_id,
            user_id=user_id,
            n_nodes=n_nodes,
        )

        if n_nodes <= 50:
            max_iters = 300
        elif n_nodes <= 80:
            max_iters = 400
        else:
            max_iters = 450
        print(f"[lambda] n_nodes={n_nodes}, max_iters={max_iters}")

        t0 = time.perf_counter()
        df_ordered, alns_meta = solve_alns_to_df_later_supernode(
            payload=matrix_json,
            matrix_key="dist_matrix",
            max_iters=max_iters,
            seed=None,
            use_cache=True,

            miss_enable_bestof_k=False,
            miss_bestof_k=8,
            miss_short_iters=1500,
            miss_refine=True,

            selected_start_tracking_number=user_selected_start_tn,
            selected_end_tracking_number=user_selected_end_tn,

            group_same_road_addr2=True,
            group_same_road_apartment=True,
            group_same_coords=True,
            group_same_road=True,

            return_meta=True,
        )
        cache_hit = bool((alns_meta or {}).get("cache_hit", False))
        log_perf(
            "alns_total",
            t0,
            request_id=request_id,
            user_id=user_id,
            rows=int(len(df_ordered)) if isinstance(df_ordered, pd.DataFrame) else None,
            cache_hit=cache_hit,
            solver_runtime_ms=(alns_meta or {}).get("runtime_ms"),
            max_iters=max_iters,
        )

        if isinstance(df_ordered, pd.DataFrame) and "apartment_flag" in df_ordered.columns:
            df_ordered = df_ordered.drop(columns=["apartment_flag"])

        print(f"[lambda] df_ordered rows={int(len(df_ordered))}")

        packed = pack_df(df_ordered, sample_n=500)

        if isinstance(df_ordered, pd.DataFrame) and "tracking_number" in df_ordered.columns:
            packed["item_count"] = int((df_ordered["tracking_number"] != "unit").sum())

        cache_hit = bool((alns_meta or {}).get("cache_hit", False))
        n_nodes = len((matrix_json or {}).get("address_list", []))
        item_count = None
        if isinstance(df_ordered, pd.DataFrame) and "tracking_number" in df_ordered.columns:
            item_count = int((df_ordered["tracking_number"] != "unit").sum())

        has_start = user_selected_start_tn is not None
        has_end = user_selected_end_tn is not None

        reason_code = "OPTIMIZED_ROUTE"
        reason = "Route optimized successfully."

        if cache_hit:
            reason_code = "OPTIMIZED_ROUTE_CACHE_HIT"
            reason = "Route optimized successfully (cache hit)."

        if has_start or has_end:
            reason_code = "OPTIMIZED_ROUTE_WITH_FIXED_POINTS"
            reason = "Route optimized successfully with user-selected start/end."

        s3_saved = None
        s3_saved_ok = None
        t0 = time.perf_counter()

        try:
            s3_saved = save_result_to_s3(
                request_id=request_id,
                user_id=user_id,
                user_selected_start_tn=user_selected_start_tn,
                user_selected_end_tn=user_selected_end_tn,
                method=method,
                path=path,
                reason_code=reason_code,
                reason=reason,
                packed=packed,
                alns_meta=alns_meta,
            )
            s3_saved_ok = bool(s3_saved)

            log_perf(
                "s3_save_result",
                t0,
                request_id=request_id,
                user_id=user_id,
                saved=s3_saved_ok,
                s3_key=(s3_saved or {}).get("key"),
                body_size=(s3_saved or {}).get("body_size"),
                etag=(s3_saved or {}).get("etag"),
                s3_request_id=(s3_saved or {}).get("s3_request_id"),
            )

        except Exception as s3e:
            s3_saved_ok = False
            print(json.dumps({
                "tag": "s3_verify",
                "step": "put_object_failed",
                "request_id": request_id,
                "user_id": user_id,
                "error": repr(s3e),
                "traceback": traceback.format_exc(),
            }, ensure_ascii=False))

        eta_t0 = time.perf_counter()
        try:
            eta_trigger = invoke_eta_calculate_async(
                user_id=user_id,
                request_id=request_id,
                user_selected_start_tn=user_selected_start_tn,
                user_selected_end_tn=user_selected_end_tn,
                s3_saved=s3_saved,
                reason_code=reason_code,
            )
            log_perf(
                "eta_async_invoke",
                eta_t0,
                request_id=request_id,
                user_id=user_id,
                eta_invoked=(eta_trigger or {}).get("invoked"),
                eta_function_name=(eta_trigger or {}).get("function_name"),
                eta_status_code=(eta_trigger or {}).get("status_code"),
                eta_reason=(eta_trigger or {}).get("reason"),
            )
        except Exception as etae:
            eta_trigger = {
                "enabled": True,
                "invoked": False,
                "reason": "ETA_ASYNC_INVOKE_FAILED",
                "error": repr(etae),
            }
            print(json.dumps({
                "tag": "eta_trigger",
                "step": "async_invoke_failed",
                "request_id": request_id,
                "user_id": user_id,
                "error": repr(etae),
                "traceback": traceback.format_exc(),
            }, ensure_ascii=False))
            log_perf(
                "eta_async_invoke",
                eta_t0,
                request_id=request_id,
                user_id=user_id,
                eta_invoked=False,
                eta_reason="ETA_ASYNC_INVOKE_FAILED",
            )

        lambda_status = "success"
        return _json_response(200, {
            "meta": {
                "input": {
                    "user_id": user_id,
                    "user_selected_start_tn": user_selected_start_tn,
                    "user_selected_end_tn": user_selected_end_tn,
                },
                "api": {
                    "httpMethod": method,
                    "path": path,
                },
                "cache": {
                    "hit": bool((alns_meta or {}).get("cache_hit", False)),
                    "key16": str((alns_meta or {}).get("key", ""))[:16],
                },
                "status": "OK",
                "reason_code": reason_code,
                "reason": reason,
                "eta_trigger": eta_trigger
            },
            "result": {
                "df_ordered": packed,
            }
        }, request_id=request_id)

    except Exception as e:
        lambda_status = "error"
        print(json.dumps({
            "tag": "lambda_error",
            "step": "unhandled_exception",
            "request_id": request_id,
            "user_id": user_id,
            "error": repr(e),
            "traceback": traceback.format_exc(),
        }, ensure_ascii=False))

        error_code = "INTERNAL_SERVER_ERROR"
        error_message = "Internal server error."


        return _json_response(500, {
            "error": {
                "code": error_code,
                "message": error_message,
            },
            "meta": {
                "api": {"httpMethod": method, "path": path},
            }
        }, request_id=request_id)

    finally:
        log_perf(
            "lambda_total",
            req_t0,
            request_id=request_id,
            status=lambda_status,
            user_id=user_id,
            item_count=item_count,
            n_nodes=n_nodes,
            cache_hit=cache_hit,
            s3_saved=s3_saved_ok,
        )