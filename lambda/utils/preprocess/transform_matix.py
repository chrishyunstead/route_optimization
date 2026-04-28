import json
import time

import requests

OSRM_SESSION = requests.Session()


def _log_tm_perf(step, start_ts, request_id=None, **extra):
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


def transform_input_data(df):
    """
    df -> {"address_geocode_list":[...]} 생성
    """
    address_geocode_list = []

    address_geocode_list.append({
        "id": 1,
        "lat": float(df.iloc[0]["unit_lat"]),
        "lng": float(df.iloc[0]["unit_lng"]),
        "tracking_number": "unit",
    })

    for idx, row in enumerate(df.itertuples(index=False), start=2):
        address_geocode_list.append({
            "id": idx,
            "lat": float(row.lat),
            "lng": float(row.lng),
            "tracking_number": str(row.tracking_number),
            "Area": row.Area,
            "address_road": str(row.address_road),
            "address2": str(row.address2),
        })

    return {"address_geocode_list": address_geocode_list}


def build_osrm_matrix_payload(
    address_geocode_json: dict,
    osrm_base_url: str = "http://osrm.osrm.local:5000",
    timeout: int = 300,
    request_id: str | None = None,
):
    """
    {"address_geocode_list":[...]} -> OSRM Table API 호출 -> 최종 결과 JSON 생성

    return:
      {
        "dist_matrix": [[...], ...],
        "address_list": [...]
      }
    """
    address_list = address_geocode_json["address_geocode_list"]
    coords = ";".join([f'{a["lng"]},{a["lat"]}' for a in address_list])

    url = f"{osrm_base_url}/table/v1/driving/{coords}"
    params = {"annotations": "distance"}

    t0 = time.perf_counter()
    try:
        r = OSRM_SESSION.get(url, params=params, timeout=timeout)
        status_code = r.status_code
        r.raise_for_status()
        data = r.json()

        _log_tm_perf(
            "osrm_table_call",
            t0,
            request_id=request_id,
            n_nodes=len(address_list),
            status_code=status_code,
        )
    except Exception as e:
        _log_tm_perf(
            "osrm_table_call",
            t0,
            request_id=request_id,
            n_nodes=len(address_list),
            error=repr(e),
        )
        raise

    dist = data.get("distances")
    if dist is None:
        raise RuntimeError(f"OSRM table 응답에 distances가 없습니다. keys={list(data.keys())}")

    return {
        "dist_matrix": dist,
        "address_list": address_list,
    }


def transform_input_data_with_osrm_matrix(
    df,
    osrm_base_url: str = "http://osrm.osrm.local:5000",
    timeout: int = 300,
    request_id: str | None = None,
):
    """
    df -> address_geocode_list 생성 -> OSRM table 호출 -> 최종 JSON 리턴
    """
    total_t0 = time.perf_counter()

    t0 = time.perf_counter()
    payload = transform_input_data(df)
    _log_tm_perf(
        "osrm_payload_build",
        t0,
        request_id=request_id,
        rows=int(len(df)),
    )

    result = build_osrm_matrix_payload(
        payload,
        osrm_base_url=osrm_base_url,
        timeout=timeout,
        request_id=request_id,
    )

    _log_tm_perf(
        "osrm_matrix_transform_total",
        total_t0,
        request_id=request_id,
        n_nodes=len(result["address_list"]),
    )

    return result