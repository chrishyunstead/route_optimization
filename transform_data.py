# transform_data.py
import json
import requests


def transform_input_data(df):
    """
    df -> {"address_geocode_list":[...]} 생성
    """
    address_geocode_list = []

    # 1️⃣ id = 1 : unit 정보
    address_geocode_list.append({
        "id": 1,
        "lat": float(df.iloc[0]["unit_lat"]),
        "lng": float(df.iloc[0]["unit_lng"]),
        "tracking_number": "unit",
    })

    # 2️⃣ id = 2부터 : dataframe 순서대로
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


def build_osrm_matrix_payload(address_geocode_json: dict, osrm_base_url: str = "http://localhost:5050", timeout: int = 60):
    """
    {"address_geocode_list":[...]} -> OSRM Table API 호출 -> 최종 결과 JSON 생성

    return:
      {
        "dist_matrix": [[...], ...],   # meters
        "dur_matrix":  [[...], ...],   # seconds
        "address_list": [...]
      }
    """
    address_list = address_geocode_json["address_geocode_list"]

    # OSRM은 lng,lat 순서
    coords = ";".join([f'{a["lng"]},{a["lat"]}' for a in address_list])

    url = f"{osrm_base_url}/table/v1/driving/{coords}"
    params = {"annotations": "distance,duration"}

    r = requests.get(url, params=params, timeout=timeout)
    r.raise_for_status()
    data = r.json()

    dist = data.get("distances")
    dur = data.get("durations")

    if dist is None or dur is None:
        raise RuntimeError(f"OSRM table 응답에 distances/durations가 없습니다. keys={list(data.keys())}")

    return {
        "dist_matrix": dist,
        "dur_matrix": dur,
        "address_list": address_list
    }


def transform_input_data_with_osrm_matrix(df, osrm_base_url: str = "http://localhost:5050", timeout: int = 60):
    """
    df -> address_geocode_list 생성 -> OSRM table 호출 -> 최종 JSON 리턴
    """
    payload = transform_input_data(df)
    return build_osrm_matrix_payload(payload, osrm_base_url=osrm_base_url, timeout=timeout)
