# search_gis.py
from __future__ import annotations

from typing import Any, Dict, List, Set
import pandas as pd


class SearchProcessor:
    def __init__(self, pg_dbhandler):
        self.pg_dbhandler = pg_dbhandler

    # ----------------------------
    # (A) matrix_json에서 address_road 값 수집
    # ----------------------------
    def _collect_address_roads(self, matrix_json: Dict[str, Any]) -> List[str]:
        addr_list = matrix_json.get("address_list") or matrix_json.get("address_geocode_list")
        if not addr_list:
            raise KeyError("matrix_json must contain 'address_list' or 'address_geocode_list'.")

        roads: Set[str] = set()
        for a in addr_list:
            road = (a.get("address_road") or "").strip()
            if road:
                roads.add(road)

        return sorted(roads)

    # ----------------------------
    # (B) apartment_flag 조회 쿼리 생성
    #  - BOOL -> 1/0 변환은 SQL에서 처리
    # ----------------------------
    def _build_apartment_flag_query(self) -> str:
        return """
            SELECT
                a.address_road AS address_road,
                MAX(CASE WHEN upper(trim(b.apartment_flag)) IN ('Y','1','T') THEN 1 ELSE 0 END) AS apartment_flag
            FROM address.bldgbldg b
            JOIN address.addresses a
            ON b.building_management_number = a.building_management_number
            WHERE a.address_road = ANY(%s::text[])
            GROUP BY a.address_road;
        """.strip()

    # ----------------------------
    # (C) DB에서 apartment_flag 매핑 dict 생성
    # ----------------------------
    def fetch_apartment_flags(self, address_roads: List[str]) -> Dict[str, int]:
        if not address_roads:
            return {}

        query = self._build_apartment_flag_query()

        df: pd.DataFrame = self.pg_dbhandler.fetch_data("gis", query, params=(address_roads,))
        print(f"[apartment_flag] roads={len(address_roads)} df_none={df is None} df_rows={(0 if df is None else len(df))}")

        if df is None or df.empty:
            return {}

        # address_road -> apartment_flag(1/0)
        return dict(zip(df["address_road"], df["apartment_flag"]))

    # ----------------------------
    # (D) matrix_json에 apartment_flag(1/0) 주입
    # ----------------------------
    def attach_apartment_flag(self, matrix_json: Dict[str, Any], default_value: int = 0) -> Dict[str, Any]:
        addr_list_key = "address_list" if matrix_json.get("address_list") else "address_geocode_list"
        addr_list = matrix_json.get(addr_list_key) or []
        if not addr_list:
            return matrix_json

        address_roads = self._collect_address_roads(matrix_json)
        road_to_flag = self.fetch_apartment_flags(address_roads)

        # id별로 "각 딕셔너리 안에" 키 추가
        for a in addr_list:
            road = (a.get("address_road") or "").strip()
            a["apartment_flag"] = int(road_to_flag.get(road, default_value))

        matrix_json[addr_list_key] = addr_list
        return matrix_json
