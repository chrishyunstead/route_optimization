import os
import folium
from folium.features import DivIcon
import pandas as pd

def render_order_map_route_optimization(
    df_ordered,
    *,
    lat_col="lat",
    lng_col="lng",
    out_html=None,
    zoom_start=13,
    use_tooltip=True,
    use_popup=False,
):
    """
    df_ordered 시각화
    - tracking_number == 'unit' → 빨간 원 + 'U'
    - 나머지 배송지 → 하얀 원 + ordering 숫자
    """

    if df_ordered is None or len(df_ordered) == 0:
        print("[Folium] df_ordered is empty. skip.")
        return None

    if out_html is None:
        raise ValueError("out_html is required")

    out_html = out_html.replace(":", "-").replace(" ", "_")
    os.makedirs(os.path.dirname(out_html), exist_ok=True)

    df = df_ordered.copy()
    df[lat_col] = df[lat_col].astype(float)
    df[lng_col] = df[lng_col].astype(float)

    # ✅ ordering 기준 정렬 (unit은 보통 ordering NaN or 0 이라 맨 앞/뒤로 감)
    if "ordering" in df.columns:
        df = df.sort_values(
            ["ordering", "sub_order"],
            na_position="last"
        ).reset_index(drop=True)

    center_lat = float(df[lat_col].mean())
    center_lng = float(df[lng_col].mean())
    m = folium.Map(location=[center_lat, center_lng], zoom_start=zoom_start)

    bounds = df[[lat_col, lng_col]].values.tolist()

    for _, row in df.iterrows():
        lat = float(row[lat_col])
        lng = float(row[lng_col])

        tracking_number = row.get("tracking_number", "")
        address_road = row.get("address_road", "")
        apartment_flag = row.get("apartment_flag", "")
        ordering = row.get("ordering", "")
        sub_order = row.get("sub_order", "")

        # =========================
        # ✅ UNIT 마커
        # =========================
        if tracking_number == "unit":
            folium.Marker(
                location=[lat, lng],
                icon=DivIcon(
                    icon_size=(30, 30),
                    icon_anchor=(15, 15),
                    html="""
                    <div style="
                        background-color: #ff4d4d;
                        border: 2px solid black;
                        border-radius: 50%;
                        width: 30px;
                        height: 30px;
                        text-align: center;
                        line-height: 30px;
                        font-size: 14px;
                        font-weight: bold;
                        color: white;
                    ">U</div>
                    """
                ),
                tooltip="UNIT (출발지)",
            ).add_to(m)
            continue # UNIT 마커는 여기서 끝

        # =========================
        # 배송지 마커
        # =========================
        info_html = f"""
        <div style="font-size: 13px; line-height: 1.4; white-space: nowrap;">
            <b>tracking_number</b>: {tracking_number}<br/>
            <b>address_road</b>: {address_road}<br/>
            <b>lat</b>: {lat}<br/>
            <b>lng</b>: {lng}<br/>
            <b>apartment_flag</b>: {apartment_flag}<br/>
            <b>ordering</b>: {ordering}<br/>
            <b>sub_order</b>: {sub_order}<br/>
        </div>
        """

        tooltip = info_html if use_tooltip else None
        popup = folium.Popup(info_html, max_width=450) if use_popup else None

        folium.Marker(
            location=[lat, lng],
            icon=DivIcon(
                icon_size=(24, 24),
                icon_anchor=(12, 12),
                html=f"""
                <div style="
                    background-color: white;
                    border: 1px solid black;
                    border-radius: 50%;
                    width: 24px;
                    height: 24px;
                    text-align: center;
                    line-height: 24px;
                    font-size: 12px;
                    font-weight: bold;
                ">{ordering}</div>
                """
            ),
            tooltip=tooltip,
            popup=popup,
        ).add_to(m)

    if len(bounds) >= 2:
        m.fit_bounds(bounds, padding=(30, 30))

    m.save(out_html)
    print(f"[Folium] map saved: {out_html} (rows={len(df)})")
    return out_html

def render_raw_time_map(
    df_raw,
    *,
    lat_col="lat",
    lng_col="lng",
    ts_col="timestamp_delivery_complete",
    unit_lat_col="unit_lat",
    unit_lng_col="unit_lng",
    out_html=None,
    zoom_start=13,
    use_tooltip=True,
    use_popup=False,
):
    if df_raw is None or len(df_raw) == 0:
        print("[Folium] df_raw is empty. skip.")
        return None
    if out_html is None:
        raise ValueError("out_html is required")

    out_html = out_html.replace(":", "-").replace(" ", "_")
    os.makedirs(os.path.dirname(out_html), exist_ok=True)

    df = df_raw.copy()

    # -------------------------
    # 좌표 정리
    # -------------------------
    df[lat_col] = pd.to_numeric(df[lat_col], errors="coerce")
    df[lng_col] = pd.to_numeric(df[lng_col], errors="coerce")
    df = df.dropna(subset=[lat_col, lng_col]).reset_index(drop=True)

    if ts_col in df.columns:
        df[ts_col] = pd.to_datetime(df[ts_col], errors="coerce")
        df = df.sort_values([ts_col], na_position="last").reset_index(drop=True)

    # -------------------------
    # 지도 생성
    # -------------------------
    center_lat = float(df[lat_col].mean())
    center_lng = float(df[lng_col].mean())
    m = folium.Map(location=[center_lat, center_lng], zoom_start=zoom_start)

    bounds = df[[lat_col, lng_col]].values.tolist()

    # -------------------------
    # ✅ UNIT 마커 (1회)
    # -------------------------
    if unit_lat_col in df.columns and unit_lng_col in df.columns:
        unit_lat = pd.to_numeric(df.iloc[0][unit_lat_col], errors="coerce")
        unit_lng = pd.to_numeric(df.iloc[0][unit_lng_col], errors="coerce")

        if pd.notna(unit_lat) and pd.notna(unit_lng):
            folium.Marker(
                location=[unit_lat, unit_lng],
                icon=DivIcon(
                    icon_size=(30, 30),
                    icon_anchor=(15, 15),
                    html="""
                    <div style="
                        background-color: #ff4d4d;
                        border: 2px solid black;
                        border-radius: 50%;
                        width: 30px;
                        height: 30px;
                        text-align: center;
                        line-height: 30px;
                        font-size: 14px;
                        font-weight: bold;
                        color: white;
                    ">U</div>
                    """
                ),
                tooltip="UNIT (출발지)",
            ).add_to(m)

            bounds.append([unit_lat, unit_lng])

    # -------------------------
    # 배송지 마커
    # -------------------------
    for idx, row in df.iterrows():
        order_time = idx + 1
        lat = float(row[lat_col])
        lng = float(row[lng_col])

        info_html = f"""
        <div style="font-size:13px; line-height:1.4;">
            <b>time_order</b>: {order_time}<br/>
            <b>timestamp_delivery_complete</b>: {row.get(ts_col,"")}<br/>
            <b>tracking_number</b>: {row.get("tracking_number","")}<br/>
            <b>address_road</b>: {row.get("address_road","")}<br/>
            <b>apartment_flag</b>: {row.get("apartment_flag","")}<br/>
        </div>
        """

        tooltip = info_html if use_tooltip else None
        popup = folium.Popup(info_html, max_width=450) if use_popup else None

        folium.Marker(
            location=[lat, lng],
            icon=DivIcon(
                icon_size=(24, 24),
                icon_anchor=(12, 12),
                html=f"""
                <div style="
                    background-color: #4da3ff;
                    border: 1px solid black;
                    border-radius: 50%;
                    width: 24px;
                    height: 24px;
                    text-align: center;
                    line-height: 24px;
                    font-size: 12px;
                    font-weight: bold;
                    color: white;
                ">{order_time}</div>
                """
            ),
            tooltip=tooltip,
            popup=popup,
        ).add_to(m)

    if len(bounds) >= 2:
        m.fit_bounds(bounds, padding=(30, 30))

    m.save(out_html)
    print(f"[Folium] raw-time map saved: {out_html} (rows={len(df)})")
    return out_html
