import pandas as pd
import os
from datetime import timedelta, datetime
from data_load import AutoContainerGeneration
from transform_data import transform_input_data_with_osrm_matrix
# from start_docker import ensure_osrm_running_with_csv_check
from start_docker_os import ensure_osrm_running_with_csv_check
from traffic_pipeline import run_traffic_to_osrm_csv
# from alns_pipeline import solve_alns_to_df
from db_handler_pg import PGDBHandler
from search_gis import SearchProcessor
from test_visualize import render_order_map_route_optimization, render_raw_time_map
from alns_pipeline_later_supernode import solve_alns_to_df_later_supernode
from alns_pipeline_later_supernode import eval_alns_metrics


# 1) 트래픽 데이터 -> segment-speed CSV 생성 (updates_seoul_midpoint_100m.csv)
snapshot_path, segment_csv_path, meta, now_str = run_traffic_to_osrm_csv(debug=True)
print("snapshot:", snapshot_path)
print("segment:", segment_csv_path)
print("now_str:", now_str)
print("OSRM segment-speed CSV 생성 완료:", segment_csv_path)
print("시간:", now_str)

# 2) OSRM 서버 ensure (컨테이너/헬스체크/CSV변경 시 반영)
ensure_osrm_running_with_csv_check(
    start_kwargs={
        "data_dir": "tsp-route-optimization/osrm-data",
        "dataset": "osrm-region",
        "max_table_size": 200,
        "shm_size": "2g",
        "port_host": 5050,
    }
)
print("OSRM 서버 실행/점검 완료")

# 3) 데이터 로딩
generator = AutoContainerGeneration(version=2, debug=True)
area_cluster, unit = generator.fetch_all_data()

print(f"Cluster 생성완료=> {area_cluster}")
print(f"Unit 생성완료=> {unit}")

# 4) 데이터 Merge
df = pd.merge(area_cluster, unit, on="Area", how="inner")

# 배송완료 건 제외 (필요 시)
# df = df[df["timestamp_delivery_complete"].isna()].reset_index(drop=True)

print(f"병합된 df=>\n{df.head()}")

# 5) OSRM 매트릭스 변환
matrix_json = transform_input_data_with_osrm_matrix(df)
print("OSRM 매트릭스 생성완료")

# 6) Apartment_flag 검증
# print("Apartment_flag 검증시작")
# pg_dbhandler = PGDBHandler()
# search_processor = SearchProcessor(pg_dbhandler)
# matrix_json = search_processor.attach_apartment_flag(matrix_json)

# DB 핸들러 종료
# pg_dbhandler.close()

# flag 검증
# flags = [a.get("apartment_flag") for a in matrix_json.get("address_list", [])]
# print("[apartment_flag] value_counts:", pd.Series(flags).value_counts(dropna=False).to_dict())

# 7) ALNS 파이프라인
# 사용자 선택 (tracking_number)
user_selected_start_tn = None  # 예: '261803950931' or None => 배포 시, 변수로 설정
user_selected_end_tn = None    # 예: '261803950931' or None => 배포 시, 변수로 설정

df_ordered = solve_alns_to_df_later_supernode(
    payload=matrix_json,
    matrix_key="dist_matrix",
    max_iters=5000,
    seed=None,
    use_cache=True,

    miss_enable_bestof_k=False, # best-of-k OFF (켜져 있을때 계산량 증가 우려...) / seed 후보 여러 개
    miss_bestof_k=8,
    miss_short_iters=1500,
    miss_refine=True,

    # 사용자 선택
    selected_start_tracking_number=user_selected_start_tn,  
    selected_end_tracking_number=user_selected_end_tn,     

    group_same_road_addr2=True,
    group_same_road_apartment=True,
    group_same_coords=True,
    group_same_road=True,
)
print(f'df_ordered:\n{df_ordered}')

metrics = eval_alns_metrics(payload=matrix_json, start_id=1, max_iters=5000)
print(f'초기해: {metrics["init_cost"]}, 최적해: {metrics["best_cost"]}, 개선율(%): {metrics["improve_pct"]}, 실행시간: {metrics["runtime_ms"]}')
print(f'초기해 개선 파악: init_order==best_order ?! => {metrics["init_order"] == metrics["best_order"]}')
print(f'초기해: {metrics["init_order"][:]}')
print(f'최적해: {metrics["best_order"][:]}')

# 8) 결과 저장 및 시각화 옵션 설정
def _env_bool(name: str, default: bool = False) -> bool:
    v = os.getenv(name)
    if v is None:
        return default
    return v.strip().lower() in ("1", "true", "t", "yes", "y", "on")

def _env_str(name: str, default: str) -> str:
    v = os.getenv(name)
    return default if v is None or not v.strip() else v.strip()

# -----------------------------------------
# 배포/운영용 토글
# - 로컬 테스트: ENABLE_SAVE_CSV=1, ENABLE_VISUALIZE=1
# - 베타/프로덕션: 기본 False로 두고 필요할 때만 켬
# -----------------------------------------
ENABLE_SAVE_CSV   = _env_bool("ENABLE_SAVE_CSV", default=False)
ENABLE_VISUALIZE  = _env_bool("ENABLE_VISUALIZE", default=False)

OUT_CSV_DIR = _env_str("OUT_CSV_DIR", "./opti_test_csv")
OUT_MAP_DIR = _env_str("OUT_MAP_DIR", "./opti_test_visualization")

dt = datetime.now()
bucket_minute = (dt.minute // 5) * 5
bucket_time = dt.replace(minute=bucket_minute, second=0, microsecond=0)
bucket_time = bucket_time.strftime("%Y-%m-%d_%H-%M-00")

# CSV 저장 (선택 사항)
if ENABLE_SAVE_CSV:
    os.makedirs(OUT_CSV_DIR, exist_ok=True)
    out_csv = f"{OUT_CSV_DIR}/{bucket_time}_ordered_result.csv"
    df_ordered.to_csv(out_csv, index=False, encoding="utf-8")
    print(f"[CSV] saved: {out_csv} (rows={len(df_ordered)})")
else:
    print("[CSV] saving disabled (ENABLE_SAVE_CSV=0). skip.")

# 시각화 저장 (선택 사항)
if ENABLE_VISUALIZE:
    os.makedirs(OUT_MAP_DIR, exist_ok=True)
    render_order_map_route_optimization( df_ordered, out_html=f"./{OUT_MAP_DIR}/{bucket_time}_ordered_map.html", use_tooltip=True, use_popup=False, )
    render_raw_time_map( df, out_html=f"./{OUT_MAP_DIR}/{bucket_time}_raw_time_map.html", use_tooltip=True, )
else:
    print("[Folium] visualization disabled (ENABLE_VISUALIZE=0). skip.")
