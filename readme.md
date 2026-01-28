# Route Optimization (OSRM + Python)

대용량 지도 데이터(OSM PBF)와 OSRM 로컬 라우팅 서버를 기반으로, 배송 지점(위경도)들의 **최적 경로(라우트)**를 생성하는 프로젝트입니다.

> ⚠️ 이 레포는 **대용량 파일을 GitHub에 올리지 않기 위해** `.gitignore`가 강하게 적용되어 있습니다.  
> 따라서 **클론만으로는 바로 실행되지 않으며**, 아래 “필수 데이터 준비” + “OSRM 전처리”를 먼저 해야 합니다.

---

## 목차
- [1. 프로젝트 개요](#1-프로젝트-개요)
- [2. 레포에 포함되지 않는 파일들](#2-레포에-포함되지-않는-파일들)
- [3. 디렉토리 구조](#3-디렉토리-구조)
- [4. 요구사항](#4-요구사항)
- [5. 설치](#5-설치)
- [6. 환경변수-env-설정](#6-환경변수-env-설정)
- [7. 필수 데이터 준비-git에-없음](#7-필수-데이터-준비-git에-없음)
- [8. OSRM 지도 데이터 전처리](#8-osrm-지도-데이터-전처리)
- [9. OSRM 서버 실행](#9-osrm-서버-실행)
- [10. 파이프라인 실행](#10-파이프라인-실행)
- [11. 결과물-output](#11-결과물-output)
- [12. 트러블슈팅](#12-트러블슈팅)
- [13. 전체 로직 문서](#13-전체-로직-문서)

---

## 1) 프로젝트 개요

- 입력: 배송지 위경도/주소 기반 데이터 + (필요 시) 그리드/폴리곤 데이터
- 라우팅: 로컬 OSRM 서버(`localhost`)를 통해 구간 거리/시간/geometry를 계산
- 최적화: Python 파이프라인에서 최적 경로 산출
- 출력: CSV / 지도 HTML 시각화 등

---

## 2) 레포에 포함되지 않는 파일들

`.gitignore`에 의해 아래 항목들은 Git에 포함되지 않습니다.

### 2.1 환경/로컬 설정
- `.env`
- `optivenv/`

### 2.2 OSRM 산출물 (로컬에서 생성)
- `tsp-route-optimization/osrm-data/*.osrm*`

### 2.3 대용량 입력 데이터 (로컬에서 준비)
- `tsp-route-optimization/osrm-data/*.osm.pbf`
- `seoul_poly/*.osm.pbf`
- `grid/*.feather`
- `grid/*.json`
- `grid/*.npz`
- `seoul_poly/*.shp`, `*.shx`, `*.dbf`, `*.prj`, `*.cpg`

### 2.4 실행 결과물 (로컬에서 생성)
- `tsp-route-optimization/osrm-data/history/`
- `opti_test_visualization/*.html`
- `opti_test_csv/*.csv`
- `grid_traffic_output/*.csv`

---

## 3) 디렉토리 구조

(핵심 폴더만 요약)

- `tsp-route-optimization/`
  - `osrm-data/` : OSRM 입력(PBF) 및 전처리 산출물(`*.osrm*`)
- `osrm/`
  - `car.lua` : OSRM profile (한국 실정 맞춤 수정본 등)
- `grid/` : 대용량 그리드 관련 파일(레포 미포함)
- `seoul_poly/` : 서울 폴리곤/쉐이프파일(레포 미포함)
- `opti_test_visualization/` : 지도 HTML 결과(레포 미포함)
- `opti_test_csv/` : CSV 결과(레포 미포함)

---

## 4) 요구사항

- Python 3.11.x
- Docker (OSRM 실행)
- (권장) 가상환경 (예: `optivenv/`)

## 5) 설치
```bash      
# (예시) 가상환경 생성
python -m venv optivenv
source optivenv/bin/activate  # mac/linux 
```


## 6) 의존성 설치
```bash 
pip install -r requirements.txt
```

## 7) 환경변수 (.env) 설정
.env는 Git에 포함되지 않습니다. 아래 중 한 위치(또는 프로젝트 구조에 맞는 위치)에 생성하세요.

프로젝트 루트: .env

## 8) 필수 데이터 준비 (Git에 없음)
클론 후 아래 데이터는 로컬에서 반드시 준비해야 합니다.

### 8.1 Grid 데이터 (요청/공유 필요)
위치: grid/

형태: *.feather, *.json, *.npz

### 8.2 서울 폴리곤 데이터 (요청/공유 필요)
위치: seoul_poly/

형태: Shapefile set (*.shp, *.shx, *.dbf, *.prj, *.cpg) 등

### 8.3 지도 원본 PBF (OSRM 입력)
우측 링크에서 `.osm.pbf` 파일을 다운로드하신 후 아래 폴더에 넣어주세요.
[south-korea-latest.osm.pbf](https://download.geofabrik.de/asia/south-korea.html)

위치: tsp-route-optimization/osrm-data/

파일 예시: south-korea-latest.osm.pbf

## 9) OSRM 전처리(최초 1회) - MLD 기반
아래 과정은 `south-korea-latest.osm.pbf`를 OSRM 라우팅 파일로 만드는 단계입니다.  
(이 단계가 완료되어야 `osrm-customize`, `osrm-datastore`, `osrm-routed`가 동작합니다)

### 9.1 osrm-extract (profile: car.lua)
```bash
docker run -t \
  -v "${PWD}/tsp-route-optimization/osrm-data:/data" \
  -v "${PWD}/osrm:/profiles" \
  ghcr.io/project-osrm/osrm-backend:v5.27.1 \
  osrm-extract -p /profiles/car.lua /data/south-korea-latest.osm.pbf
```

### 9.2 osrm-partition (MLD)
```bash
docker run -t \
  -v "${PWD}/tsp-route-optimization/osrm-data:/data" \
  ghcr.io/project-osrm/osrm-backend:v5.27.1 \
  osrm-partition /data/south-korea-latest.osrm
```

### 9.3 osrm-customize (MLD, 초기 Customize)
```bash 
docker run -t \
  -v "${PWD}/tsp-route-optimization/osrm-data:/data" \
  ghcr.io/project-osrm/osrm-backend:v5.27.1 \
  osrm-customize /data/south-korea-latest.osrm
```

### 9.4 산출물 확인
전처리 완료 후 아래 폴더에 *.osrm* 파일들이 생성되어야 합니다.

tsp-route-optimization/osrm-data/


## 10) 파이프라인 실행
Docker가 실행 중인 상태에서 Python 파이프라인을 실행합니다.

```bash
python app.py
```

## 11) 결과물 (Output)
결과물은 Git에 포함되지 않으며 로컬에 생성됩니다.

opti_test_visualization/ : 지도 시각화 HTML

opti_test_csv/ : 결과 CSV

grid_traffic_output/ : 부가 산출 CSV

tsp-route-optimization/osrm-data/history/ : 실행 히스토리(있는 경우)

## 12) 트러블슈팅
### 12.1 클론 후 바로 실행이 안 돼요
정상입니다. .gitignore로 인해 필수 대용량 입력/산출물이 레포에 없습니다.
아래 순서로 준비하세요.

grid/, seoul_poly/, south-korea-latest.osm.pbf 준비

osrm-extract 실행

osrm-partition/customize 또는 osrm-contract 실행

osrm-routed 실행

python app.py 실행

### 12.2 *.osrm 또는 *.osrm* 파일이 없다고 나와요
osrm-extract가 성공했는지 확인하세요.

그 다음 단계(MLD면 partition/customize, CH면 contract)를 수행했는지 확인하세요.

파일은 tsp-route-optimization/osrm-data/ 아래에 생성됩니다.

### 12.3 결과 파일이 Git에 안 올라가요
의도된 동작입니다. 결과물은 아래 패턴으로 Git에서 제외됩니다.

opti_test_visualization/*.html

opti_test_csv/*.csv

grid_traffic_output/*.csv

## 13) 전체 로직 문서
전체 파이프라인 상세 문서(Notion):

https://www.notion.so/delivus/Route_Optimizaition_pipeline-2e024414f57f80e68a82c294a9ed35b7?source=copy_link