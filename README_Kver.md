# Delivery Route Optimization Lambda

## AWS Lambda 기반 실시간 배송 경로 최적화 API

배송기사의 당일 미배송 물량을 조회하고, 실제 도로망 기반 거리/시간 행렬(OSRM)과 ALNS 최적화 알고리즘을 결합하여 **최적 방문 순서**를 자동 산출하는 서버리스 경로 최적화 프로젝트입니다.  
최적화 결과는 API 응답으로 반환되며, S3에 저장되어 Athena 분석 및 ETA Lambda 연동에 활용됩니다.

---

## Executive Impact

| Metric | Before | After | Impact |
|---|---:|---:|---|
| Average Response Time | 15.55 sec | 5.38 sec | **65% 개선** |
| 배송 순서 판단 | 기사 경험 의존 | 최적화 알고리즘 기반 | 운영 표준화 |
| 결과 추적 | 어려움 | S3/Athena 기반 추적 | 분석 가능 |
| ETA 연동 | 별도 구조 필요 | Lambda 비동기 연동 | downstream 확장 |

---

## Preview

<p align="center">
  <img src="./lambda/docs/images/app_route.png" width="320" alt="Route Optimization App Screenshot">
</p>

---

## Business Problem

기존 라스트마일 배송에서는 배송기사가 주소 목록을 직접 보고 경험적으로 방문 순서를 판단해야 했습니다.

- 배송 순서가 기사 숙련도에 크게 의존
- 비효율적인 이동 동선 발생
- 신규 기사에게 경로 판단 부담 증가
- 추천 경로 사용 여부를 데이터로 검증하기 어려움
- ETA 시스템과 연동 가능한 표준화된 방문 순서 데이터 필요

---

## Solution

배송기사 ID를 입력받아 당일 미배송 물량을 조회하고, 출차지와 배송지 좌표를 OSRM Table API로 변환한 뒤 ALNS 알고리즘으로 최적 방문 순서를 계산합니다.

```text
user_id 입력
      ↓
당일 미배송 물량 조회
      ↓
출차지 + 배송지 좌표 구성
      ↓
OSRM Table API 거리/시간 행렬 생성
      ↓
ALNS 방문 순서 최적화
      ↓
동일 주소/좌표 후처리
      ↓
S3 저장 및 API 응답
      ↓
ETA Lambda 비동기 호출
```

---

## My Role

### System Design

- AWS Lambda 기반 경로 최적화 API 설계
- Lambda Container Image 기반 배포 구조 구성
- S3 저장 구조 및 Athena 조회 구조 설계
- ETA Lambda와 비동기 연동 구조 설계

### Optimization Logic

- OSRM Table API 기반 거리/시간 행렬 생성
- ALNS 기반 방문 순서 최적화 로직 적용
- 시작점/종료점 고정 옵션 구현
- 동일 주소 및 동일 좌표 배송건 후처리
- 캐시 재사용 및 정수화 기반 성능 개선

### Operation

- CloudWatch 로그 기반 장애 추적
- OSRM timeout 및 connection error 대응
- Lambda cold start 및 응답속도 개선
- 운영 데이터 기반 추천 경로 검증 구조 구축

---

## Core Features

### 1. Driver-specific Delivery Query

`user_id`를 기준으로 해당 기사에게 할당된 당일 미배송 배송건을 조회합니다.

- 송장번호
- 도로명주소 / 상세주소
- 위도 / 경도
- 권역 및 섹터 정보
- 출차지 좌표

DB 접속 정보는 코드에 직접 저장하지 않고 AWS SSM Parameter Store를 통해 관리합니다.

### 2. OSRM Distance/Duration Matrix

실제 도로망 기반 이동 비용을 사용하기 위해 OSRM Table API를 호출합니다.

```text
출차지 + 배송지 목록
      ↓
OSRM Table API
      ↓
N x N distance/duration matrix
      ↓
ALNS input
```

### 3. ALNS Optimization

ALNS(Adaptive Large Neighborhood Search)를 적용하여 방문 순서를 최적화합니다.

- 출차지를 시작 노드로 설정
- 사용자가 지정한 시작 송장번호 고정
- 사용자가 지정한 종료 송장번호 고정
- 배송지 수에 따른 탐색 반복 횟수 조정
- 동일 주소/동일 좌표 배송건 후처리
- 캐시가 존재하는 경우 기존 최적화 결과 재사용

### 4. S3 Result Save & Athena Analysis

```text
s3://{bucket}/{prefix}/dt=YYYY-MM-DD/user_id={user_id}/request_id={request_id}.json
```

S3 결과는 Athena 외부 테이블 및 flatten view로 연결하여 분석할 수 있습니다.

- 기사별 추천 경로 생성 이력 확인
- 추천 순서와 실제 배송 완료 순서 비교
- API 호출량 및 request_id 추적
- ETA 계산 Lambda 연동 대상 검증

### 5. ETA Lambda Integration

경로 최적화 완료 후 ETA Lambda를 비동기로 호출할 수 있습니다.

```text
Route Optimization Lambda
      ↓
S3 Result Save
      ↓
ETA Calculate Lambda async invoke
      ↓
DynamoDB ETA Update
```

---

## Architecture

```text
Flex App / TMS
      ↓
Lambda Function URL or API Gateway
      ↓
Route Optimization Lambda
      ↓
MySQL Delivery Data Query
      ↓
OSRM Table API
      ↓
ALNS Optimization
      ↓
JSON Response
      ↓
S3 Result Save
      ↓
Athena Analysis / ETA Lambda
```

---

## API Specification

### Endpoint

```http
POST /route-opt
```

### Request Body

| Field | Type | Required | Description |
|---|---|---|---|
| `user_id` | integer | Y | 배송기사 사용자 ID |
| `user_selected_start_tn` | string/null | N | 시작 지점으로 고정할 송장번호 |
| `user_selected_end_tn` | string/null | N | 종료 지점으로 고정할 송장번호 |

### Example Request

```json
{
  "user_id": 26854,
  "user_selected_start_tn": null,
  "user_selected_end_tn": null
}
```

### Example Response

```json
{
  "success": true,
  "meta": {
    "status": "OK",
    "reason_code": "OPTIMIZED_ROUTE",
    "cache": {
      "hit": false
    },
    "s3": {
      "saved": true
    }
  },
  "result": {
    "df_ordered": {
      "columns": [
        "id",
        "tracking_number",
        "address_road",
        "lat",
        "lng",
        "ordering",
        "sub_order"
      ],
      "data": []
    }
  }
}
```

---

## Error Handling

| Error Code | Description |
|---|---|
| `MISSING_BODY` | 요청 body 누락 |
| `INVALID_JSON_BODY` | JSON 형식 오류 |
| `MISSING_USER_ID` | `user_id` 누락 |
| `INVALID_USER_ID` | `user_id` 타입 오류 |
| `INVALID_SAME_START_END` | 시작/종료 송장번호가 동일함 |
| `NO_SHIPPING_DATA` | 해당 기사에게 조회되는 배송 데이터 없음 |
| `START_TN_NOT_FOUND` | 시작 송장번호가 배송 목록에 없음 |
| `END_TN_NOT_FOUND` | 종료 송장번호가 배송 목록에 없음 |
| `MERGE_EMPTY` | 배송 데이터와 출차지 데이터 병합 실패 |
| `INVALID_COORDINATES` | 좌표값 비정상 |
| `SWAPPED_COORDINATES_DETECTED` | 위도/경도 뒤집힘 의심 |
| `INTERNAL_SERVER_ERROR` | 서버 내부 오류 |

---

## Tech Stack

| Category | Stack |
|---|---|
| Runtime | Python |
| Infra | AWS Lambda, AWS SAM, CloudFormation |
| Packaging | Docker, ECR |
| Database | MySQL |
| Storage | Amazon S3 |
| Analysis | Amazon Athena |
| Routing Engine | OSRM |
| Optimization | ALNS |
| Data Processing | Pandas, NumPy |
| Monitoring | CloudWatch Logs |

---

## Project Structure

```text
.
├── app.py
├── Dockerfile
├── template.yaml
├── requirements.txt
├── queries/
│   ├── item.py
│   └── unit.py
├── utils/
│   ├── db_handler.py
│   └── preprocess/
├── alns_later_supernode/
│   ├── api.py
│   ├── solver.py
│   ├── operators.py
│   ├── postprocess.py
│   ├── cache.py
│   └── payload.py
├── docs/
│   └── images/
└── events/
    └── example_route_opt.json
```

---

## Security / Redaction

포트폴리오 공개를 위해 다음 항목은 제거하거나 샘플 값으로 대체했습니다.

- 실제 AWS Account ID
- 실제 DB 접속 정보
- 실제 S3 Bucket 이름
- 실제 송장번호 / 사용자 ID
- 내부 테이블명 일부
- 운영 배포용 `samconfig.toml`

---

## Key Takeaway

> 기사 경험에 의존하던 배송 방문 순서 판단을 OSRM + ALNS 기반 경로 최적화 API로 전환하고,  
> S3/Athena/ETA Lambda와 연결 가능한 서버리스 운영 구조로 확장한 프로젝트입니다.
