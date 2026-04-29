# Delivery Route Optimization Lambda

> **실제 배송 운영환경에서 사용된 AWS Lambda 기반 경로 최적화 시스템**  
> 배송기사의 당일 미배송 물량을 조회하고, 실제 도로망 기반 거리행렬(OSRM) + 탐색 최적화(ALNS)를 통해 최적 방문 순서를 자동 산출하는 서버리스 프로젝트입니다.

---

# Preview

## Mobile App Result

![App Screenshot](./lambda/docs/images/app_route.png)

## Demo Video

<video src="./lambda/docs/demo.mp4" controls width="360"></video>

---

# Why This Project?

기존 배송 운영에서는 기사님이 직접 배송 순서를 판단하거나 경험적으로 움직이는 경우가 많았습니다.

- 배송 순서가 기사 숙련도에 따라 달라짐
- 비효율 동선 발생
- SLA 지연 가능성 증가
- 신규 기사 적응 어려움
- 운영팀 통제 어려움

이를 해결하기 위해 **배송 데이터를 기반으로 자동 경로 최적화 API**를 구축했습니다.

---

# Core Features

## 1. 기사별 당일 배송 물량 자동 조회
`user_id` 기준 당일 미배송 아이템 조회

## 2. 실제 도로망 기준 거리행렬 생성 (OSRM)

출차지 + 배송지 목록 → OSRM Table API → N x N 거리행렬

## 3. ALNS 기반 최적 방문 순서 계산

- 출차지 시작 고정
- 특정 송장 시작점 지정
- 특정 송장 종료점 지정
- 동일 주소 그룹 후처리

## 4. 결과 로그 자동 저장

S3 저장 및 Athena 조회 가능

## 5. ETA 시스템 연동

Route Optimization → S3 Save → ETA Lambda Trigger

---

# Architecture

Flex App / TMS → Lambda API → MySQL → OSRM → ALNS → JSON Response → S3 → ETA

---

# Performance Improvement

| 구분 | 개선 전 | 개선 후 |
|------|--------|--------|
| 평균 응답시간 | 15.55초 | 5.38초 |

- 클러스터링 시간 약 38% 절감
- 평균 이동거리 약 15% 개선
- SLA 약 3.4% 개선

---

# Tech Stack

Python / AWS Lambda / AWS SAM / Docker / ECR / MySQL / S3 / Athena / DynamoDB / OSRM / ALNS / Pandas / NumPy

---

# What I Did

- Lambda API 설계 및 운영 배포
- 경로 최적화 비즈니스 로직 구현
- OSRM + ALNS 연동
- S3 저장 구조 설계
- Athena 조회 환경 구축
- 속도 개선 및 캐싱 적용
- Cold Start 최소화
- Timeout 대응