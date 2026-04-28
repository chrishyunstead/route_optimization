# start_docker.py
# ============================================================
# OSRM MLD 컨테이너/서버 상태 보장 + 교통량 CSV 변경 감지 반영
#
# 핵심 개선점
# 1) ✅ "반영 성공 후"에만 해시를 저장 (실패했는데도 해시가 갱신되는 버그 방지)
# 2) ✅ ensure()에서 start_osrm_server 파라미터를 그대로 넘길 수 있게(start_kwargs)
# 3) ✅ 동시 실행 꼬임 방지 파일락(fcntl) 추가
# ============================================================

from __future__ import annotations

import hashlib
import subprocess
import time
from pathlib import Path
from typing import Optional

import requests

# 동시 실행 락 (macOS/Linux)
try:
    import fcntl  # type: ignore
except Exception:
    fcntl = None  # Windows 등에서는 None

STATE_DIR = Path(".osrm_state")
STATE_DIR.mkdir(exist_ok=True)

HASH_FILE = STATE_DIR / "updates_seoul_midpoint_100m.csv.sha256"
LOCK_FILE = STATE_DIR / "osrm.lock"


# -------------------------
# Shell helper
# -------------------------
def run(cmd: str, capture_output: bool = False) -> str | None:
    """
    shell=True로 실행. 실패 시 예외 발생(subprocess.CalledProcessError).
    """
    result = subprocess.run(
        cmd,
        shell=True,
        executable="/bin/bash",
        check=True,
        stdout=subprocess.PIPE if capture_output else None,
        stderr=subprocess.PIPE if capture_output else None,
        text=True,
    )
    return result.stdout.strip() if capture_output else None


# -------------------------
# Lock helper
# -------------------------
def acquire_lock() -> Optional[object]:
    """
    같은 머신에서 동시에 ensure가 실행되면 docker rm/exec가 꼬일 수 있어 락을 건다.
    macOS/Linux: fcntl flock 사용
    """
    if fcntl is None:
        return None
    fp = open(LOCK_FILE, "w")
    fcntl.flock(fp, fcntl.LOCK_EX)
    return fp


# -------------------------
# Docker & OSRM health
# -------------------------
def start_docker_mac():
    """
    macOS에서 Docker Desktop이 꺼져있을 때 켜고 docker info가 될 때까지 대기.
    """
    subprocess.run(["open", "-a", "Docker"], check=True)
    for _ in range(30):
        try:
            subprocess.run(
                ["docker", "info"],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                check=True,
            )
            print("Docker 실행 완료")
            return
        except subprocess.CalledProcessError:
            time.sleep(2)
    raise RuntimeError("Docker 실행 실패 (docker info가 끝내 응답하지 않음)")


def is_container_running(name: str) -> bool:
    out = run(
        f'docker ps --filter "name=^{name}$" --format "{{{{.Names}}}}"',
        capture_output=True,
    )
    return out == name


def osrm_healthcheck(osrm_base_url: str = "http://localhost:5050", timeout: int = 2) -> bool:
    """
    /route 한 번 쏴서 200 + JSON code=Ok 확인
    (좌표는 아무거나, 한국 내 좌표로 고정)
    """
    url = f"{osrm_base_url}/route/v1/driving/127.04497,37.47287;127.04342,37.49374?overview=false"
    try:
        r = requests.get(url, timeout=timeout)
        if r.status_code != 200:
            return False
        j = r.json()
        return j.get("code") == "Ok"
    except Exception:
        return False


def wait_for_osrm(osrm_base_url: str = "http://localhost:5050", max_wait_sec: int = 30) -> bool:
    """
    서버 뜰 때까지 짧게 폴링
    """
    t0 = time.time()
    while time.time() - t0 < max_wait_sec:
        if osrm_healthcheck(osrm_base_url=osrm_base_url):
            return True
        time.sleep(1)
    return False


def restart_container(name: str = "osrm-mld-ab"):
    run(f"docker restart {name}")


# -------------------------
# Hash helpers (중요: 성공 후에만 기록)
# -------------------------
def sha256_file(path: str) -> str:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"CSV 파일이 존재하지 않습니다: {path}")

    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def read_last_hash() -> Optional[str]:
    if HASH_FILE.exists():
        v = HASH_FILE.read_text().strip()
        return v or None
    return None


def write_last_hash(h: str) -> None:
    HASH_FILE.write_text(h)


# -------------------------
# OSRM start / reload
# -------------------------
def start_osrm_server(
    data_dir: str = '${PWD}/tsp-route-optimization/osrm-data',
    osrm_file: str = "/data/south-korea-latest.osrm",
    speed_csv: str = "/data/updates_seoul_midpoint_100m.csv",
    image: str = "ghcr.io/project-osrm/osrm-backend:v5.27.1",
    name: str = "osrm-mld-ab",
    port_host: int = 5050,
    port_container: int = 5000,
    max_table_size: int = 200,
    shm_size: str = "2g",
    dataset: str = "osrm-region",
):
    """
    컨테이너 재생성 + (customize 반영) + routed 기동
    """
    data_vol = f'-v "{data_dir}:/data"'

    # 기존 컨테이너 제거
    run(f'docker rm -f {name} >/dev/null 2>&1 || true')

    # CSV 반영 (customize)
    run(
        f'docker run --rm {data_vol} {image} '
        f'osrm-customize {osrm_file} --segment-speed-file {speed_csv}'
    )

    # 서버 실행 (datastore + routed)
    run(
        f'docker run -d --name {name} -p {port_host}:{port_container} --shm-size={shm_size} {data_vol} {image} '
        f'sh -lc "osrm-datastore --dataset-name {dataset} {osrm_file} '
        f'&& exec osrm-routed --algorithm mld --shared-memory --dataset-name {dataset} --max-table-size {max_table_size}"'
    )

    print("OSRM 서버 새로 기동 완료")


def reload_traffic_in_running_container(
    name: str = "osrm-mld-ab",
    osrm_file: str = "/data/south-korea-latest.osrm",
    speed_csv: str = "/data/updates_seoul_midpoint_100m.csv",
    dataset: str = "osrm-region",
):
    """
    실행 중 컨테이너에 교통량 CSV 반영:
    - osrm-customize
    - osrm-datastore (shared-memory dataset 재적재)
    """
    run(
        f'docker exec {name} osrm-customize {osrm_file} '
        f'--segment-speed-file {speed_csv}'
    )
    run(
        f'docker exec {name} osrm-datastore --dataset-name {dataset} {osrm_file}'
    )
    print("교통량 CSV 변경 반영 완료 (customize + datastore)")


# -------------------------
# Main: ensure + csv change
# -------------------------
def ensure_osrm_running_with_csv_check(
    container_name: str = "osrm-mld-ab",
    local_csv_path: str = "./tsp-route-optimization/osrm-data/updates_seoul_midpoint_100m.csv",
    osrm_base_url: str = "http://localhost:5050",
    healthcheck_wait_sec: int = 30,
    speed_csv_in_container: str = "/data/updates_seoul_midpoint_100m.csv",
    start_kwargs: Optional[dict] = None,
):
    """
    1) 컨테이너 실행/응답 상태 점검
       - 실행 중이고 응답 OK면 그대로
       - 실행 중이나 응답 X면 restart → 실패 시 재생성
       - 실행 중 아님이면 Docker 켜고 재생성
    2) 로컬 CSV 해시 비교로 변경 감지
    3) 변경 시:
       - 새로 부팅했으면 start_osrm_server에서 이미 customize+datastore 했으므로 skip
       - 기존 실행 컨테이너면 customize+datastore 수행
    4) ✅ 반영 성공 후에만 해시 저장
    """
    lock_fp = acquire_lock()
    try:
        start_kwargs = start_kwargs or {}

        did_boot = False

        # --- 0) 변경 여부 계산 (단, 해시 기록은 "성공 후"에만!) ---
        new_hash = sha256_file(local_csv_path)
        old_hash = read_last_hash()
        changed = (old_hash != new_hash)

        # --- 1) 컨테이너/서버 상태 보장 ---
        running = is_container_running(container_name)

        if running:
            if not osrm_healthcheck(osrm_base_url=osrm_base_url):
                print(f"OSRM 컨테이너 '{container_name}'는 실행 중이나 서버 응답 없음 → 재시작 시도")
                restart_container(container_name)
                if not wait_for_osrm(osrm_base_url=osrm_base_url, max_wait_sec=healthcheck_wait_sec):
                    print("재시작 후에도 health check 실패 → 컨테이너 재생성으로 복구")
                    start_docker_mac()
                    start_osrm_server(name=container_name, **start_kwargs)
                    did_boot = True
                    if not wait_for_osrm(osrm_base_url=osrm_base_url, max_wait_sec=healthcheck_wait_sec):
                        raise RuntimeError("OSRM 서버 복구 실패(재생성 후에도 health check 불가)")
        else:
            start_docker_mac()
            start_osrm_server(name=container_name, **start_kwargs)
            did_boot = True
            if not wait_for_osrm(osrm_base_url=osrm_base_url, max_wait_sec=healthcheck_wait_sec):
                raise RuntimeError("OSRM 서버 기동 후 health check 실패")

        # --- 2) CSV 변경 없으면 종료 ---
        if not changed:
            print("교통량 변화가 없습니다 기존 경로로 안내합니다")
            return

        print("교통량 CSV 변경 감지 → 반영 수행")

        # --- 3) 방금 부팅했으면 start_osrm_server에서 이미 반영됨 ---
        if did_boot:
            # ✅ 부팅 성공 상태에서만 해시 기록
            write_last_hash(new_hash)
            return

        # --- 4) 실행 중 컨테이너에 반영 (실패 시 복구) ---
        try:
            reload_traffic_in_running_container(
                name=container_name,
                speed_csv=speed_csv_in_container,
                # 필요시 start_kwargs에서 osrm_file/dataset을 받아 동기화해도 됨:
                osrm_file=start_kwargs.get("osrm_file", "/data/south-korea-latest.osrm"),
                dataset=start_kwargs.get("dataset", "osrm-region"),
            )
        except Exception as e:
            print(f"customize/datastore 반영 중 오류 → 재기동으로 복구합니다: {e}")
            restart_container(container_name)

        # --- 5) 안정화 체크 ---
        if not wait_for_osrm(osrm_base_url=osrm_base_url, max_wait_sec=healthcheck_wait_sec):
            print("반영 후 health check 실패 → 짧게 재기동하여 안정화")
            restart_container(container_name)
            if not wait_for_osrm(osrm_base_url=osrm_base_url, max_wait_sec=healthcheck_wait_sec):
                print("재기동으로도 복구 안됨 → 컨테이너 재생성으로 복구")
                start_osrm_server(name=container_name, **start_kwargs)
                if not wait_for_osrm(osrm_base_url=osrm_base_url, max_wait_sec=healthcheck_wait_sec):
                    raise RuntimeError("OSRM 서버 복구 실패(반영 이후)")

        # ✅ 여기까지 오면 '반영 및 서버 정상'이므로 해시 기록
        write_last_hash(new_hash)

    finally:
        if lock_fp is not None:
            try:
                lock_fp.close()
            except Exception:
                pass