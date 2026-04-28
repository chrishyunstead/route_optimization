# start_docker.py
# ============================================================
# OSRM MLD 컨테이너/서버 상태 보장 + 교통량 CSV 변경 감지 반영
# - mac / windows 공용
#
# 핵심 개선점
# 1) "반영 성공 후"에만 해시 저장 (실패했는데도 해시 갱신되는 버그 방지)
# 2) ensure()에서 start_osrm_server 파라미터를 그대로 넘김(start_kwargs)
# 3) 동시 실행 꼬임 방지 파일락: mac/linux(fcntl) + windows(msvcrt)
# 4) shell=True, /bin/bash, ${PWD}, bash-ism(>/dev/null || true) 제거
# ============================================================

from __future__ import annotations

import hashlib
import os
import sys
import subprocess
import time
from pathlib import Path
from typing import Optional, Dict, Any, ContextManager
from contextlib import contextmanager

import requests


# -------------------------
# Platform flags
# -------------------------
IS_WINDOWS = os.name == "nt"
IS_MAC = sys.platform == "darwin"
IS_LINUX = sys.platform.startswith("linux")


# -------------------------
# State paths
# -------------------------
STATE_DIR = Path(".osrm_state")
STATE_DIR.mkdir(exist_ok=True)

HASH_FILE = STATE_DIR / "updates_total_midpoint_100m.csv.sha256"
LOCK_FILE = STATE_DIR / "osrm.lock"


# -------------------------
# Path helpers
# -------------------------
def resolve_path(p: str | Path) -> Path:
    """
    - '${PWD}/...' 같은 bash 스타일 제거/치환
    - '~' 확장
    - 상대경로면 CWD 기준으로 resolve
    """
    s = str(p)
    if "${PWD}" in s:
        s = s.replace("${PWD}", str(Path.cwd()))
    s = os.path.expanduser(s)
    return Path(s).resolve()


def docker_host_path_str(p: Path) -> str:
    """
    docker -v 마운트용 host path 문자열
    - Windows: 백슬래시 -> 슬래시로 변환 (Docker Desktop에서 호환성이 더 좋음)
    """
    s = str(p)
    if IS_WINDOWS:
        s = s.replace("\\", "/")
    return s


# -------------------------
# Subprocess helpers (NO shell)
# -------------------------
def run_cmd(args: list[str], *, capture_output: bool = False, check: bool = True) -> str | None:
    """
    OS 무관하게 동작하도록 shell=False + list args 강제
    """
    r = subprocess.run(
        args,
        check=check,
        stdout=subprocess.PIPE if capture_output else None,
        stderr=subprocess.PIPE if capture_output else None,
        text=True,
    )
    return r.stdout.strip() if capture_output else None


def docker_info_ok() -> bool:
    try:
        subprocess.run(
            ["docker", "info"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=True,
        )
        return True
    except Exception:
        return False


def start_docker_desktop_if_needed(max_wait_sec: int = 60) -> None:
    """
    Docker Desktop이 꺼져있으면 켜고 docker info 응답할 때까지 대기.
    - mac: open -a Docker
    - windows: Docker Desktop.exe 후보 경로에서 실행
    """
    if docker_info_ok():
        return

    if IS_MAC:
        subprocess.run(["open", "-a", "Docker"], check=False)

    elif IS_WINDOWS:
        candidates = [
            Path(os.environ.get("ProgramFiles", "")) / "Docker" / "Docker" / "Docker Desktop.exe",
            Path(os.environ.get("ProgramFiles(x86)", "")) / "Docker" / "Docker" / "Docker Desktop.exe",
            Path(os.environ.get("LocalAppData", "")) / "Programs" / "Docker" / "Docker Desktop.exe",
        ]
        exe = next((p for p in candidates if p.exists()), None)
        if exe:
            subprocess.run([str(exe)], check=False)

    deadline = time.time() + max_wait_sec
    while time.time() < deadline:
        if docker_info_ok():
            print("Docker 실행 완료")
            return
        time.sleep(2)

    raise RuntimeError("Docker가 실행되지 않았습니다. Docker Desktop을 켠 뒤 다시 실행해 주세요.")


def docker_rm_force(name: str) -> None:
    """
    bash의 `docker rm -f name >/dev/null 2>&1 || true` 대체
    """
    subprocess.run(
        ["docker", "rm", "-f", name],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        check=False,
    )


# -------------------------
# Cross-platform file lock
# -------------------------
@contextmanager
def cross_platform_file_lock(lock_path: Path) -> ContextManager[Optional[object]]:
    """
    mac/linux: fcntl
    windows: msvcrt
    """
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    fp = open(lock_path, "w")

    if IS_WINDOWS:
        import msvcrt  # type: ignore

        msvcrt.locking(fp.fileno(), msvcrt.LK_LOCK, 1)
        try:
            yield fp
        finally:
            try:
                msvcrt.locking(fp.fileno(), msvcrt.LK_UNLCK, 1)
            finally:
                fp.close()
    else:
        import fcntl  # type: ignore

        fcntl.flock(fp, fcntl.LOCK_EX)
        try:
            yield fp
        finally:
            try:
                fcntl.flock(fp, fcntl.LOCK_UN)
            finally:
                fp.close()


# -------------------------
# Docker & OSRM health
# -------------------------
def is_container_running(name: str) -> bool:
    out = run_cmd(
        ["docker", "ps", "--filter", f"name=^{name}$", "--format", "{{.Names}}"],
        capture_output=True,
        check=True,
    )
    return out == name


def osrm_healthcheck(osrm_base_url: str = "http://localhost:5050", timeout: int = 2) -> bool:
    """
    /route 한 번 쏴서 200 + JSON code=Ok 확인
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
    t0 = time.time()
    while time.time() - t0 < max_wait_sec:
        if osrm_healthcheck(osrm_base_url=osrm_base_url):
            return True
        time.sleep(1)
    return False


def restart_container(name: str) -> None:
    run_cmd(["docker", "restart", name], capture_output=False, check=False)


# -------------------------
# Hash helpers (성공 후에만 기록)
# -------------------------
def sha256_file(path: str | Path) -> str:
    p = resolve_path(path)
    if not p.exists():
        raise FileNotFoundError(f"CSV 파일이 존재하지 않습니다: {p}")

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
    data_dir: str = "tsp-route-optimization/osrm-data",
    osrm_file: str = "/data/south-korea-latest.osrm",
    speed_csv: str = "/data/updates_total_midpoint_100m.csv",
    image: str = "ghcr.io/project-osrm/osrm-backend:v5.27.1",
    name: str = "osrm-mld-ab",
    port_host: int = 5050,
    port_container: int = 5000,
    max_table_size: int = 200,
    shm_size: str = "2g",
    dataset: str = "osrm-region",
) -> None:
    """
    컨테이너 재생성 + (customize 반영) + routed 기동
    """
    host_data_dir = resolve_path(data_dir)
    mount = f"{docker_host_path_str(host_data_dir)}:/data"

    # 기존 컨테이너 제거
    docker_rm_force(name)

    # CSV 반영 (customize)
    run_cmd(
        [
            "docker",
            "run",
            "--rm",
            "-v",
            mount,
            image,
            "osrm-customize",
            osrm_file,
            "--segment-speed-file",
            speed_csv,
        ],
        check=True,
    )

    # 서버 실행 (datastore + routed)
    # (이 쉘은 컨테이너 내부 sh 이므로 OS 영향 없음)
    run_cmd(
        [
            "docker",
            "run",
            "-d",
            "--name",
            name,
            "-p",
            f"{port_host}:{port_container}",
            "--shm-size",
            shm_size,
            "-v",
            mount,
            image,
            "sh",
            "-lc",
            (
                f"osrm-datastore --dataset-name {dataset} {osrm_file} "
                f"&& exec osrm-routed --algorithm mld --shared-memory "
                f"--dataset-name {dataset} --max-table-size {max_table_size}"
            ),
        ],
        check=True,
    )

    print("OSRM 서버 새로 기동 완료")


def reload_traffic_in_running_container(
    name: str = "osrm-mld-ab",
    osrm_file: str = "/data/south-korea-latest.osrm",
    speed_csv: str = "/data/updates_total_midpoint_100m.csv",
    dataset: str = "osrm-region",
) -> None:
    """
    실행 중 컨테이너에 교통량 CSV 반영:
    - osrm-customize
    - osrm-datastore (shared-memory dataset 재적재)
    """
    run_cmd(
        ["docker", "exec", name, "osrm-customize", osrm_file, "--segment-speed-file", speed_csv],
        check=True,
    )
    run_cmd(
        ["docker", "exec", name, "osrm-datastore", "--dataset-name", dataset, osrm_file],
        check=True,
    )
    print("교통량 CSV 변경 반영 완료 (customize + datastore)")


# -------------------------
# Main: ensure + csv change
# -------------------------
def ensure_osrm_running_with_csv_check(
    container_name: str = "osrm-mld-ab",
    local_csv_path: str = "./tsp-route-optimization/osrm-data/updates_total_midpoint_100m.csv",
    osrm_base_url: str = "http://localhost:5050",
    healthcheck_wait_sec: int = 30,
    speed_csv_in_container: str = "/data/updates_total_midpoint_100m.csv",
    start_kwargs: Optional[Dict[str, Any]] = None,
) -> None:
    """
    1) 컨테이너 실행/응답 상태 점검
       - 실행 중이고 응답 OK면 그대로
       - 실행 중이나 응답 X면 restart → 실패 시 재생성
       - 실행 중 아님이면 Docker 켜고 재생성
    2) 로컬 CSV 해시 비교로 변경 감지
    3) 변경 시:
       - 새로 부팅했으면 start_osrm_server에서 이미 customize+datastore 했으므로 skip
       - 기존 실행 컨테이너면 customize+datastore 수행
    4) 반영 성공 후에만 해시 저장
    """
    start_kwargs = start_kwargs or {}

    with cross_platform_file_lock(LOCK_FILE):
        print(
        f"[ENV] os_name={os.name} sys_platform={sys.platform} "
        f"IS_WINDOWS={IS_WINDOWS} IS_MAC={IS_MAC} IS_LINUX={IS_LINUX}"
        )
        print(
            f"[ENV] cwd={Path.cwd()} "
            f"data_dir={start_kwargs.get('data_dir', 'tsp-route-optimization/osrm-data')} "
            f"resolved_data_dir={resolve_path(start_kwargs.get('data_dir', 'tsp-route-optimization/osrm-data'))}"
        )
        print(
            f"[ENV] python={sys.version.split()[0]} docker_info_ok={docker_info_ok()}"
        )
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
                    start_docker_desktop_if_needed()
                    start_osrm_server(name=container_name, **start_kwargs)
                    did_boot = True

                    if not wait_for_osrm(osrm_base_url=osrm_base_url, max_wait_sec=healthcheck_wait_sec):
                        raise RuntimeError("OSRM 서버 복구 실패(재생성 후에도 health check 불가)")
        else:
            start_docker_desktop_if_needed()
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
            write_last_hash(new_hash)
            return

        # --- 4) 실행 중 컨테이너에 반영 (실패 시 복구) ---
        try:
            reload_traffic_in_running_container(
                name=container_name,
                speed_csv=speed_csv_in_container,
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
                start_docker_desktop_if_needed()
                start_osrm_server(name=container_name, **start_kwargs)

                if not wait_for_osrm(osrm_base_url=osrm_base_url, max_wait_sec=healthcheck_wait_sec):
                    raise RuntimeError("OSRM 서버 복구 실패(반영 이후)")

        # ✅ 여기까지 오면 '반영 및 서버 정상'이므로 해시 기록
        write_last_hash(new_hash)