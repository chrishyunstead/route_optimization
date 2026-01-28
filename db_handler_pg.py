# utils/data/db_handler.py
import os, asyncio
import pandas as pd
from sshtunnel import SSHTunnelForwarder

# sync PG driver (psycopg v3)
import psycopg
from psycopg.rows import dict_row
# async PG driver
import asyncpg


class PGDBHandler:
    """SSH 터널을 자동으로 열어 PostgreSQL 에 동기·비동기로 접속"""

    def __init__(self):
        # ── 환경 변수 읽기 ───────────────────────────────────────────
        # DB (원격)
        self._remote_host   = os.getenv("PG_HOST")
        self._remote_port   = int(os.getenv("PG_PORT", 5432))  # PG 기본 5432
        self.pg_user        = os.getenv("PG_USER")
        self.pg_password    = os.getenv("PG_PASSWORD")
        self.pg_database    = os.getenv("PG_DATABASE")

        # SSH (Bastion)
        self._ssh_host  = os.getenv("PG_SSH_HOST")
        self._ssh_port  = int(os.getenv("PG_SSH_PORT", 22))
        self._ssh_user  = os.getenv("PG_SSH_USER")
        self._ssh_pkey  = os.path.expanduser(os.getenv(r"SSH_PRIVATE_KEY"))

        # 터널 객체 및 로컬 포트 자리
        self._tunnel: SSHTunnelForwarder | None = None
        self.pg_host = "127.0.0.1"
        self.pg_port = None  # 나중에 터널 열면서 채움

        # async pool
        self.pool: asyncpg.pool.Pool | None = None

        # 바로 터널 열기
        self._start_ssh_tunnel()

    # ──────────────────────────────────────────────────────────────
    def _start_ssh_tunnel(self):
        """SSH 터널을 열고 로컬 포트 할당"""
        self._tunnel = SSHTunnelForwarder(
            (self._ssh_host, self._ssh_port),
            ssh_username=self._ssh_user,
            ssh_pkey=self._ssh_pkey,
            remote_bind_address=(self._remote_host, self._remote_port),
            local_bind_address=("127.0.0.1", 0),  # 0 → 임의의 빈 포트
            set_keepalive=30,  # 30초마다 keepalive
        )
        self._tunnel.start()
        self.pg_port = self._tunnel.local_bind_port
        print(f"[DBHandler] SSH tunnel open → 127.0.0.1:{self.pg_port}")

    def _stop_ssh_tunnel(self):
        if self._tunnel and self._tunnel.is_active:
            self._tunnel.stop()
            print("[DBHandler] SSH tunnel closed")

    def _ensure_tunnel(self):
        if self._tunnel is None or (not self._tunnel.is_active):
            try:
                self._stop_ssh_tunnel()
            except Exception:
                pass
            self._start_ssh_tunnel()

    def fetch_data(self, database, query, params=None):
        if database == "gis":
            database = self.pg_database

        for attempt in (1, 2):
            conn = None
            try:
                self._ensure_tunnel()

                conn = psycopg.connect(
                    host=self.pg_host,
                    port=self.pg_port,
                    user=self.pg_user,
                    password=self.pg_password,
                    dbname=database,
                    connect_timeout=10,
                    row_factory=dict_row,
                )
                with conn.cursor() as cur:
                    cur.execute(query, params) if params is not None else cur.execute(query)
                    rows = cur.fetchall()
                return pd.DataFrame(rows)

            except Exception as e:
                print(f"[DBHandler] sync query error (attempt={attempt}): {e}")
                # Broken pipe류면 터널 재시작 후 재시도
                try:
                    self._stop_ssh_tunnel()
                except Exception:
                    pass
                if attempt == 2:
                    return None
            finally:
                try:
                    if conn:
                        conn.close()
                except Exception:
                    pass


    # ── 비동기 쿼리 (asyncpg) ─────────────────────────────────────
    async def init_pool(self, database: str | None = None):
        if self.pool is None:
            self.pool = await asyncpg.create_pool(
                host=self.pg_host,
                port=self.pg_port,
                user=self.pg_user,
                password=self.pg_password,
                database=database or self.pg_database,
                min_size=1,
                max_size=5,
                command_timeout=30,
            )

    async def fetch_data_async(self, query: str, params: tuple | list | None = None, database: str | None = None):
        try:
            await self.init_pool(database=database)
            assert self.pool is not None
            async with self.pool.acquire() as conn:
                records = await conn.fetch(query, *(params or []))
                return [dict(r) for r in records]
        except Exception as e:
            print(f"[DBHandler] async query error: {e}")
            return None

    # ── 소멸자: 프로그램 종료 시 터널 닫기 ─────────────────────────
    def close(self):
        # async pool 먼저 닫기
        try:
            if self.pool is not None:
                loop = None
                try:
                    loop = asyncio.get_running_loop()
                except RuntimeError:
                    loop = None

                if loop and loop.is_running():
                    loop.create_task(self.pool.close())
                else:
                    asyncio.run(self.pool.close())
                self.pool = None
        except Exception:
            pass

        # 터널 닫기
        try:
            self._stop_ssh_tunnel()
        except Exception:
            pass
