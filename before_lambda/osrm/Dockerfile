FROM osrm/osrm-backend

# 작업 디렉토리 설정
WORKDIR /data

# PBF 및 Lua 파일 복사
COPY car.lua /opt/car.lua
COPY lua_lib /opt/lib
COPY south-korea-latest.osm.pbf /data/south-korea-latest.osm.pbf

# Lua 모듈 경로 지정
ENV LUA_PATH="/opt/lib/?.lua;;"

# Extract + Contract 단계 실행
RUN osrm-extract -p /opt/car.lua /data/south-korea-latest.osm.pbf && \
    osrm-contract /data/south-korea-latest.osrm

# OSRM 라우팅 서버 실행
CMD ["osrm-routed", "/data/south-korea-latest.osrm"]