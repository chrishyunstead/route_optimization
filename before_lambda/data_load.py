# 데이터 파이프라인
import pandas as pd
import numpy as np
import pymysql
from sshtunnel import SSHTunnelForwarder
import os
from dotenv import load_dotenv

class AutoContainerGeneration:
    def __init__(self, version=1, debug=False):
        self.version = version
        self.debug = debug
        print(f"AutoContainerGeneration version: {version}")

        load_dotenv()

    # 공통 MySQL 데이터 추출 메서드
    def fetch_data(self, query, ssh_host, ssh_user, ssh_private_key, mysql_host, mysql_port, mysql_user, mysql_password, mysql_database):
        """
        MySQL 데이터를 추출하여 DF로 변환
        """
        try:
            with SSHTunnelForwarder(
                (ssh_host, 22),
                ssh_username=ssh_user,
                ssh_private_key=ssh_private_key,
                remote_bind_address=(mysql_host, mysql_port)
            ) as tunnel:
                print("SSH 터널 연결 성공")

                with pymysql.connect(
                    host='127.0.0.1', 
                    user=mysql_user,
                    passwd=mysql_password,
                    db=mysql_database,
                    charset='utf8',
                    port=tunnel.local_bind_port,
                    cursorclass=pymysql.cursors.DictCursor) as conn:
                    with conn.cursor() as cur:
                        cur.execute(query)
                        results = cur.fetchall()
                        print("쿼리 실행 완료")

                        # 결과를 DF로 변환

                        df = pd.DataFrame(results)
                        return df

        except Exception as e:
            print(f"Error fetching data for: {e}")
            return None

    # 전체 데이터 추출
    def fetch_all_data(self):
        """
        MySQL 쿼리를 실행하여 Shipping Items와 Bunny Schedule 데이터를 DF로 변환
        """
        # SSH 및 MySQL 설정
        ssh_host = os.getenv("SSH_HOST_VER_1")
        ssh_user = os.getenv("SSH_USER")
        ssh_private_key = os.getenv(r"SSH_PRIVATE_KEY")
        

        mysql_host = os.getenv("MYSQL_HOST")
        mysql_port = 3306
        mysql_user = os.getenv("MYSQL_USER")
        mysql_password = os.getenv("MYSQL_PASSWORD")
        mysql_database = os.getenv("MYSQL_DATABASE")

        area_cluster_query = """
        select
        REGEXP_REPLACE(location_sector.code, '[0-9]', '') AS 'Area',
        shipping_shippingitem.tracking_number,
        location_address.lat as "lat",
        location_address.lng as "lng",
        location_address.address_road,
        location_address.address2,
        DATE_ADD(shipping_shippingitem.timestamp_delivery_complete, INTERVAL 9 HOUR) AS "timestamp_delivery_complete"
        from
        shipping_shippingitem
        JOIN shipping_container
        ON shipping_shippingitem.shipping_container_id = shipping_container.id
        join hub_cluster_plan_item
        on shipping_shippingitem.id = hub_cluster_plan_item.shipping_item_id
        join hub_cluster_plan
        on hub_cluster_plan_item.cluster_plan_id = hub_cluster_plan.id
        join location_sector
        on shipping_shippingitem.designated_sector_id = location_sector.id
        join location_address
        on shipping_shippingitem.address_id = location_address.id
        JOIN user_profile_rider
        ON shipping_container.user_id = user_profile_rider.user_id
        where shipping_shippingitem.timestamp_delivery_complete is not null
        AND hub_cluster_plan.plan_date = DATE_SUB(CURDATE(), INTERVAL 1 DAY)
       	AND user_profile_rider.user_id = '26854';
        """
        unit_query = """
        SELECT 
            REGEXP_REPLACE(location_sector.code, '[0-9]', '') as Area, 
            location_address.lat as unit_lat,
            location_address.lng as 'unit_lng'
        FROM 
        location_unitstorage
        JOIN location_address 
        ON location_unitstorage.address_id  = location_address.id
        JOIN location_unitstorage_sectors
        ON location_unitstorage_sectors.locationunitstorage_id = location_unitstorage.id
        JOIN location_sector
        ON location_unitstorage_sectors.sector_id = location_sector.id
        WHERE 
        location_unitstorage.is_active = 1
        AND
        location_sector.allowed = 1
        GROUP BY REGEXP_REPLACE(location_sector.code, '[0-9]', '')
        ORDER BY REGEXP_REPLACE(location_sector.code, '[0-9]', '');
        """

        # 각 쿼리 결과를 DataFrame으로 가져오기
        # container = self.fetch_data(container_query, ssh_host, ssh_user, ssh_private_key,
        #                                           mysql_host, mysql_port, mysql_user, mysql_password, mysql_database)
        area_cluster = self.fetch_data(area_cluster_query, ssh_host, ssh_user, ssh_private_key,
                                                  mysql_host, mysql_port, mysql_user, mysql_password, mysql_database)
        unit = self.fetch_data(unit_query, ssh_host, ssh_user, ssh_private_key,
                                                  mysql_host, mysql_port, mysql_user, mysql_password, mysql_database)   
        # API INPUT 형식에 맞게 타입 변경
        area_cluster['lat'] = area_cluster['lat'].astype(float)
        area_cluster['lng'] = area_cluster['lng'].astype(float)

        unit['unit_lat'] = unit['unit_lat'].astype(float)
        unit['unit_lng'] = unit['unit_lng'].astype(float)

        return area_cluster, unit
        