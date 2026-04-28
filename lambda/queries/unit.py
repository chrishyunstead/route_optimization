# unit.py
class UnitDatasetQuery:
    def __init__(self, db_handler):
        self.db_handler = db_handler

    def unit_dataset_df(self):
        query = r"""
        SELECT
            REGEXP_REPLACE(location_sector.code, '[0-9]', '') AS Area,
            location_address.lat AS unit_lat,
            location_address.lng AS unit_lng
        FROM location_sector
        JOIN location_unitstorage
        ON location_unitstorage.id = location_sector.unit_id
        JOIN location_address
        ON location_address.id = location_unitstorage.address_id
        WHERE
            location_unitstorage.is_active = 1
        AND location_sector.allowed = 1
        GROUP BY
            REGEXP_REPLACE(location_sector.code, '[0-9]', '')
        ORDER BY
            REGEXP_REPLACE(location_sector.code, '[0-9]', '');
        """
        return self.db_handler.fetch_data("daas", query, query_name="unit_dataset_df")