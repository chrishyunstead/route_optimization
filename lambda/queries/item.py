# item.py
class ItemDatasetQuery:
    def __init__(self, db_handler):
        self.db_handler = db_handler

    def item_dataset_df(self, user_id):
        query = f"""
        SELECT
            REGEXP_REPLACE(location_sector.code, '[0-9]', '') AS 'Area',
            shipping_shippingitem.tracking_number,
            location_address.lat as "lat",
            location_address.lng as "lng",
            location_address.address_road,
            location_address.address2
        FROM shipping_shippingitem
            JOIN shipping_shippingitemtimetable
            ON shipping_shippingitem.id = shipping_shippingitemtimetable.shipping_item_id
            JOIN shipping_container
            ON shipping_shippingitem.shipping_container_id = shipping_container.id
            JOIN location_sector
            ON shipping_shippingitem.designated_sector_id = location_sector.id
            JOIN location_address
            ON shipping_shippingitem.address_id = location_address.id
            JOIN user_profile_rider
            ON shipping_container.user_id = user_profile_rider.user_id
            WHERE user_profile_rider.user_id = {user_id}
            AND shipping_shippingitemtimetable.timestamp_delivery_complete is null
            AND shipping_shippingitemtimetable.timestamp_return_collected is null
            AND shipping_container.timestamp_checkin is not null
            AND DATE_ADD(shipping_container.timestamp_checkin, INTERVAL 9 HOUR)
                >= CONCAT(CURDATE(), ' 10:00:00')
            AND DATE_ADD(shipping_container.timestamp_checkin, INTERVAL 9 HOUR)
                <=  CONCAT(DATE_ADD(CURDATE(), INTERVAL 1 DAY), ' 03:00:00')
        """
        return self.db_handler.fetch_data("daas", query, query_name="item_dataset_df")