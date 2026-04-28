from __future__ import annotations

from typing import Any, Dict, Tuple

from .payload import get_address_list


def remap_unit_to_zero(
    payload: Dict[str, Any],
    *,
    unit_original_id: int,
    matrix_key: str,
) -> Tuple[Dict[str, Any], Dict[int, int]]:
    """Relabel only the 'id' field so that unit becomes 0.

    - Address list order (matrix indices) is unchanged.
    - Matrix itself is NOT reordered.

    Returns:
      (new_payload, new_to_old_id_map)
    """
    address_list = get_address_list(payload)
    old_ids = [int(r["id"]) for r in address_list]
    if int(unit_original_id) not in old_ids:
        raise ValueError(f"unit_original_id={unit_original_id} not found in payload ids.")

    new_to_old: Dict[int, int] = {}
    used = set()
    next_id = 1

    new_address_list = []
    for rec in address_list:
        old_id = int(rec["id"])
        if old_id == int(unit_original_id):
            new_id = 0
        else:
            while next_id in used or next_id == 0:
                next_id += 1
            new_id = next_id
            next_id += 1

        used.add(new_id)
        new_to_old[new_id] = old_id

        new_rec = dict(rec)
        new_rec["id"] = new_id
        new_address_list.append(new_rec)

    new_payload = dict(payload)
    if payload.get("address_list"):
        new_payload["address_list"] = new_address_list
    else:
        new_payload["address_geocode_list"] = new_address_list

    new_payload.setdefault("meta", {})
    new_payload["meta"].update(
        {
            "unit_remapped_to_zero": True,
            "unit_original_id": int(unit_original_id),
            "matrix_key": matrix_key,
        }
    )

    return new_payload, new_to_old
