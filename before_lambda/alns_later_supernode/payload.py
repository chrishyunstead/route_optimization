from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import numpy as np

def drop_node_from_payload(payload: Dict[str, Any], *, drop_id: int, matrix_key: str) -> Dict[str, Any]:
    """
    payload에서 특정 id(drop_id) 노드를 제거하고,
    dist_matrix에서도 해당 row/col 제거한 새 payload 반환.
    """
    address_list = get_address_list(payload)

    drop_id = int(drop_id)
    drop_idx = None
    for i, r in enumerate(address_list):
        if int(r["id"]) == drop_id:
            drop_idx = i
            break

    # 없으면 그대로 반환
    if drop_idx is None:
        return payload

    new_list = [r for r in address_list if int(r["id"]) != drop_id]

    m = get_matrix(payload, matrix_key)
    m2 = np.delete(np.delete(m, drop_idx, axis=0), drop_idx, axis=1)

    new_payload = dict(payload)

    if "address_list" in payload and payload.get("address_list"):
        new_payload["address_list"] = new_list
    else:
        new_payload["address_geocode_list"] = new_list

    new_payload[matrix_key] = m2.tolist()

    new_payload.setdefault("meta", {})
    new_payload["meta"].update(
        {"dropped_node_id": drop_id, "dropped_node_index": drop_idx, "matrix_key": matrix_key}
    )
    return new_payload

def get_address_list(payload: Dict[str, Any]) -> List[Dict[str, Any]]:
    if payload.get("address_list"):
        return payload["address_list"]
    if payload.get("address_geocode_list"):
        return payload["address_geocode_list"]
    raise KeyError("payload must contain 'address_list' or 'address_geocode_list'.")


def get_matrix(payload: Dict[str, Any], matrix_key: str) -> np.ndarray:
    if matrix_key not in payload:
        raise KeyError(f"payload must contain '{matrix_key}'.")
    m = np.array(payload[matrix_key], dtype=float)
    if m.ndim != 2 or m.shape[0] != m.shape[1]:
        raise ValueError(f"{matrix_key} must be a square 2D matrix. got shape={m.shape}")
    return m


def build_id_maps(address_list: List[Dict[str, Any]]) -> Tuple[Dict[int, int], Dict[int, Dict[str, Any]]]:
    id2idx = {int(rec["id"]): i for i, rec in enumerate(address_list)}
    id2rec = {int(rec["id"]): rec for rec in address_list}
    return id2idx, id2rec


def tracking_number_to_id(payload: Dict[str, Any], tracking_number: Optional[str]) -> Optional[int]:
    if tracking_number is None:
        return None
    tn = str(tracking_number).strip()
    if not tn:
        return None

    address_list = get_address_list(payload)
    found = [int(rec["id"]) for rec in address_list if str(rec.get("tracking_number") or "").strip() == tn]

    if not found:
        raise ValueError(f"tracking_number='{tn}' not found in payload.")
    if len(found) > 1:
        raise ValueError(f"tracking_number='{tn}' is not unique in payload. matched ids={found}")
    return found[0]


def get_area(rec: Dict[str, Any]) -> Any:
    return rec.get("Area") or rec.get("area") or rec.get("region") or rec.get("code_base")
