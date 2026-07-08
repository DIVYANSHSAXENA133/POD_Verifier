"""
Bangalore 5PM SDD — Automated network optimizer (one day of orders).

Rules (aligned with tour builder):
- 5:00 PM pickup + 90 min halt → depart ~6:30 PM
- Road distance = haversine × 1.4
- Vehicle speeds: Big 10, Medium 15, Small 40 km/h
- Stop 7 min per hub; cross-dock delay 20 min
- Green ≤ 8:30 PM, Yellow ≤ 9:30 PM
- LM + franchise hub locations are FIXED
- Priority: maximize green hub coverage, then minimize cost

Inputs:
- Blitz_Orders_Cleaned.csv (day 1 orders)
- query_result hubs + warehouse lat/lng CSV
"""

from __future__ import annotations

import json
import math
import re
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

CFG = {
    "pickup_hr": 17.0,
    "halt_hr": 1.5,
    "depart_hr": 18.5,
    "stop_min": 7,
    "road_factor": 1.4,
    "green_hr": 20.5,
    "yellow_hr": 21.5,
    "cross_dock_min": 20,
    "lm_cost_per_order": 30,
    "day_number": 1,
}

VEHICLES = [
    {"type": "Big", "capacity": 500, "cost": 2000, "speed": 10},
    {"type": "Medium", "capacity": 250, "cost": 1500, "speed": 15},
    {"type": "Small", "capacity": 50, "cost": 1000, "speed": 40},
]

ORDERS_PATH = Path("/Users/divyanshsaxena/Downloads/Blitz_Orders_Cleaned.csv")
QUERY_RESULT_PATH = Path(
    "/Users/divyanshsaxena/Downloads/query_result_2026-05-25T13_38_36.614839528Z.xlsx"
)
WAREHOUSE_CSV_PATH = Path(
    "/Users/divyanshsaxena/Downloads/query_result_2026-05-28T09_02_21.591250752Z.csv"
)
OUT_DIR = Path(__file__).parent
OUT_JSON = OUT_DIR / "optimized_network_day1.json"
OUT_HTML = OUT_DIR / "optimized_network_day1.html"

# Max hubs per tour before splitting (keeps routes feasible)
MAX_HUBS_PER_TOUR = 5
MAX_TOUR_VOLUME = 200  # split earlier for green feasibility


# ---------------------------------------------------------------------------
# Geo
# ---------------------------------------------------------------------------

def haversine(lat1, lon1, lat2, lon2) -> float:
    r = 6371
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dp = math.radians(lat2 - lat1)
    dl = math.radians(lon2 - lon1)
    a = math.sin(dp / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return 2 * r * math.asin(math.sqrt(a))


def road_km(lat1, lon1, lat2, lon2) -> float:
    return haversine(lat1, lon1, lat2, lon2) * CFG["road_factor"]


def nearest_idx(lat, lng, points: List[Tuple[float, float]]) -> int:
    best, bi = float("inf"), 0
    for i, (la, lo) in enumerate(points):
        d = haversine(lat, lng, la, lo)
        if d < best:
            best, bi = d, i
    return bi


# ---------------------------------------------------------------------------
# Hub / warehouse loading (reuse geocoding from generate_network_tool)
# ---------------------------------------------------------------------------

BANGALORE_COORDS = {
    "marathahalli": (12.9591, 77.6974),
    "whitefield": (12.9698, 77.7500),
    "domlur": (12.9610, 77.6387),
    "hsr layout": (12.9116, 77.6389),
    "koramangala": (12.9352, 77.6245),
    "indiranagar": (12.9784, 77.6408),
    "hebbal": (13.0358, 77.5970),
    "nagasandra": (13.0485, 77.5170),
    "mathikere": (13.0200, 77.5700),
    "yelahanka": (13.1007, 77.5963),
    "btm layout": (12.9166, 77.6101),
    "jayanagar": (12.9250, 77.5838),
    "jp nagar": (12.9063, 77.5857),
    "bommasandra": (12.8160, 77.6940),
    "electronic city": (12.8456, 77.6603),
    "560037": (12.9591, 77.6974),
    "560102": (12.9116, 77.6389),
    "560076": (12.9166, 77.6101),
}


def geocode_node(row) -> Tuple[float, float]:
    text = f"{row.get('Address', '')} {row.get('Node Name', '')} {row.get('Display Name', '')}".lower()
    for pc in re.findall(r"56\d{4}", text):
        if pc in BANGALORE_COORDS:
            return BANGALORE_COORDS[pc]
    for area, coords in BANGALORE_COORDS.items():
        if area in text:
            return coords
    return (12.9716, 77.5946)


def parse_lat_lng(lat_str, lng_str) -> Optional[Tuple[float, float]]:
    try:
        lat_val = float(re.sub(r"[°\s]*[NSns]?\s*$", "", str(lat_str).strip()))
        lng_val = float(re.sub(r"[°\s]*[EWew]?\s*$", "", str(lng_str).strip()))
        if 10 < lat_val < 20 and 70 < lng_val < 85:
            return (lat_val, lng_val)
    except (ValueError, TypeError):
        pass
    return None


def load_hubs() -> List[dict]:
    nodes = pd.read_excel(QUERY_RESULT_PATH)
    blr = nodes[nodes["City Name"] == "Bangalore"].copy()
    blr["coords"] = blr.apply(geocode_node, axis=1)
    hubs = []
    for _, row in blr.iterrows():
        lat, lng = row["coords"]
        hubs.append(
            {
                "id": int(row["Node ID"]),
                "name": str(row["Display Name"]),
                "type": "franchise_hub" if row["Node Type"] == "quick_hub" else row["Node Type"],
                "lat": lat,
                "lng": lng,
            }
        )
    return hubs


def load_warehouses() -> List[dict]:
    wh_df = pd.read_csv(WAREHOUSE_CSV_PATH)
    blr = wh_df[wh_df["City"] == "Bangalore"].copy()
    whs = []
    for _, row in blr.iterrows():
        coords = parse_lat_lng(row.get("Lat", ""), row.get("Lng", ""))
        if not coords:
            continue
        wid = row.get("Warehouse Int ID") or row.get("Warehouse ID")
        if pd.isna(wid):
            continue
        whs.append(
            {
                "id": int(wid),
                "name": str(row.get("Warehouse Name", ""))[:60],
                "lat": coords[0],
                "lng": coords[1],
            }
        )
    # dedupe
    seen = set()
    out = []
    for w in whs:
        if w["id"] not in seen:
            seen.add(w["id"])
            out.append(w)
    return out


# ---------------------------------------------------------------------------
# Tour timing & vehicle selection
# ---------------------------------------------------------------------------

def tour_arrivals(
    wh_lat: float,
    wh_lng: float,
    hub_ids: List[int],
    hub_by_id: Dict[int, dict],
    speed: float,
    start_lat: Optional[float] = None,
    start_lng: Optional[float] = None,
    start_time: Optional[float] = None,
) -> List[Tuple[int, float]]:
    t = start_time if start_time is not None else CFG["depart_hr"]
    lat = start_lat if start_lat is not None else wh_lat
    lng = start_lng if start_lng is not None else wh_lng
    results = []
    for hid in hub_ids:
        h = hub_by_id[hid]
        t += road_km(lat, lng, h["lat"], h["lng"]) / speed
        results.append((hid, t))
        t += CFG["stop_min"] / 60
        lat, lng = h["lat"], h["lng"]
    return results


def last_arrival(
    wh_lat, wh_lng, hub_ids, hub_by_id, speed, **kwargs
) -> float:
    arr = tour_arrivals(wh_lat, wh_lng, hub_ids, hub_by_id, speed, **kwargs)
    return arr[-1][1] if arr else CFG["depart_hr"]


def select_vehicle_for_tour(
    wh_lat, wh_lng, hub_ids, hub_by_id, shipments: int
) -> dict:
    if shipments <= 0:
        shipments = 1
    candidates = []
    for v in VEHICLES:
        count = math.ceil(shipments / v["capacity"])
        cost = count * v["cost"]
        la = last_arrival(wh_lat, wh_lng, hub_ids, hub_by_id, v["speed"])
        candidates.append(
            {
                **v,
                "count": count,
                "total_cost": cost,
                "last_arrival": la,
                "is_green": la <= CFG["green_hr"],
                "is_yellow": la <= CFG["yellow_hr"],
            }
        )
    green = [c for c in candidates if c["is_green"]]
    if green:
        return min(green, key=lambda c: c["total_cost"])
    return candidates[0]  # fastest (Small)


def select_vehicles_capacity(shipments: int) -> List[dict]:
    """Cheapest vehicle mix for capacity only."""
    if shipments <= 0:
        return []
    vs = sorted(VEHICLES, key=lambda x: -x["capacity"])
    best_combo, best_cost = None, float("inf")
    cap_big = min(math.ceil(shipments / vs[0]["capacity"]), 10)
    cap_med = min(math.ceil(shipments / vs[1]["capacity"]), 10)
    for b in range(cap_big + 1):
        for m in range(cap_med + 1):
            tc = b * vs[0]["capacity"] + m * vs[1]["capacity"]
            if tc >= shipments:
                cost = b * vs[0]["cost"] + m * vs[1]["cost"]
                if cost < best_cost:
                    best_cost, best_combo = cost, (b, m, 0)
                break
            need = shipments - tc
            s = math.ceil(need / vs[2]["capacity"])
            cost = b * vs[0]["cost"] + m * vs[1]["cost"] + s * vs[2]["cost"]
            if cost < best_cost:
                best_cost, best_combo = cost, (b, m, s)
    b, m, s = best_combo
    out = []
    if b:
        out.append({**vs[0], "count": b})
    if m:
        out.append({**vs[1], "count": m})
    if s:
        out.append({**vs[2], "count": s})
    return out


def nn_order(wh_lat, wh_lng, hub_ids: List[int], hub_by_id: dict) -> List[int]:
    remaining = hub_ids[:]
    order = []
    lat, lng = wh_lat, wh_lng
    while remaining:
        best_i, best_d = 0, float("inf")
        for i, hid in enumerate(remaining):
            h = hub_by_id[hid]
            d = road_km(lat, lng, h["lat"], h["lng"])
            if d < best_d:
                best_d, best_i = d, i
        hid = remaining.pop(best_i)
        order.append(hid)
        h = hub_by_id[hid]
        lat, lng = h["lat"], h["lng"]
    return order


def split_hubs_into_tours(hub_volumes: Dict[int, int]) -> List[List[int]]:
    """Split hubs into tours by volume and count."""
    hubs = sorted(hub_volumes.keys(), key=lambda h: -hub_volumes[h])
    tours: List[List[int]] = []
    current: List[int] = []
    current_vol = 0

    for hid in hubs:
        vol = hub_volumes[hid]
        if current and (
            len(current) >= MAX_HUBS_PER_TOUR
            or current_vol + vol > MAX_TOUR_VOLUME
        ):
            tours.append(current)
            current, current_vol = [], 0
        current.append(hid)
        current_vol += vol
    if current:
        tours.append(current)
    return tours


def split_tour_half(hub_ids: List[int]) -> Tuple[List[int], List[int]]:
    if len(hub_ids) <= 1:
        return hub_ids, []
    mid = len(hub_ids) // 2
    return hub_ids[:mid], hub_ids[mid:]


def build_tour(
    wh: dict,
    hub_ids: List[int],
    hub_volumes: Dict[int, int],
    hub_by_id: dict,
) -> TourResult:
    ordered = nn_order(wh["lat"], wh["lng"], hub_ids, hub_by_id)
    shipments = sum(hub_volumes.get(h, 0) for h in ordered)
    veh = select_vehicle_for_tour(wh["lat"], wh["lng"], ordered, hub_by_id, shipments)
    arrs = tour_arrivals(wh["lat"], wh["lng"], ordered, hub_by_id, veh["speed"])
    arrival_rows = []
    for hid, t in arrs:
        if t <= CFG["green_hr"]:
            st = "green"
        elif t <= CFG["yellow_hr"]:
            st = "yellow"
        else:
            st = "red"
        arrival_rows.append(
            {
                "hub_id": hid,
                "hub_name": hub_by_id[hid]["name"],
                "arrival_hr": round(t, 3),
                "arrival": f"{int(t)}:{int((t % 1) * 60):02d}",
                "orders": hub_volumes.get(hid, 0),
                "status": st,
            }
        )
    if veh["last_arrival"] <= CFG["green_hr"]:
        status = "green"
    elif veh["last_arrival"] <= CFG["yellow_hr"]:
        status = "yellow"
    else:
        status = "red"
    return TourResult(
        hub_ids=ordered,
        hub_volumes={h: hub_volumes[h] for h in ordered},
        vehicle=veh,
        arrivals=arrival_rows,
        fm_cost=veh["total_cost"],
        status=status,
    )


def ensure_green_tours(
    wh: dict,
    tours: List[TourResult],
    hub_volumes: Dict[int, int],
    hub_by_id: dict,
    depth: int = 0,
) -> List[TourResult]:
    """Split tours until all stops are green or max depth."""
    if depth > 6:
        return tours
    out: List[TourResult] = []
    for t in tours:
        not_all_green = any(a["status"] != "green" for a in t.arrivals)
        if not_all_green and len(t.hub_ids) > 1:
            a, b = split_tour_half(t.hub_ids)
            sub_a = build_tour(wh, a, hub_volumes, hub_by_id)
            sub_b = build_tour(wh, b, hub_volumes, hub_by_id)
            out.extend(ensure_green_tours(wh, [sub_a, sub_b], hub_volumes, hub_by_id, depth + 1))
        else:
            out.append(t)
    return out


# ---------------------------------------------------------------------------
# Optimization per warehouse
# ---------------------------------------------------------------------------

@dataclass
class TourResult:
    hub_ids: List[int]
    hub_volumes: Dict[int, int]
    vehicle: dict
    arrivals: List[dict]
    fm_cost: int
    status: str  # green / yellow / red


@dataclass
class WarehousePlan:
    warehouse: dict
    tours: List[TourResult] = field(default_factory=list)
    total_orders: int = 0
    fm_cost: int = 0


def optimize_warehouse(
    wh: dict,
    hub_volumes: Dict[int, int],
    hub_by_id: dict,
) -> WarehousePlan:
    plan = WarehousePlan(warehouse=wh, total_orders=sum(hub_volumes.values()))
    tour_groups = split_hubs_into_tours(hub_volumes)

    raw_tours: List[TourResult] = []
    for group in tour_groups:
        raw_tours.append(build_tour(wh, group, hub_volumes, hub_by_id))
    final_tours = ensure_green_tours(wh, raw_tours, hub_volumes, hub_by_id)
    for tr in final_tours:
        plan.tours.append(tr)
        plan.fm_cost += tr.fm_cost

    return plan


def load_day_orders(day: int = 1) -> pd.DataFrame:
    chunks = []
    for chunk in pd.read_csv(ORDERS_PATH, chunksize=200_000):
        d = chunk[chunk["day_number"] == day]
        if len(d):
            chunks.append(d)
    return pd.concat(chunks, ignore_index=True) if chunks else pd.DataFrame()


def assign_orders(
    orders: pd.DataFrame,
    hubs: List[dict],
    warehouses: List[dict],
) -> Tuple[Dict[int, Dict[int, int]], Dict[str, int], dict]:
    """
    Returns:
      wh_hub_vol[wh_id][hub_id] = order count
      store_wh_map[pickup_id] -> wh_id
      stats
    """
    hub_pts = [(h["lat"], h["lng"]) for h in hubs]
    wh_pts = [(w["lat"], w["lng"]) for w in warehouses]
    hub_ids = [h["id"] for h in hubs]

    wh_hub: Dict[int, Dict[int, int]] = defaultdict(lambda: defaultdict(int))
    store_wh: Dict[str, int] = {}

    for _, row in orders.iterrows():
        plat, plng = float(row["pickup_lat"]), float(row["pickup_lng"])
        dlat, dlng = float(row["drop_lat"]), float(row["drop_lng"])
        pid = str(row["pickup_id"])

        if pid not in store_wh:
            wi = nearest_idx(plat, plng, wh_pts)
            store_wh[pid] = warehouses[wi]["id"]

        hi = nearest_idx(dlat, dlng, hub_pts)
        hid = hub_ids[hi]
        wid = store_wh[pid]
        wh_hub[wid][hid] += 1

    stats = {
        "orders": len(orders),
        "stores": len(store_wh),
        "warehouse_used": len(wh_hub),
        "hub_used": len({h for w in wh_hub.values() for h in w}),
    }
    return wh_hub, store_wh, stats


# ---------------------------------------------------------------------------
# HTML report
# ---------------------------------------------------------------------------

def write_html(
    plans: List[WarehousePlan],
    hubs: List[dict],
    stats: dict,
    hub_totals: Dict[int, int],
) -> None:
    total_orders = sum(p.total_orders for p in plans)
    total_fm = sum(p.fm_cost for p in plans)
    total_lm = total_orders * CFG["lm_cost_per_order"]
    total_cost = total_fm + total_lm

    green_hubs = yellow_hubs = red_hubs = 0
    for p in plans:
        for t in p.tours:
            for a in t.arrivals:
                if a["status"] == "green":
                    green_hubs += 1
                elif a["status"] == "yellow":
                    yellow_hubs += 1
                else:
                    red_hubs += 1

    wh_rows = ""
    for p in sorted(plans, key=lambda x: -x.total_orders):
        wh_rows += f"<tr><td>{p.warehouse['name']}</td><td>{p.total_orders}</td><td>{len(p.tours)}</td><td>₹{p.fm_cost}</td></tr>"
        for ti, t in enumerate(p.tours, 1):
            v = t.vehicle
            wh_rows += (
                f"<tr><td colspan='4' style='padding-left:20px;font-size:12px;color:#888;'>"
                f"Tour {ti}: {v['count']}×{v['type']} @ {v['speed']}km/h — "
                f"{t.status.upper()} — ₹{t.fm_cost} — hubs: {', '.join(str(h) for h in t.hub_ids)}"
                f"</td></tr>"
            )

    hub_rows = ""
    for h in sorted(hubs, key=lambda x: -hub_totals.get(x["id"], 0)):
        vol = hub_totals.get(h["id"], 0)
        if vol == 0:
            continue
        hub_rows += f"<tr><td>{h['name']}</td><td>{h['type']}</td><td>{vol}</td></tr>"

    html = f"""<!DOCTYPE html>
<html><head><meta charset="utf-8"><title>Optimized Network — Day {CFG['day_number']}</title>
<style>
body {{ font-family: system-ui; background:#1a1a2e; color:#eee; padding:24px; }}
h1 {{ color:#00d4aa; }}
.card {{ background:#16213e; padding:16px; border-radius:8px; margin:12px 0; }}
.grid {{ display:grid; grid-template-columns: repeat(4,1fr); gap:12px; }}
.stat .v {{ font-size:28px; font-weight:bold; color:#00d4aa; }}
table {{ width:100%; border-collapse:collapse; font-size:13px; }}
th,td {{ padding:8px; border-bottom:1px solid #333; text-align:left; }}
th {{ color:#00d4aa; }}
</style></head><body>
<h1>Bangalore 5PM SDD — Optimized Network (Day {CFG['day_number']})</h1>
<p>Orders assigned to nearest <b>fixed</b> LM/franchise hub; pickups mapped to nearest warehouse. 
Vehicle choice: green-feasible first, then lowest cost.</p>
<div class="grid">
  <div class="card stat"><div class="v">{stats['orders']}</div><div>Orders</div></div>
  <div class="card stat"><div class="v">{stats['warehouse_used']}</div><div>Warehouses used</div></div>
  <div class="card stat"><div class="v">₹{total_cost:,}</div><div>Total cost</div></div>
  <div class="card stat"><div class="v">₹{total_cost/max(total_orders,1):.1f}</div><div>Cost / order</div></div>
</div>
<div class="card">
  <h3>Hub arrival feasibility (tour stops)</h3>
  <p>🟢 Green ≤ 8:30 PM: <b>{green_hubs}</b> &nbsp; 🟡 Yellow ≤ 9:30: <b>{yellow_hubs}</b> &nbsp; 🔴 Red: <b>{red_hubs}</b></p>
  <p>FM ₹{total_fm:,} + LM ₹{total_lm:,} (₹{CFG['lm_cost_per_order']}/order)</p>
</div>
<div class="card"><h3>Warehouses & tours</h3>
<table><tr><th>Warehouse</th><th>Orders</th><th>Tours</th><th>FM cost</th></tr>{wh_rows}</table></div>
<div class="card"><h3>Hub volumes</h3>
<table><tr><th>Hub</th><th>Type</th><th>Orders</th></tr>{hub_rows}</table></div>
<p style="color:#888;font-size:12px;">Import <code>optimized_network_day1.json</code> into the tour builder for manual tweaks.</p>
</body></html>"""
    OUT_HTML.write_text(html, encoding="utf-8")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("Loading hubs & warehouses...")
    hubs = load_hubs()
    warehouses = load_warehouses()
    hub_by_id = {h["id"]: h for h in hubs}

    print(f"Loading day {CFG['day_number']} orders...")
    orders = load_day_orders(CFG["day_number"])
    print(f"  {len(orders)} orders")

    print("Assigning orders → hub (drop) & store → warehouse (pickup)...")
    wh_hub_vol, store_wh, stats = assign_orders(orders, hubs, warehouses)
    wh_by_id = {w["id"]: w for w in warehouses}

    hub_totals: Dict[int, int] = defaultdict(int)
    for wv in wh_hub_vol.values():
        for hid, c in wv.items():
            hub_totals[hid] += c

    # Only optimize warehouses with volume; top N by volume to keep runtime reasonable
    wh_sorted = sorted(
        wh_hub_vol.items(), key=lambda x: sum(x[1].values()), reverse=True
    )
    # Use all warehouses with orders (5730 orders / ~86 wh is fine)
    plans: List[WarehousePlan] = []
    print("Optimizing tours per warehouse...")
    for wid, hv in wh_sorted:
        if wid not in wh_by_id:
            continue
        wh = wh_by_id[wid]
        plan = optimize_warehouse(wh, dict(hv), hub_by_id)
        plans.append(plan)
        g = sum(1 for t in plan.tours for a in t.arrivals if a["status"] == "green")
        print(f"  {wh['name'][:40]:40} | {plan.total_orders:5} ord | {len(plan.tours)} tours | green stops: {g}")

    total_orders = sum(p.total_orders for p in plans)
    total_fm = sum(p.fm_cost for p in plans)
    total_lm = total_orders * CFG["lm_cost_per_order"]

    export = {
        "config": CFG,
        "vehicles": VEHICLES,
        "stats": stats,
        "summary": {
            "total_orders": total_orders,
            "fm_cost": total_fm,
            "lm_cost": total_lm,
            "total_cost": total_fm + total_lm,
            "cost_per_order": round((total_fm + total_lm) / max(total_orders, 1), 2),
            "warehouses": len(plans),
        },
        "hubs": hubs,
        "warehouse_plans": [
            {
                "warehouse_id": p.warehouse["id"],
                "warehouse_name": p.warehouse["name"],
                "lat": p.warehouse["lat"],
                "lng": p.warehouse["lng"],
                "total_orders": p.total_orders,
                "fm_cost": p.fm_cost,
                "tours": [
                    {
                        "hub_ids": t.hub_ids,
                        "hub_volumes": t.hub_volumes,
                        "vehicle": {
                            "type": t.vehicle["type"],
                            "count": t.vehicle["count"],
                            "speed": t.vehicle["speed"],
                            "total_cost": t.vehicle["total_cost"],
                            "last_arrival": t.vehicle["last_arrival"],
                        },
                        "status": t.status,
                        "arrivals": t.arrivals,
                        "fm_cost": t.fm_cost,
                    }
                    for t in p.tours
                ],
            }
            for p in plans
        ],
        "store_to_warehouse": store_wh,
    }

    OUT_JSON.write_text(json.dumps(export, indent=2), encoding="utf-8")
    write_html(plans, hubs, stats, hub_totals)

    print("\n" + "=" * 60)
    print(f"Orders:        {total_orders}")
    print(f"Warehouses:    {len(plans)}")
    print(f"FM cost:       ₹{total_fm:,}")
    print(f"LM cost:       ₹{total_lm:,}")
    print(f"Total cost:    ₹{total_fm + total_lm:,}")
    print(f"Cost/order:    ₹{(total_fm + total_lm) / max(total_orders, 1):.2f}")
    print(f"\nSaved: {OUT_JSON}")
    print(f"Saved: {OUT_HTML}")


if __name__ == "__main__":
    main()
