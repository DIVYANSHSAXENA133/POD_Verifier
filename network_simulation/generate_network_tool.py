"""
Bangalore 5PM SDD Network Simulation - Interactive Tour Builder
================================================================
Generates an interactive HTML tool where you can:
1. Select a warehouse → see which hubs are Green/Yellow/Red
2. Build a tour (WH → LM1 → LM2 → ...) with cumulative timing
3. Each hub recolors based on actual arrival time in the tour

Timing Model:
- 5:00 PM: Pickup from warehouse (First Mile)
- 1.5 hr halt: Sorting & bagging at warehouse
- 6:30 PM: Vehicle departs warehouse
- Travel time: based on haversine distance @ configurable avg speed
- Each LM stop: 7 minutes (configurable)
- Green: arrival <= 8:30 PM
- Yellow: arrival <= 9:30 PM
- Red: arrival > 9:30 PM

Architecture is modular - supports adding cross-docking centers as intermediate nodes.
"""

import pandas as pd
import json
import math
import re

# =============================================================================
# CONFIG (easily adjustable for iterations)
# =============================================================================

CONFIG = {
    'pickup_time_hr': 17.0,          # 5:00 PM
    'halt_duration_hr': 1.5,         # Sorting & bagging
    'departure_time_hr': 18.5,       # 6:30 PM (pickup + halt)
    'stop_duration_min': 7,          # Minutes per LM stop
    'avg_speed_kmph': 25,            # Average city speed
    'green_cutoff_hr': 20.5,         # 8:30 PM
    'yellow_cutoff_hr': 21.5,        # 9:30 PM
    'cross_dock_halt_min': 0,        # Future: time at cross-dock center
}

# =============================================================================
# DATA LOADING
# =============================================================================

QUERY_RESULT_PATH = '/Users/divyanshsaxena/Downloads/query_result_2026-05-25T13_38_36.614839528Z.xlsx'
WAREHOUSE_CSV_PATH = '/Users/divyanshsaxena/Downloads/query_result_2026-05-28T09_02_21.591250752Z.csv'
OUTPUT_PATH = '/Users/divyanshsaxena/Desktop/POD_Verifier/network_simulation/bangalore_5pm_tour_builder.html'

nodes_df = pd.read_excel(QUERY_RESULT_PATH)
warehouses_df = pd.read_csv(WAREHOUSE_CSV_PATH)

blr_nodes = nodes_df[nodes_df['City Name'] == 'Bangalore'].copy()
blr_warehouses = warehouses_df[warehouses_df['City'] == 'Bangalore'].copy()

# =============================================================================
# GEOCODING (Bangalore areas by pincode/name)
# =============================================================================

BANGALORE_COORDS = {
    'marathahalli': (12.9591, 77.6974),
    'whitefield': (12.9698, 77.7500),
    'domlur': (12.9610, 77.6387),
    'hsr layout': (12.9116, 77.6389),
    'hsr': (12.9116, 77.6389),
    'koramangala': (12.9352, 77.6245),
    'indiranagar': (12.9784, 77.6408),
    'hebbal': (13.0358, 77.5970),
    'nagasandra': (13.0485, 77.5170),
    'mathikere': (13.0200, 77.5700),
    'sundar nagar': (13.0200, 77.5700),
    'gokula': (13.0200, 77.5700),
    'chokkanahalli': (13.0700, 77.5900),
    'yelahanka': (13.1007, 77.5963),
    'btm layout': (12.9166, 77.6101),
    'btm': (12.9166, 77.6101),
    'jayanagar': (12.9250, 77.5838),
    'jp nagar': (12.9063, 77.5857),
    'konanakunte': (12.8878, 77.5737),
    'bannerghatta': (12.8876, 77.5973),
    'bommasandra': (12.8160, 77.6940),
    'chandapura': (12.8015, 77.7070),
    'begur': (12.8720, 77.6340),
    'electronic city': (12.8456, 77.6603),
    'rajarajeshwari nagar': (12.9200, 77.5190),
    'rr nagar': (12.9200, 77.5190),
    'pattanagere': (12.9200, 77.5190),
    'vijayanagar': (12.9716, 77.5366),
    'chamrajpet': (12.9600, 77.5650),
    'peenya': (13.0300, 77.5200),
    'devarabeesanahalli': (12.9570, 77.7150),
    'd.b.halli': (12.9570, 77.7150),
    'outer ring road': (12.9591, 77.6974),
    '560037': (12.9591, 77.6974),
    '560098': (12.9200, 77.5190),
    '560073': (13.0485, 77.5170),
    '560054': (13.0200, 77.5700),
    '560102': (12.9116, 77.6389),
    '560076': (12.9166, 77.6101),
    '560062': (12.8878, 77.5737),
    '560041': (12.9250, 77.5838),
    '560099': (12.8160, 77.6940),
    '560103': (12.9570, 77.7150),
    '560066': (12.9698, 77.7500),
    '560071': (12.9610, 77.6387),
    '560032': (13.0358, 77.5970),
    '560018': (12.9600, 77.5650),
    '560064': (13.0700, 77.5900),
    '560114': (12.8720, 77.6340),
}

def parse_lat_lng(lat_str, lng_str):
    """Parse lat/lng from format like '13.05274700° N' / '77.69412000° E'"""
    try:
        lat_str = str(lat_str).strip()
        lng_str = str(lng_str).strip()
        lat_val = float(re.sub(r'[°\s]*[NSns]?\s*$', '', lat_str))
        lng_val = float(re.sub(r'[°\s]*[EWew]?\s*$', '', lng_str))
        if 'S' in lat_str.upper():
            lat_val = -lat_val
        if 'W' in lng_str.upper():
            lng_val = -lng_val
        if 10 < lat_val < 20 and 70 < lng_val < 85:
            return (lat_val, lng_val)
    except (ValueError, TypeError):
        pass
    return None


def geocode_node(row):
    address = str(row.get('Address', '')).lower()
    name = str(row.get('Node Name', '')).lower()
    display = str(row.get('Display Name', '')).lower()
    search_text = f"{address} {name} {display}"

    pincodes = re.findall(r'56\d{4}', search_text)
    if pincodes:
        for pc in pincodes:
            if pc in BANGALORE_COORDS:
                return BANGALORE_COORDS[pc]

    for area, coords in BANGALORE_COORDS.items():
        if area in search_text:
            return coords

    return (12.9716, 77.5946)


# Geocode all nodes (hubs)
blr_nodes['coords'] = blr_nodes.apply(geocode_node, axis=1)
blr_nodes['lat'] = blr_nodes['coords'].apply(lambda x: x[0])
blr_nodes['lng'] = blr_nodes['coords'].apply(lambda x: x[1])

# Parse warehouse lat/lng from CSV (actual coordinates provided)
blr_warehouses['parsed_coords'] = blr_warehouses.apply(
    lambda row: parse_lat_lng(row.get('Lat', ''), row.get('Lng', '')), axis=1
)
# Filter out warehouses with invalid coordinates
blr_warehouses = blr_warehouses[blr_warehouses['parsed_coords'].notna()].copy()
blr_warehouses['lat'] = blr_warehouses['parsed_coords'].apply(lambda x: x[0])
blr_warehouses['lng'] = blr_warehouses['parsed_coords'].apply(lambda x: x[1])

# =============================================================================
# BUILD JSON DATA FOR FRONTEND
# =============================================================================

# All delivery hubs (LM + Quick hubs) - these are the drop points
hubs_data = []
for _, row in blr_nodes.iterrows():
    hubs_data.append({
        'id': int(row['Node ID']),
        'name': row['Node Name'],
        'display_name': row['Display Name'],
        'type': 'franchise_hub' if row['Node Type'] == 'quick_hub' else row['Node Type'],
        'lat': round(row['lat'], 6),
        'lng': round(row['lng'], 6),
        'address': str(row['Address'])[:100],
        'sort_code': row['Sort Codes'],
        'clusters': str(row.get('Clusters', ''))
    })

# Warehouses (from new CSV with actual lat/lng)
wh_data = []
for _, row in blr_warehouses.iterrows():
    wh_id = row.get('Warehouse Int ID') or row.get('Warehouse ID')
    if pd.isna(wh_id):
        continue
    company_names = str(row.get('Company Names', ''))
    wh_data.append({
        'id': int(wh_id),
        'name': str(row.get('Warehouse Name', ''))[:60],
        'user': company_names[:40] if company_names != 'nan' else str(row.get('Warehouse Name', '')),
        'lat': round(row['lat'], 6),
        'lng': round(row['lng'], 6),
        'current_cutoff': 'N/A',
    })

# Deduplicate warehouses by ID
seen_wh = set()
unique_wh = []
for wh in wh_data:
    if wh['id'] not in seen_wh:
        seen_wh.add(wh['id'])
        unique_wh.append(wh)
wh_data = unique_wh

print(f"Hubs: {len(hubs_data)}, Warehouses: {len(wh_data)}")

# =============================================================================
# GENERATE INTERACTIVE HTML
# =============================================================================

html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Bangalore 5PM SDD - Network Tour Builder</title>
    <link rel="stylesheet" href="https://unpkg.com/leaflet@1.9.4/dist/leaflet.css" />
    <script src="https://unpkg.com/leaflet@1.9.4/dist/leaflet.js"></script>
    <style>
        * {{ margin: 0; padding: 0; box-sizing: border-box; }}
        body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif; display: flex; height: 100vh; }}
        
        #sidebar {{
            width: 420px;
            background: #1a1a2e;
            color: #eee;
            overflow-y: auto;
            padding: 16px;
            display: flex;
            flex-direction: column;
            gap: 12px;
        }}
        
        #map {{ flex: 1; }}
        
        h1 {{ font-size: 16px; color: #00d4aa; margin-bottom: 4px; }}
        h2 {{ font-size: 13px; color: #aaa; font-weight: normal; margin-bottom: 8px; }}
        h3 {{ font-size: 13px; color: #00d4aa; margin: 8px 0 4px; }}
        
        .config-section {{
            background: #16213e;
            border-radius: 8px;
            padding: 12px;
        }}
        
        .config-row {{
            display: flex;
            justify-content: space-between;
            align-items: center;
            margin: 4px 0;
            font-size: 12px;
        }}
        
        .config-row label {{ color: #aaa; }}
        .config-row input {{
            width: 70px;
            background: #0f3460;
            border: 1px solid #444;
            color: #fff;
            padding: 3px 6px;
            border-radius: 4px;
            text-align: right;
            font-size: 12px;
        }}
        
        select {{
            width: 100%;
            background: #0f3460;
            border: 1px solid #444;
            color: #fff;
            padding: 8px;
            border-radius: 6px;
            font-size: 12px;
        }}
        
        .tour-list {{
            list-style: none;
            max-height: 200px;
            overflow-y: auto;
        }}
        
        .tour-item {{
            display: flex;
            align-items: center;
            gap: 8px;
            padding: 6px 8px;
            margin: 3px 0;
            border-radius: 6px;
            font-size: 12px;
            background: #16213e;
        }}
        
        .tour-item .dot {{
            width: 10px;
            height: 10px;
            border-radius: 50%;
            flex-shrink: 0;
        }}
        
        .tour-item .time {{ color: #aaa; margin-left: auto; font-family: monospace; }}
        
        .btn {{
            padding: 8px 12px;
            border: none;
            border-radius: 6px;
            cursor: pointer;
            font-size: 12px;
            font-weight: 600;
        }}
        
        .btn-primary {{ background: #00d4aa; color: #1a1a2e; }}
        .btn-danger {{ background: #e94560; color: #fff; }}
        .btn-secondary {{ background: #0f3460; color: #aaa; border: 1px solid #444; }}
        
        .btn:hover {{ opacity: 0.85; }}
        
        .stats-grid {{
            display: grid;
            grid-template-columns: 1fr 1fr 1fr;
            gap: 8px;
        }}
        
        .stat-card {{
            background: #0f3460;
            border-radius: 6px;
            padding: 8px;
            text-align: center;
        }}
        
        .stat-card .value {{ font-size: 18px; font-weight: bold; }}
        .stat-card .label {{ font-size: 10px; color: #aaa; }}
        .stat-green .value {{ color: #00d4aa; }}
        .stat-yellow .value {{ color: #ffc107; }}
        .stat-red .value {{ color: #e94560; }}
        
        .legend {{
            display: flex;
            gap: 12px;
            font-size: 11px;
            padding: 6px 0;
            flex-wrap: wrap;
        }}
        .legend-item {{ display: flex; align-items: center; gap: 4px; }}
        .legend-dot {{ width: 10px; height: 10px; border-radius: 50%; }}
        .legend-square {{ width: 10px; height: 10px; border-radius: 2px; }}
        
        .filter-row {{
            display: flex;
            gap: 10px;
            font-size: 11px;
            align-items: center;
            margin: 4px 0;
        }}
        .filter-row label {{ color: #aaa; cursor: pointer; display: flex; align-items: center; gap: 4px; }}
        
        .timeline {{
            font-size: 11px;
            color: #aaa;
            border-left: 2px solid #444;
            padding-left: 12px;
            margin: 8px 0;
        }}
        .timeline div {{ margin: 3px 0; }}
        .timeline .time-marker {{ color: #00d4aa; font-family: monospace; }}
        
        #hub-click-hint {{
            font-size: 11px;
            color: #ffc107;
            font-style: italic;
            margin-top: 4px;
        }}
        
        .cross-dock-section {{
            border: 1px dashed #444;
            border-radius: 8px;
            padding: 10px;
            opacity: 0.7;
        }}
        .cross-dock-section h3 {{ color: #666; }}
        
        .cost-panel {{
            background: #0a1628;
            border: 1px solid #00d4aa;
            border-radius: 8px;
            padding: 12px;
        }}
        .cost-panel h3 {{ color: #00d4aa; }}
        
        .cost-grid {{
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 6px;
            margin: 8px 0;
        }}
        .cost-card {{
            background: #16213e;
            border-radius: 6px;
            padding: 8px;
            text-align: center;
        }}
        .cost-card .value {{ font-size: 16px; font-weight: bold; color: #00d4aa; }}
        .cost-card .label {{ font-size: 9px; color: #888; text-transform: uppercase; }}
        .cost-card.total {{ grid-column: span 2; border: 1px solid #00d4aa44; }}
        .cost-card.total .value {{ font-size: 22px; }}
        
        .vehicle-table {{
            width: 100%;
            font-size: 11px;
            border-collapse: collapse;
            margin: 6px 0;
        }}
        .vehicle-table th {{ color: #888; text-align: left; padding: 3px 4px; border-bottom: 1px solid #333; }}
        .vehicle-table td {{ padding: 3px 4px; color: #ccc; }}
        .vehicle-table input {{
            width: 55px;
            background: #0f3460;
            border: 1px solid #333;
            color: #fff;
            padding: 2px 4px;
            border-radius: 3px;
            font-size: 11px;
            text-align: right;
        }}
        
        .hub-volume-list {{
            max-height: 180px;
            overflow-y: auto;
            font-size: 11px;
        }}
        .hub-vol-row {{
            display: flex;
            align-items: center;
            gap: 4px;
            padding: 2px 0;
            border-bottom: 1px solid #1a1a2e;
        }}
        .hub-vol-row input {{
            width: 45px;
            background: #0f3460;
            border: 1px solid #333;
            color: #fff;
            padding: 2px 4px;
            border-radius: 3px;
            font-size: 11px;
            text-align: right;
        }}
        .hub-vol-row .hub-name {{ flex: 1; color: #aaa; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }}
        
        .cost-breakdown {{
            font-size: 11px;
            margin: 8px 0;
            padding: 8px;
            background: #16213e;
            border-radius: 6px;
        }}
        .cost-breakdown .row {{
            display: flex;
            justify-content: space-between;
            padding: 2px 0;
        }}
        .cost-breakdown .row.section-header {{
            color: #00d4aa;
            font-weight: bold;
            border-bottom: 1px solid #333;
            margin-top: 6px;
            padding-bottom: 4px;
        }}
        .cost-breakdown .row .label {{ color: #aaa; }}
        .cost-breakdown .row .val {{ color: #fff; font-family: monospace; }}
    </style>
</head>
<body>
    <div id="sidebar">
        <div>
            <h1>Bangalore 5PM SDD Network</h1>
            <h2>Tour Builder & Feasibility Analyzer</h2>
        </div>
        
        <!-- Config -->
        <div class="config-section">
            <h3>Timing Parameters</h3>
            <div class="config-row">
                <label>FM Pickup Time</label>
                <input type="text" id="cfg-pickup" value="17:00" onchange="recalculate()">
            </div>
            <div class="config-row">
                <label>Halt (sorting/bagging) min</label>
                <input type="number" id="cfg-halt" value="90" onchange="recalculate()">
            </div>
            <div class="config-row">
                <label>Road Distance Factor</label>
                <input type="number" id="cfg-road-factor" value="1.4" step="0.1" onchange="recalculate()">
            </div>
            <div class="config-row" style="margin-top:4px;border-top:1px solid #333;padding-top:4px;">
                <label>Speed: Big Vehicle (km/h)</label>
                <input type="number" id="cfg-speed-big" value="10" onchange="recalculate()">
            </div>
            <div class="config-row">
                <label>Speed: Medium Vehicle (km/h)</label>
                <input type="number" id="cfg-speed-med" value="15" onchange="recalculate()">
            </div>
            <div class="config-row">
                <label>Speed: Small Vehicle (km/h)</label>
                <input type="number" id="cfg-speed-small" value="40" onchange="recalculate()">
            </div>
            <div class="config-row">
                <label>Stop Duration (min)</label>
                <input type="number" id="cfg-stop" value="7" onchange="recalculate()">
            </div>
            <div class="config-row">
                <label>Green Cutoff</label>
                <input type="text" id="cfg-green" value="20:30" onchange="recalculate()">
            </div>
            <div class="config-row">
                <label>Yellow Cutoff</label>
                <input type="text" id="cfg-yellow" value="21:30" onchange="recalculate()">
            </div>
        </div>
        
        <!-- Multi-Warehouse Network -->
        <div class="config-section">
            <h3>Network Warehouses</h3>
            <div style="display:flex; gap:4px; margin-bottom:6px;">
                <select id="warehouse-add-select" style="flex:1;">
                    <option value="">-- Add warehouse to network --</option>
                </select>
                <button class="btn btn-primary" onclick="addWarehouseToNetwork()" style="font-size:10px;padding:4px 8px;">+ Add</button>
            </div>
            <div id="network-wh-list" style="max-height:140px;overflow-y:auto;margin-bottom:6px;"></div>
            <div style="font-size:10px;color:#888;" id="network-wh-summary">No warehouses in network</div>
        </div>
        
        <!-- Active Warehouse (for tour building) -->
        <div class="config-section">
            <h3>Active Warehouse (Tour Building)</h3>
            <select id="active-wh-select" style="margin-bottom:6px;"></select>
            <div class="timeline" id="timing-display" style="display:none;">
                <div><span class="time-marker" id="td-pickup"></span> FM Pickup</div>
                <div><span class="time-marker" id="td-depart"></span> Vehicle Departs (post halt)</div>
                <div>... tour drops ...</div>
            </div>
        </div>
        
        <!-- Stats -->
        <div class="stats-grid" id="stats-grid" style="display:none;">
            <div class="stat-card stat-green"><div class="value" id="stat-green">0</div><div class="label">GREEN (≤8:30)</div></div>
            <div class="stat-card stat-yellow"><div class="value" id="stat-yellow">0</div><div class="label">YELLOW (≤9:30)</div></div>
            <div class="stat-card stat-red"><div class="value" id="stat-red">0</div><div class="label">RED (>9:30)</div></div>
        </div>
        <div id="stats-breakdown" style="display:none; font-size:11px; color:#aaa; padding:0 4px;">
            <span id="stat-lm-detail"></span><br>
            <span id="stat-fh-detail"></span>
        </div>
        
        <!-- Tour Builder -->
        <div class="config-section">
            <h3>Tour Builder</h3>
            <div class="legend">
                <div class="legend-item"><div class="legend-dot" style="background:#00d4aa;"></div> Green ≤ 8:30PM</div>
                <div class="legend-item"><div class="legend-dot" style="background:#ffc107;"></div> Yellow ≤ 9:30PM</div>
                <div class="legend-item"><div class="legend-dot" style="background:#e94560;"></div> Red > 9:30PM</div>
            </div>
            <div class="legend" style="margin-top:4px;">
                <div class="legend-item"><div class="legend-dot" style="background:#888;border:2px solid #fff;"></div> LM Hub (circle)</div>
                <div class="legend-item"><div class="legend-square" style="background:#888;border:2px solid #fff;"></div> Franchise Hub (square)</div>
            </div>
            <div class="filter-row">
                <label><input type="checkbox" id="filter-lm" checked onchange="recalculate()"> LM Hubs ({{len(blr_nodes[blr_nodes['Node Type'] == 'lm_hub'])}})</label>
                <label><input type="checkbox" id="filter-franchise" checked onchange="recalculate()"> Franchise Hubs ({{len(blr_nodes[blr_nodes['Node Type'] == 'quick_hub'])}})</label>
            </div>
            <p id="hub-click-hint">Left-click hubs → add to active tour | Right-click LM hub → toggle cross-dock</p>
            
            <!-- Active Tour Selector -->
            <h3 style="margin-top:6px;font-size:12px;">Active Tour</h3>
            <select id="active-tour-select" style="margin-bottom:6px;"></select>
            <div style="display:flex; gap:6px; margin-bottom:8px; flex-wrap:wrap;">
                <button class="btn btn-primary" onclick="addNewDirectTour()" style="font-size:10px;">+ New Direct Tour (from WH)</button>
                <button class="btn btn-secondary" onclick="addSubtour()" style="font-size:10px;">+ New Sub-Tour</button>
                <select id="cd-target-select" style="font-size:10px;max-width:120px;display:none;"></select>
            </div>
            
            <!-- Active Tour Stops -->
            <h3 style="font-size:11px;color:#aaa;" id="active-tour-label">Tour 1 (WH → Direct)</h3>
            <ol class="tour-list" id="tour-list"></ol>
            <div style="display:flex; gap:8px; margin-top:8px;">
                <button class="btn btn-danger" onclick="clearActiveTour()">Clear</button>
                <button class="btn btn-secondary" onclick="undoLastStop()">Undo Last</button>
                <button class="btn btn-primary" onclick="optimizeTour()">Auto-Optimize</button>
                <button class="btn btn-danger" onclick="deleteActiveTour()" style="font-size:10px;">Delete Tour</button>
            </div>
            <div id="tour-summary" style="margin-top:8px; font-size:11px; color:#aaa;"></div>
        </div>
        
        <!-- Cross-Docking -->
        <div class="config-section" style="border:1px solid #00d4aa55;">
            <h3 style="color:#00d4aa;">Cross-Docking Stations</h3>
            <div class="config-row">
                <label>Cross-dock delay (min)</label>
                <input type="number" id="cfg-crossdock" value="20" onchange="recalculate()">
            </div>
            <p style="font-size:11px; color:#aaa; margin-top:4px;">
                Right-click any hub (LM or Franchise) on map to toggle it as a cross-dock.<br>
                Sub-tours fan out from cross-dock with added delay.
            </p>
            <div id="crossdock-list" style="margin-top:8px;"></div>
            
            <div id="subtour-summary" style="margin-top:6px; font-size:11px; color:#aaa;"></div>
        </div>
        
        <!-- COST SIMULATION -->
        <div class="config-section cost-panel">
            <h3>Cost Simulation</h3>
            
            <!-- Order Volume -->
            <div class="config-row">
                <label>LM Cost / Order (₹)</label>
                <input type="number" id="cfg-lm-cost" value="30" onchange="recalculate()">
            </div>
            <p style="font-size:10px;color:#888;margin-top:4px;">Order load per warehouse is set in the Network Warehouses section above.</p>
            <button class="btn btn-secondary" onclick="distributeOrdersEvenly()" style="margin-top:4px;font-size:10px;">Distribute Evenly</button>
            <button class="btn btn-secondary" onclick="distributeOrdersProportional()" style="margin-top:4px;font-size:10px;">Proportional by Distance</button>
            
            <!-- Vehicle Config -->
            <h3 style="margin-top:10px;font-size:12px;">Vehicle Types</h3>
            <table class="vehicle-table">
                <tr><th>Type</th><th>Capacity</th><th>Cost/Tour (₹)</th></tr>
                <tr>
                    <td>Big</td>
                    <td><input type="number" id="v-big-cap" value="500" onchange="recalculate()"></td>
                    <td><input type="number" id="v-big-cost" value="2000" onchange="recalculate()"></td>
                </tr>
                <tr>
                    <td>Medium</td>
                    <td><input type="number" id="v-med-cap" value="250" onchange="recalculate()"></td>
                    <td><input type="number" id="v-med-cost" value="1500" onchange="recalculate()"></td>
                </tr>
                <tr>
                    <td>Small</td>
                    <td><input type="number" id="v-small-cap" value="50" onchange="recalculate()"></td>
                    <td><input type="number" id="v-small-cost" value="1000" onchange="recalculate()"></td>
                </tr>
            </table>
            
            <!-- Per-Hub Order Volumes -->
            <h3 style="margin-top:10px;font-size:12px;">Orders per Hub (in active tours)</h3>
            <div class="hub-volume-list" id="hub-volume-list"></div>
            
            <!-- Live Cost Display -->
            <div class="cost-grid" id="cost-grid" style="margin-top:10px;">
                <div class="cost-card"><div class="value" id="cost-fm">₹0</div><div class="label">First Mile</div></div>
                <div class="cost-card"><div class="value" id="cost-mm">₹0</div><div class="label">Mid Mile (CD)</div></div>
                <div class="cost-card"><div class="value" id="cost-lm">₹0</div><div class="label">Last Mile</div></div>
                <div class="cost-card"><div class="value" id="cost-per-order">₹0</div><div class="label">Cost/Order</div></div>
                <div class="cost-card total"><div class="value" id="cost-total">₹0</div><div class="label">Total Network Cost</div></div>
            </div>
            
            <!-- Detailed Breakdown -->
            <div class="cost-breakdown" id="cost-breakdown"></div>
        </div>
    </div>
    
    <div id="map"></div>

    <script>
    // =========================================================================
    // DATA
    // =========================================================================
    const HUBS = {json.dumps(hubs_data)};
    const WAREHOUSES = {json.dumps(wh_data)};
    
    // =========================================================================
    // STATE
    // =========================================================================
    
    // Multi-warehouse network state
    // networkWarehouses: whId -> state object with wh, load, tours, directTourCount, crossDockHubs
    let networkWarehouses = {{}};
    let activeWhId = null;  // Currently active warehouse for tour building
    
    // Convenience getters for active warehouse state
    function getActiveWHState() {{
        if (!activeWhId || !networkWarehouses[activeWhId]) return null;
        return networkWarehouses[activeWhId];
    }}
    
    // Legacy compatibility: these now point to active warehouse's data
    function get_selectedWarehouse() {{
        const state = getActiveWHState();
        return state ? state.wh : null;
    }}
    function get_tours() {{
        const state = getActiveWHState();
        return state ? state.tours : {{ direct_1: [] }};
    }}
    function get_crossDockHubs() {{
        const state = getActiveWHState();
        return state ? state.crossDockHubs : new Set();
    }}
    function get_activeTourKey() {{
        const state = getActiveWHState();
        return state ? state.activeTourKey : 'direct_1';
    }}
    function set_activeTourKey(val) {{
        const state = getActiveWHState();
        if (state) state.activeTourKey = val;
    }}
    
    // Shorthand property access - proxies to active warehouse state
    function _defProp(name, getter, setter) {{
        var desc = {{get: getter}};
        if (setter) desc.set = setter;
        Object.defineProperty(window, name, desc);
    }}
    _defProp('selectedWarehouse', get_selectedWarehouse);
    _defProp('tours', get_tours, function(v) {{ var s = getActiveWHState(); if(s) s.tours = v; }});
    _defProp('crossDockHubs', get_crossDockHubs, function(v) {{ var s = getActiveWHState(); if(s) s.crossDockHubs = v; }});
    _defProp('activeTourKey', get_activeTourKey, set_activeTourKey);
    
    let hubMarkers = {{}};
    let warehouseMarkers = {{}};
    let directLines = [];
    let cdMarkerOverlays = {{}};
    
    // =========================================================================
    // MAP INIT
    // =========================================================================
    const map = L.map('map').setView([12.9716, 77.6200], 11);
    L.tileLayer('https://{{s}}.basemaps.cartocdn.com/dark_all/{{z}}/{{x}}/{{y}}{{r}}.png', {{
        attribution: '© CartoDB',
        maxZoom: 19
    }}).addTo(map);
    
    // =========================================================================
    // UTILITY FUNCTIONS
    // =========================================================================
    function haversine(lat1, lon1, lat2, lon2) {{
        const R = 6371;
        const dLat = (lat2 - lat1) * Math.PI / 180;
        const dLon = (lon2 - lon1) * Math.PI / 180;
        const a = Math.sin(dLat/2) * Math.sin(dLat/2) +
                  Math.cos(lat1 * Math.PI/180) * Math.cos(lat2 * Math.PI/180) *
                  Math.sin(dLon/2) * Math.sin(dLon/2);
        return R * 2 * Math.atan2(Math.sqrt(a), Math.sqrt(1-a));
    }}
    
    function parseTime(timeStr) {{
        const [h, m] = timeStr.split(':').map(Number);
        return h + m / 60;
    }}
    
    function formatTime(decimalHours) {{
        const h = Math.floor(decimalHours);
        const m = Math.round((decimalHours - h) * 60);
        return `${{String(h).padStart(2,'0')}}:${{String(m).padStart(2,'0')}}`;
    }}
    
    function getConfig() {{
        return {{
            pickupTime: parseTime(document.getElementById('cfg-pickup').value),
            haltMin: parseInt(document.getElementById('cfg-halt').value),
            roadFactor: parseFloat(document.getElementById('cfg-road-factor').value),
            speedBig: parseInt(document.getElementById('cfg-speed-big').value),
            speedMed: parseInt(document.getElementById('cfg-speed-med').value),
            speedSmall: parseInt(document.getElementById('cfg-speed-small').value),
            stopMin: parseInt(document.getElementById('cfg-stop').value),
            greenCutoff: parseTime(document.getElementById('cfg-green').value),
            yellowCutoff: parseTime(document.getElementById('cfg-yellow').value),
            crossDockMin: parseInt(document.getElementById('cfg-crossdock').value),
        }};
    }}
    
    // Road distance = haversine * road factor (accounts for actual road routing)
    function roadDistance(lat1, lon1, lat2, lon2) {{
        const cfg = getConfig();
        return haversine(lat1, lon1, lat2, lon2) * cfg.roadFactor;
    }}
    
    // Simulate tour timing at a given speed, returns last arrival time
    function simulateTourLastArrival(tourKey, speed) {{
        const tourHubs = tours[tourKey] || [];
        if (!selectedWarehouse || tourHubs.length === 0) return 0;
        
        const cfg = getConfig();
        const baseDepartTime = cfg.pickupTime + cfg.haltMin / 60;
        
        let currentTime, currentLat, currentLng;
        
        if (tourKey.startsWith('direct_')) {{
            currentTime = baseDepartTime;
            currentLat = selectedWarehouse.lat;
            currentLng = selectedWarehouse.lng;
        }} else {{
            // Cross-dock sub-tour: depart from CD hub after it's reached via a direct tour + delay
            const parts = tourKey.split('_');
            const cdHubId = parseInt(parts[1]);
            const cdHub = HUBS.find(h => h.id === cdHubId);
            if (!cdHub) return 99;
            const arrivalAtCD = getCrossDockArrivalTime(cdHubId);
            currentTime = (arrivalAtCD || baseDepartTime) + cfg.crossDockMin / 60;
            currentLat = cdHub.lat;
            currentLng = cdHub.lng;
        }}
        
        let lastArrival = currentTime;
        for (let i = 0; i < tourHubs.length; i++) {{
            const hub = HUBS.find(h => h.id === tourHubs[i]);
            if (!hub) continue;
            const dist = roadDistance(currentLat, currentLng, hub.lat, hub.lng);
            currentTime += dist / speed;
            lastArrival = currentTime;
            currentTime += cfg.stopMin / 60;
            currentLat = hub.lat;
            currentLng = hub.lng;
        }}
        
        return lastArrival;
    }}
    
    // Select optimal vehicle for a tour:
    // Priority: 1) capacity fit, 2) all-green timing, 3) cheapest among green options
    function selectTourVehicle(tourKey) {{
        const cfg = getConfig();
        const tourHubs = tours[tourKey] || [];
        let tourShipments = 0;
        tourHubs.forEach(hId => {{ tourShipments += (hubOrderVolumes[hId] || 0); }});
        if (tourShipments === 0) tourShipments = 1; // avoid zero
        
        const vehicleCfg = getVehicleConfig();
        // All possible options that satisfy capacity (sorted by speed: fastest first)
        const options = [
            {{ type: 'Small', capacity: vehicleCfg[2].capacity, cost: vehicleCfg[2].cost, speed: cfg.speedSmall }},
            {{ type: 'Medium', capacity: vehicleCfg[1].capacity, cost: vehicleCfg[1].cost, speed: cfg.speedMed }},
            {{ type: 'Big', capacity: vehicleCfg[0].capacity, cost: vehicleCfg[0].cost, speed: cfg.speedBig }},
        ];
        
        // For each option, calculate: how many vehicles needed + total cost + is tour green?
        const candidates = [];
        for (const opt of options) {{
            const count = Math.ceil(tourShipments / opt.capacity);
            const totalCost = count * opt.cost;
            const lastArrival = simulateTourLastArrival(tourKey, opt.speed);
            const isGreen = lastArrival <= cfg.greenCutoff;
            const isYellow = lastArrival <= cfg.yellowCutoff;
            candidates.push({{ ...opt, count, totalCost, lastArrival, isGreen, isYellow }});
        }}
        
        // Selection priority:
        // 1. Pick green options first (all arrivals within green cutoff)
        const greenOpts = candidates.filter(c => c.isGreen);
        if (greenOpts.length > 0) {{
            // Among green options, pick cheapest
            greenOpts.sort((a, b) => a.totalCost - b.totalCost);
            return greenOpts[0];
        }}
        
        // 2. No green option exists → pick fastest (to maximize green stops)
        // Fastest = smallest vehicle (highest speed)
        return candidates[0]; // Already sorted fastest-first (Small)
    }}
    
    // Get speed for a tour (uses optimal vehicle selection)
    function getTourSpeed(tourKey) {{
        const result = selectTourVehicle(tourKey);
        return result.speed;
    }}
    
    // Default speed when no tour context (for direct arrival color estimates)
    function getDefaultSpeed() {{
        const cfg = getConfig();
        // Use small vehicle speed for optimistic direct estimate
        return cfg.speedSmall;
    }}
    
    function getColor(arrivalHr, cfg) {{
        if (arrivalHr <= cfg.greenCutoff) return '#00d4aa';
        if (arrivalHr <= cfg.yellowCutoff) return '#ffc107';
        return '#e94560';
    }}
    
    function getColorName(arrivalHr, cfg) {{
        if (arrivalHr <= cfg.greenCutoff) return 'green';
        if (arrivalHr <= cfg.yellowCutoff) return 'yellow';
        return 'red';
    }}
    
    // =========================================================================
    // CROSS-DOCK ARRIVAL: find when a CD hub is actually reached in a direct tour
    // =========================================================================
    function getCrossDockArrivalTime(cdHubId) {{
        if (!selectedWarehouse) return null;
        const cfg = getConfig();
        const baseDepartTime = cfg.pickupTime + cfg.haltMin / 60;
        
        // Search all direct tours for the CD hub
        const directKeys = Object.keys(tours).filter(k => k.startsWith('direct_'));
        for (const tourKey of directKeys) {{
            const tourHubs = tours[tourKey] || [];
            const idx = tourHubs.indexOf(cdHubId);
            if (idx === -1) continue;
            
            // Simulate this tour up to the CD hub stop
            const speed = getTourSpeed(tourKey);
            let currentTime = baseDepartTime;
            let currentLat = selectedWarehouse.lat;
            let currentLng = selectedWarehouse.lng;
            
            for (let i = 0; i <= idx; i++) {{
                const hub = HUBS.find(h => h.id === tourHubs[i]);
                if (!hub) continue;
                const dist = roadDistance(currentLat, currentLng, hub.lat, hub.lng);
                currentTime += dist / speed;
                if (i === idx) return currentTime; // arrival at CD
                currentTime += cfg.stopMin / 60;
                currentLat = hub.lat;
                currentLng = hub.lng;
            }}
        }}
        
        // CD hub not found in any direct tour — fallback: direct WH→CD estimate
        const cdHub = HUBS.find(h => h.id === cdHubId);
        if (!cdHub) return null;
        const dist = roadDistance(selectedWarehouse.lat, selectedWarehouse.lng, cdHub.lat, cdHub.lng);
        const speed = getDefaultSpeed();
        return baseDepartTime + dist / speed;
    }}
    
    // =========================================================================
    // TOUR DEPARTURE TIME (accounts for cross-dock if sub-tour)
    // =========================================================================
    function getTourDepartureInfo(tourKey) {{
        const cfg = getConfig();
        const baseDepartTime = cfg.pickupTime + cfg.haltMin / 60;
        
        // Direct tours (direct_1, direct_2, ...) all depart from warehouse
        if (tourKey.startsWith('direct_')) {{
            return {{
                departTime: baseDepartTime,
                startLat: selectedWarehouse.lat,
                startLng: selectedWarehouse.lng,
                isCrossDock: false,
                cdHubName: null,
                tourLabel: `Direct Tour ${{tourKey.split('_')[1]}}`,
            }};
        }}
        
        // Cross-dock sub-tour: parse hub ID from key "cd_<hubId>_<n>"
        const parts = tourKey.split('_');
        const cdHubId = parseInt(parts[1]);
        const cdHub = HUBS.find(h => h.id === cdHubId);
        
        if (!cdHub || !selectedWarehouse) {{
            return {{ departTime: baseDepartTime, startLat: 0, startLng: 0, isCrossDock: true, cdHubName: '?' }};
        }}
        
        // Use actual arrival time from direct tour (CD is a stop on a direct tour)
        const arrivalAtCD = getCrossDockArrivalTime(cdHubId);
        const cdDepartTime = (arrivalAtCD || baseDepartTime) + cfg.crossDockMin / 60;
        
        return {{
            departTime: cdDepartTime,
            startLat: cdHub.lat,
            startLng: cdHub.lng,
            isCrossDock: true,
            cdHubName: cdHub.display_name,
            cdHubId: cdHubId,
            arrivalAtCD: arrivalAtCD,
            tourLabel: `CD Sub-tour ${{parts[2]}} (${{cdHub.display_name}})`,
        }};
    }}
    
    // =========================================================================
    // CALCULATE ARRIVAL TIME (direct from warehouse, no tour - for coloring)
    // =========================================================================
    function calcDirectArrival(warehouse, hub) {{
        const cfg = getConfig();
        const departTime = cfg.pickupTime + cfg.haltMin / 60;
        const dist = roadDistance(warehouse.lat, warehouse.lng, hub.lat, hub.lng);
        const speed = getDefaultSpeed();
        const travelHr = dist / speed;
        return departTime + travelHr;
    }}
    
    // For hubs served via a cross-dock: uses actual CD arrival (from direct tour) + delay + CD→hub
    function calcViaCrossDockArrival(warehouse, cdHub, hub) {{
        const cfg = getConfig();
        const arrivalAtCD = getCrossDockArrivalTime(cdHub.id);
        const cdDepartTime = (arrivalAtCD || (cfg.pickupTime + cfg.haltMin / 60)) + cfg.crossDockMin / 60;
        const distCDtoHub = roadDistance(cdHub.lat, cdHub.lng, hub.lat, hub.lng);
        const speed = getDefaultSpeed();
        const totalTime = cdDepartTime + distCDtoHub / speed;
        return totalTime;
    }}
    
    // =========================================================================
    // CALCULATE TOUR ARRIVALS (sequential with stops)
    // =========================================================================
    function calcTourArrivals(tourKey) {{
        if (!tourKey) tourKey = activeTourKey;
        const tourHubs = tours[tourKey] || [];
        if (!selectedWarehouse || tourHubs.length === 0) return [];
        
        const cfg = getConfig();
        const info = getTourDepartureInfo(tourKey);
        let currentTime = info.departTime;
        let currentLat = info.startLat;
        let currentLng = info.startLng;
        const speed = getTourSpeed(tourKey);
        
        const arrivals = [];
        
        for (let i = 0; i < tourHubs.length; i++) {{
            const hub = HUBS.find(h => h.id === tourHubs[i]);
            if (!hub) continue;
            
            const dist = roadDistance(currentLat, currentLng, hub.lat, hub.lng);
            const travelHr = dist / speed;
            currentTime += travelHr;
            
            arrivals.push({{
                hubId: hub.id,
                hubName: hub.display_name,
                hubType: hub.type,
                arrivalTime: currentTime,
                distance: dist,
                color: getColor(currentTime, cfg),
                colorName: getColorName(currentTime, cfg),
            }});
            
            currentTime += cfg.stopMin / 60;
            currentLat = hub.lat;
            currentLng = hub.lng;
        }}
        
        return arrivals;
    }}
    
    // Get the best arrival time for a hub across all tours
    function getBestArrivalForHub(hubId) {{
        let bestArrival = null;
        for (const key of Object.keys(tours)) {{
            if (tours[key].includes(hubId)) {{
                const arrivals = calcTourArrivals(key);
                const a = arrivals.find(x => x.hubId === hubId);
                if (a && (bestArrival === null || a.arrivalTime < bestArrival)) {{
                    bestArrival = a.arrivalTime;
                }}
            }}
        }}
        return bestArrival;
    }}
    
    // =========================================================================
    // RENDER HUBS ON MAP
    // =========================================================================
    function renderHubs() {{
        Object.values(hubMarkers).forEach(m => map.removeLayer(m));
        hubMarkers = {{}};
        Object.values(cdMarkerOverlays).forEach(m => map.removeLayer(m));
        cdMarkerOverlays = {{}};
        
        const showLM = document.getElementById('filter-lm').checked;
        const showFranchise = document.getElementById('filter-franchise').checked;
        const cfg = getConfig();
        
        HUBS.forEach(hub => {{
            if (hub.type === 'lm_hub' && !showLM) return;
            if (hub.type === 'franchise_hub' && !showFranchise) return;
            
            let color = '#666';
            let arrival = null;
            const isCD = crossDockHubs.has(hub.id);
            
            // Determine color based on tour membership
            if (selectedWarehouse) {{
                // Check if hub is in any tour
                const bestArr = getBestArrivalForHub(hub.id);
                if (bestArr !== null) {{
                    arrival = bestArr;
                    color = getColor(arrival, cfg);
                }} else {{
                    // Not in any tour - show direct color (or via nearest CD)
                    arrival = calcDirectArrival(selectedWarehouse, hub);
                    color = getColor(arrival, cfg);
                }}
            }}
            
            let marker;
            const isFranchise = hub.type === 'franchise_hub';
            const inAnyTour = Object.values(tours).some(t => t.includes(hub.id));
            
            if (isFranchise) {{
                const size = isCD ? 18 : 14;
                marker = L.marker([hub.lat, hub.lng], {{
                    icon: L.divIcon({{
                        html: `<div style="
                            width:${{size}}px;height:${{size}}px;
                            background:${{color}};
                            border:${{isCD ? '3px solid #ff00ff' : '2px solid ' + (inAnyTour ? '#fff' : '#888')}};
                            border-radius:2px;
                            opacity:0.85;
                            ${{inAnyTour ? 'box-shadow:0 0 6px ' + color + ';' : ''}}
                        "></div>`,
                        iconSize: [size, size],
                        iconAnchor: [size/2, size/2],
                        className: ''
                    }})
                }}).addTo(map);
            }} else {{
                marker = L.circleMarker([hub.lat, hub.lng], {{
                    radius: isCD ? 14 : 11,
                    color: isCD ? '#ff00ff' : '#fff',
                    fillColor: color,
                    fillOpacity: 0.85,
                    weight: isCD ? 4 : 3,
                }}).addTo(map);
            }}
            
            // Cross-dock overlay indicator
            if (isCD) {{
                const cdOverlay = L.marker([hub.lat, hub.lng], {{
                    icon: L.divIcon({{
                        html: `<div style="
                            width:24px;height:24px;border-radius:50%;
                            border:3px solid #ff00ff;
                            display:flex;align-items:center;justify-content:center;
                            font-size:10px;color:#ff00ff;font-weight:bold;
                            background:rgba(0,0,0,0.5);
                        ">CD</div>`,
                        iconSize: [24, 24],
                        iconAnchor: [12, 12],
                        className: ''
                    }}),
                    interactive: false,
                }}).addTo(map);
                cdMarkerOverlays[hub.id] = cdOverlay;
            }}
            
            const arrivalStr = arrival ? formatTime(arrival) : 'N/A';
            const distStr = selectedWarehouse ? 
                roadDistance(selectedWarehouse.lat, selectedWarehouse.lng, hub.lat, hub.lng).toFixed(1) : 'N/A';
            const typeLabel = isCD ? (isFranchise ? 'Franchise Hub (CROSS-DOCK)' : 'LM Hub (CROSS-DOCK)') : (isFranchise ? 'Franchise Hub' : 'Last Mile Hub');
            
            marker.bindPopup(`
                <b>${{hub.display_name}}</b><br>
                Type: <b>${{typeLabel}}</b><br>
                Sort Code: ${{hub.sort_code}}<br>
                ${{isCD ? '<b style="color:#ff00ff;">⬡ Cross-Dock Station (+'+ cfg.crossDockMin +' min delay)</b><br>' : ''}}
                ${{selectedWarehouse ? `Direct Distance: ${{distStr}} km<br>` : ''}}
                ${{arrival ? `Estimated Arrival: <b>${{arrivalStr}}</b><br>` : ''}}
                ${{inAnyTour ? '<b style="color:green;">✓ In Tour (click to remove)</b>' : '<i>Left-click: add to tour</i>'}}<br>
                <i style="color:#ff00ff;">Right-click: toggle cross-dock</i>
            `);
            
            marker.bindTooltip(`${{isCD ? '⬡ CD:' : (isFranchise ? '▪' : '●')}} ${{hub.display_name}}${{arrival ? ' | ' + arrivalStr : ''}}`, {{
                permanent: false,
                direction: 'top',
                offset: [0, -10]
            }});
            
            // Left click: add to active tour
            marker.on('click', () => addToTour(hub.id));
            // Right click (or double-click): toggle cross-dock
            marker.on('contextmenu', (e) => {{
                L.DomEvent.preventDefault(e);
                toggleCrossDock(hub.id);
            }});
            marker.on('dblclick', (e) => {{
                L.DomEvent.preventDefault(e);
                toggleCrossDock(hub.id);
            }});
            
            hubMarkers[hub.id] = marker;
        }});
    }}
    
    // =========================================================================
    // RENDER TOUR ROUTES (all tours)
    // =========================================================================
    function renderTourRoute() {{
        directLines.forEach(l => map.removeLayer(l));
        directLines = [];
        
        const directTourColors = ['#fff', '#00bfff', '#ff69b4', '#ffa500', '#7fff00', '#ff4444', '#44ffaa'];
        const whColors = ['#ff6b35', '#35b5ff', '#ff35b5', '#35ffb5', '#ffb535', '#b535ff'];
        let whColorIdx = 0;
        
        // Draw routes for ALL warehouses in network
        Object.keys(networkWarehouses).forEach(id => {{
            const whId = parseInt(id);
            const state = networkWarehouses[whId];
            const wh = state.wh;
            const isActiveWH = whId === activeWhId;
            const whColor = whColors[whColorIdx % whColors.length];
            whColorIdx++;
            
            let directIdx = 0;
            for (const [key, hubIds] of Object.entries(state.tours)) {{
                if (hubIds.length === 0) continue;
                
                // Temporarily set activeWhId to this WH for calculations
                const prevActiveWhId = activeWhId;
                activeWhId = whId;
                
                const info = getTourDepartureInfo(key);
                const arrivals = calcTourArrivals(key);
                const isActiveTour = isActiveWH && key === state.activeTourKey;
                
                activeWhId = prevActiveWhId;
                
                let points;
                if (key.startsWith('direct_')) {{
                    points = [[wh.lat, wh.lng]];
                    directIdx++;
                }} else {{
                    points = [[info.startLat, info.startLng]];
                }}
                
                arrivals.forEach(a => {{
                    const hub = HUBS.find(h => h.id === a.hubId);
                    if (hub) points.push([hub.lat, hub.lng]);
                }});
                
                for (let i = 0; i < points.length - 1; i++) {{
                    let segColor;
                    if (i === 0 && key.startsWith('direct_')) {{
                        segColor = isActiveWH ? directTourColors[directIdx % directTourColors.length] : whColor;
                    }} else {{
                        segColor = arrivals[Math.max(0, i-1)] ? arrivals[Math.max(0, i-1)].color : '#888';
                    }}
                    
                    const line = L.polyline([points[i], points[i+1]], {{
                        color: segColor,
                        weight: isActiveTour ? 4 : (isActiveWH ? 3 : 2),
                        opacity: isActiveTour ? 0.9 : (isActiveWH ? 0.7 : 0.4),
                        dashArray: key.startsWith('cd_') ? '4,4' : null,
                    }}).addTo(map);
                    directLines.push(line);
                }}
            }}
        }});
    }}
    
    // =========================================================================
    // UPDATE STATS & TOUR LIST
    // =========================================================================
    function updateUI() {{
        const cfg = getConfig();
        
        // Stats
        if (selectedWarehouse) {{
            document.getElementById('stats-grid').style.display = 'grid';
            document.getElementById('stats-breakdown').style.display = 'block';
            let green = 0, yellow = 0, red = 0;
            let lmG = 0, lmY = 0, lmR = 0, fhG = 0, fhY = 0, fhR = 0;
            
            HUBS.forEach(h => {{
                const bestArr = getBestArrivalForHub(h.id);
                const arr = bestArr !== null ? bestArr : calcDirectArrival(selectedWarehouse, h);
                const cn = getColorName(arr, cfg);
                if (cn === 'green') {{ green++; if (h.type === 'lm_hub') lmG++; else fhG++; }}
                else if (cn === 'yellow') {{ yellow++; if (h.type === 'lm_hub') lmY++; else fhY++; }}
                else {{ red++; if (h.type === 'lm_hub') lmR++; else fhR++; }}
            }});
            
            document.getElementById('stat-green').textContent = green;
            document.getElementById('stat-yellow').textContent = yellow;
            document.getElementById('stat-red').textContent = red;
            document.getElementById('stat-lm-detail').innerHTML = 
                `LM Hubs: <span style="color:#00d4aa">${{lmG}}G</span> / <span style="color:#ffc107">${{lmY}}Y</span> / <span style="color:#e94560">${{lmR}}R</span>`;
            document.getElementById('stat-fh-detail').innerHTML = 
                `Franchise: <span style="color:#00d4aa">${{fhG}}G</span> / <span style="color:#ffc107">${{fhY}}Y</span> / <span style="color:#e94560">${{fhR}}R</span>`;
        }} else {{
            document.getElementById('stats-grid').style.display = 'none';
            document.getElementById('stats-breakdown').style.display = 'none';
        }}
        
        // Active Tour list
        const tourList = document.getElementById('tour-list');
        tourList.innerHTML = '';
        const activeTour = tours[activeTourKey] || [];
        const info = selectedWarehouse ? getTourDepartureInfo(activeTourKey) : null;
        
        // Update label
        if (info) {{
            const label = document.getElementById('active-tour-label');
            if (activeTourKey.startsWith('direct_')) {{
                label.textContent = `Direct Tour ${{activeTourKey.split('_')[1]}} (WH → Hubs)`;
                label.style.color = '#00d4aa';
            }} else {{
                label.textContent = `CD Sub-tour (${{info.cdHubName}})`;
                label.style.color = '#ff00ff';
            }}
        }}
        
        // Show CD header if sub-tour
        if (info && info.isCrossDock && activeTour.length >= 0) {{
            const header = document.createElement('li');
            header.className = 'tour-item';
            header.style.background = '#2a1040';
            header.innerHTML = `
                <span style="color:#ff00ff;font-size:11px;">⬡ Via CD: ${{info.cdHubName}} | Departs CD: ${{formatTime(info.departTime)}} (+${{cfg.crossDockMin}}min)</span>
            `;
            tourList.appendChild(header);
        }}
        
        if (activeTour.length > 0) {{
            const arrivals = calcTourArrivals(activeTourKey);
            arrivals.forEach((a, i) => {{
                const li = document.createElement('li');
                li.className = 'tour-item';
                const typeTag = a.hubType === 'lm_hub' ? 'LM' : 'FH';
                li.innerHTML = `
                    <span style="color:#aaa;width:16px;">${{i+1}}.</span>
                    <span class="dot" style="background:${{a.color}};${{a.hubType === 'franchise_hub' ? 'border-radius:2px;' : ''}}"></span>
                    <span><small style="color:#888;">[${{typeTag}}]</small> ${{a.hubName}}</span>
                    <span class="time">${{formatTime(a.arrivalTime)}} (${{a.distance.toFixed(1)}}km)</span>
                `;
                tourList.appendChild(li);
            }});
            
            const lastArrival = arrivals[arrivals.length - 1];
            document.getElementById('tour-summary').innerHTML = `
                ${{activeTour.length}} stops | Last drop: ${{formatTime(lastArrival.arrivalTime)}} |
                All tours: ${{Object.keys(tours).length}} vehicle(s)
            `;
        }} else {{
            document.getElementById('tour-summary').innerHTML = `<span style="color:#666;">Click hubs on map to add stops. ${{Object.keys(tours).length}} tour(s) active.</span>`;
        }}
        
        // Update tour dropdown
        updateActiveTourDropdown();
        
        // Update cross-dock list
        renderCrossDockList();
        
        // Timing display
        if (selectedWarehouse) {{
            document.getElementById('timing-display').style.display = 'block';
            document.getElementById('td-pickup').textContent = document.getElementById('cfg-pickup').value;
            const depart = cfg.pickupTime + cfg.haltMin / 60;
            document.getElementById('td-depart').textContent = formatTime(depart);
        }}
    }}
    
    // =========================================================================
    // CROSS-DOCK MANAGEMENT
    // =========================================================================
    function toggleCrossDock(hubId) {{
        const hub = HUBS.find(h => h.id === hubId);
        if (!hub) return;
        
        if (crossDockHubs.has(hubId)) {{
            crossDockHubs.delete(hubId);
            const keysToRemove = Object.keys(tours).filter(k => k.startsWith(`cd_${{hubId}}_`));
            keysToRemove.forEach(k => delete tours[k]);
            if (activeTourKey.startsWith(`cd_${{hubId}}_`)) activeTourKey = 'direct_1';
        }} else {{
            crossDockHubs.add(hubId);
            tours[`cd_${{hubId}}_1`] = [];
        }}
        
        recalculate();
    }}
    
    function renderCrossDockList() {{
        const container = document.getElementById('crossdock-list');
        container.innerHTML = '';
        
        // Update CD target selector dropdown
        const cdSelect = document.getElementById('cd-target-select');
        cdSelect.innerHTML = '';
        
        if (crossDockHubs.size === 0) {{
            container.innerHTML = '<p style="font-size:11px;color:#666;">No cross-docks set. Right-click any hub to add one.</p>';
            cdSelect.style.display = 'none';
            return;
        }}
        
        // Show CD selector only when multiple CDs exist
        cdSelect.style.display = crossDockHubs.size > 1 ? 'inline-block' : 'none';
        
        crossDockHubs.forEach(cdId => {{
            const hub = HUBS.find(h => h.id === cdId);
            if (!hub) return;
            
            // Populate CD target dropdown
            const opt = document.createElement('option');
            opt.value = cdId;
            opt.textContent = `⬡ ${{hub.display_name}}`;
            cdSelect.appendChild(opt);
            
            const subtourKeys = Object.keys(tours).filter(k => k.startsWith(`cd_${{cdId}}_`));
            const div = document.createElement('div');
            div.style.cssText = 'background:#2a1040;border-radius:6px;padding:6px 8px;margin:4px 0;font-size:11px;';
            
            let arrivalInfo = '';
            if (selectedWarehouse) {{
                const arrTime = getCrossDockArrivalTime(cdId);
                if (arrTime) {{
                    const cfg = getConfig();
                    arrivalInfo = ` | Arrives: ${{formatTime(arrTime)}} | Departs: ${{formatTime(arrTime + cfg.crossDockMin/60)}}`;
                }} else {{
                    arrivalInfo = ' | <span style="color:#e94560;">Not on any direct tour</span>';
                }}
            }}
            
            div.innerHTML = `
                <span style="color:#ff00ff;">⬡ ${{hub.display_name}}</span>
                <span style="color:#888;">${{arrivalInfo}}</span>
                <span style="color:#888;"> | ${{subtourKeys.length}} sub-tour(s)</span>
                <button onclick="addSubtour(${{cdId}})" style="float:right;background:none;border:none;color:#00d4aa;cursor:pointer;font-size:10px;margin-left:4px;">+ Sub-tour</button>
                <button onclick="toggleCrossDock(${{cdId}})" style="float:right;background:none;border:none;color:#e94560;cursor:pointer;font-size:10px;">✕ Remove</button>
            `;
            container.appendChild(div);
        }});
    }}
    
    function updateActiveTourDropdown() {{
        const sel = document.getElementById('active-tour-select');
        const currentVal = activeTourKey;
        
        // Remove listener BEFORE rebuilding to prevent any async change events
        sel.removeEventListener('change', _tourSelectHandler);
        sel.innerHTML = '';
        
        // Direct tours
        const directKeys = Object.keys(tours).filter(k => k.startsWith('direct_')).sort();
        directKeys.forEach(key => {{
            const idx = key.split('_')[1];
            const stops = tours[key].length;
            const opt = document.createElement('option');
            opt.value = key;
            opt.textContent = `🚛 Direct Tour ${{idx}} (${{stops}} stops)`;
            sel.appendChild(opt);
        }});
        
        // CD sub-tours
        crossDockHubs.forEach(cdId => {{
            const hub = HUBS.find(h => h.id === cdId);
            if (!hub) return;
            const subtourKeys = Object.keys(tours).filter(k => k.startsWith(`cd_${{cdId}}_`)).sort();
            subtourKeys.forEach((key, idx) => {{
                const stops = tours[key].length;
                const opt = document.createElement('option');
                opt.value = key;
                opt.textContent = `⬡ CD Sub-tour ${{idx+1}}: ${{hub.display_name}} (${{stops}} stops)`;
                sel.appendChild(opt);
            }});
        }});
        
        // Restore selection — activeTourKey is the source of truth
        if ([...sel.options].some(o => o.value === currentVal)) {{
            sel.value = currentVal;
        }} else {{
            // Active tour was deleted — fall back to first option
            activeTourKey = sel.options.length > 0 ? sel.options[0].value : 'direct_1';
            sel.value = activeTourKey;
        }}
        
        // Re-add listener AFTER value is set and DOM is stable
        setTimeout(function() {{ sel.addEventListener('change', _tourSelectHandler); }}, 0);
    }}
    
    function _tourSelectHandler() {{
        const sel = document.getElementById('active-tour-select');
        const newVal = sel.value;
        if (newVal && newVal !== activeTourKey) {{
            activeTourKey = newVal;
            recalculate();
        }}
    }}
    
    function addNewDirectTour() {{
        const state = getActiveWHState();
        if (!state) {{ alert('Add a warehouse first!'); return; }}
        state.directTourCount++;
        const newKey = `direct_${{state.directTourCount}}`;
        state.tours[newKey] = [];
        state.activeTourKey = newKey;
        recalculate();
    }}
    
    function addSubtour(targetCdId) {{
        if (crossDockHubs.size === 0) {{
            alert('Add a cross-dock first! Right-click any hub on the map.');
            return;
        }}
        
        if (!targetCdId) {{
            // If currently on a CD sub-tour, use that CD
            if (activeTourKey.startsWith('cd_')) {{
                targetCdId = parseInt(activeTourKey.split('_')[1]);
            }} else if (crossDockHubs.size === 1) {{
                targetCdId = [...crossDockHubs][0];
            }} else {{
                // Multiple CDs: use the CD selector dropdown
                const sel = document.getElementById('cd-target-select');
                if (sel) targetCdId = parseInt(sel.value);
                if (!targetCdId) targetCdId = [...crossDockHubs][0];
            }}
        }}
        
        const existing = Object.keys(tours).filter(k => k.startsWith(`cd_${{targetCdId}}_`));
        const newKey = `cd_${{targetCdId}}_${{existing.length + 1}}`;
        tours[newKey] = [];
        activeTourKey = newKey;
        recalculate();
    }}
    
    function clearActiveTour() {{
        tours[activeTourKey] = [];
        recalculate();
    }}
    
    function deleteActiveTour() {{
        // Don't delete the last direct tour
        const directKeys = Object.keys(tours).filter(k => k.startsWith('direct_'));
        if (activeTourKey.startsWith('direct_') && directKeys.length <= 1) {{
            tours[activeTourKey] = [];
            recalculate();
            return;
        }}
        
        delete tours[activeTourKey];
        // Switch to first available tour
        const remaining = Object.keys(tours);
        activeTourKey = remaining[0] || 'direct_1';
        if (!tours[activeTourKey]) tours[activeTourKey] = [];
        recalculate();
    }}
    
    // =========================================================================
    // EVENT HANDLERS
    // =========================================================================
    // =========================================================================
    // MULTI-WAREHOUSE MANAGEMENT
    // =========================================================================
    function addWarehouseToNetwork() {{
        const sel = document.getElementById('warehouse-add-select');
        const whId = parseInt(sel.value);
        if (!whId || networkWarehouses[whId]) return;
        
        const wh = WAREHOUSES.find(w => w.id === whId);
        if (!wh) return;
        
        networkWarehouses[whId] = {{
            wh: wh,
            load: 500,
            tours: {{ direct_1: [] }},
            directTourCount: 1,
            crossDockHubs: new Set(),
            activeTourKey: 'direct_1',
        }};
        
        // Set as active if first warehouse added
        if (!activeWhId) activeWhId = whId;
        
        renderNetworkWHList();
        renderActiveWHDropdown();
        renderWarehouseMarkers();
        recalculate();
    }}
    
    function removeWarehouseFromNetwork(whId) {{
        delete networkWarehouses[whId];
        if (activeWhId === whId) {{
            const remaining = Object.keys(networkWarehouses);
            activeWhId = remaining.length > 0 ? parseInt(remaining[0]) : null;
        }}
        renderNetworkWHList();
        renderActiveWHDropdown();
        renderWarehouseMarkers();
        recalculate();
    }}
    
    function setActiveWarehouse(whId) {{
        if (!networkWarehouses[whId]) return;
        activeWhId = whId;
        renderNetworkWHList();
        renderActiveWHDropdown();
        recalculate();
    }}
    
    function updateWarehouseLoad(whId, load) {{
        if (!networkWarehouses[whId]) return;
        networkWarehouses[whId].load = parseInt(load) || 0;
        recalculate();
    }}
    
    function renderNetworkWHList() {{
        const container = document.getElementById('network-wh-list');
        container.innerHTML = '';
        
        const whIds = Object.keys(networkWarehouses);
        if (whIds.length === 0) {{
            document.getElementById('network-wh-summary').textContent = 'No warehouses in network';
            return;
        }}
        
        let totalLoad = 0;
        whIds.forEach(id => {{
            const whId = parseInt(id);
            const state = networkWarehouses[whId];
            const isActive = whId === activeWhId;
            totalLoad += state.load;
            
            const div = document.createElement('div');
            const bgColor = isActive ? '#1a3a5c' : '#16213e';
            const borderStyle = isActive ? '1px solid #00d4aa' : '1px solid transparent';
            const txtColor = isActive ? '#fff' : '#aaa';
            div.style.cssText = 'display:flex;align-items:center;gap:4px;padding:4px 6px;margin:2px 0;border-radius:4px;font-size:11px;background:'+bgColor+';border:'+borderStyle+';cursor:pointer;';
            div.innerHTML = '<span style="color:#ff6b35;font-weight:bold;width:14px;">W</span>' +
                '<span style="flex:1;color:'+txtColor+';overflow:hidden;text-overflow:ellipsis;white-space:nowrap;" onclick="setActiveWarehouse('+whId+')" title="'+state.wh.name+'">'+state.wh.name+'</span>' +
                '<input type="number" value="'+state.load+'" onchange="updateWarehouseLoad('+whId+', this.value)" style="width:55px;background:#0f3460;border:1px solid #333;color:#fff;padding:2px 4px;border-radius:3px;font-size:10px;text-align:right;" title="Orders from this WH">' +
                '<span style="color:#888;font-size:9px;">orders</span>' +
                '<button onclick="removeWarehouseFromNetwork('+whId+')" style="background:none;border:none;color:#e94560;cursor:pointer;font-size:12px;padding:0 2px;">✕</button>';
            container.appendChild(div);
        }});
        
        document.getElementById('network-wh-summary').innerHTML = '<b>'+whIds.length+'</b> warehouse(s) | Total load: <b style="color:#00d4aa;">'+totalLoad+'</b> orders';
    }}
    
    function renderActiveWHDropdown() {{
        const sel = document.getElementById('active-wh-select');
        sel.innerHTML = '';
        
        if (Object.keys(networkWarehouses).length === 0) {{
            const opt = document.createElement('option');
            opt.value = '';
            opt.textContent = '-- Add warehouses above --';
            sel.appendChild(opt);
            return;
        }}
        
        Object.keys(networkWarehouses).forEach(id => {{
            const whId = parseInt(id);
            const state = networkWarehouses[whId];
            const opt = document.createElement('option');
            opt.value = whId;
            opt.textContent = `${{state.wh.name}} (${{state.load}} orders)`;
            if (whId === activeWhId) opt.selected = true;
            sel.appendChild(opt);
        }});
    }}
    
    function onActiveWHChange() {{
        const sel = document.getElementById('active-wh-select');
        const whId = parseInt(sel.value);
        if (whId && networkWarehouses[whId]) {{
            activeWhId = whId;
            renderNetworkWHList();
            recalculate();
        }}
    }}
    
    function renderWarehouseMarkers() {{
        Object.values(warehouseMarkers).forEach(m => map.removeLayer(m));
        warehouseMarkers = {{}};
        
        Object.keys(networkWarehouses).forEach(id => {{
            const whId = parseInt(id);
            const state = networkWarehouses[whId];
            const isActive = whId === activeWhId;
            const sz = isActive ? 20 : 16;
            const bg = isActive ? '#ff6b35' : '#aa4422';
            const brd = isActive ? '#fff' : '#888';
            const fs = isActive ? 11 : 9;
            
            const marker = L.marker([state.wh.lat, state.wh.lng], {{
                icon: L.divIcon({{
                    html: '<div style="background:'+bg+';width:'+sz+'px;height:'+sz+'px;border-radius:3px;border:2px solid '+brd+';display:flex;align-items:center;justify-content:center;font-size:'+fs+'px;color:#fff;font-weight:bold;">W</div>',
                    iconSize: [sz, sz],
                    iconAnchor: [sz/2, sz/2],
                    className: ''
                }})
            }}).addTo(map);
            const popupText = '<b>'+state.wh.name+'</b><br>Load: '+state.load+' orders<br>'+(isActive ? '<b style="color:#00d4aa;">ACTIVE</b>' : '<i>Click to activate</i>');
            marker.bindPopup(popupText);
            marker.bindTooltip('W: '+state.wh.name+' ('+state.load+')', {{ direction: 'top', offset: [0, -10] }});
            marker.on('click', () => setActiveWarehouse(whId));
            warehouseMarkers[whId] = marker;
        }});
    }}
    
    function addToTour(hubId) {{
        if (!activeWhId || !networkWarehouses[activeWhId]) {{
            alert('Please add a warehouse to the network first!');
            return;
        }}
        
        const currentTour = tours[activeTourKey] || [];
        if (currentTour.includes(hubId)) {{
            tours[activeTourKey] = currentTour.filter(id => id !== hubId);
        }} else {{
            if (!tours[activeTourKey]) tours[activeTourKey] = [];
            tours[activeTourKey].push(hubId);
        }}
        recalculate();
    }}
    
    function clearTour() {{
        tours[activeTourKey] = [];
        recalculate();
    }}
    
    function undoLastStop() {{
        const t = tours[activeTourKey];
        if (t && t.length > 0) t.pop();
        recalculate();
    }}
    
    function optimizeTour() {{
        const t = tours[activeTourKey];
        if (!selectedWarehouse || !t || t.length < 2) return;
        
        const info = getTourDepartureInfo(activeTourKey);
        const optimized = [];
        const remaining = [...t];
        let currentLat = info.startLat;
        let currentLng = info.startLng;
        
        while (remaining.length > 0) {{
            let nearest = null;
            let nearestDist = Infinity;
            let nearestIdx = -1;
            
            remaining.forEach((hubId, idx) => {{
                const hub = HUBS.find(h => h.id === hubId);
                const d = roadDistance(currentLat, currentLng, hub.lat, hub.lng);
                if (d < nearestDist) {{
                    nearestDist = d;
                    nearest = hub;
                    nearestIdx = idx;
                }}
            }});
            
            optimized.push(remaining[nearestIdx]);
            currentLat = nearest.lat;
            currentLng = nearest.lng;
            remaining.splice(nearestIdx, 1);
        }}
        
        tours[activeTourKey] = optimized;
        recalculate();
    }}
    
    function recalculate() {{
        renderHubs();
        renderTourRoute();
        updateUI();
        autoDistributeIfNeeded();
        renderHubVolumeList();
        calculateCost();
    }}
    
    // Auto-distribute orders when hubs are added/removed from tours
    function autoDistributeIfNeeded() {{
        const state = getActiveWHState();
        if (!state) {{ hubOrderVolumes = {{}}; return; }}
        
        const allHubs = getAllHubsInTours();
        if (allHubs.length === 0) {{
            hubOrderVolumes = {{}};
            return;
        }}
        
        const total = state.load || 0;
        
        // Check if any hub in tours has no volume assigned
        const unassigned = allHubs.filter(id => !(id in hubOrderVolumes) || hubOrderVolumes[id] === 0);
        // Also remove volumes for hubs no longer in any tour
        const currentIds = new Set(allHubs);
        Object.keys(hubOrderVolumes).forEach(id => {{
            if (!currentIds.has(parseInt(id))) delete hubOrderVolumes[parseInt(id)];
        }});
        
        // If all hubs are unassigned, do even distribution
        if (unassigned.length === allHubs.length) {{
            const perHub = Math.floor(total / allHubs.length);
            let remainder = total - perHub * allHubs.length;
            allHubs.forEach((hubId, i) => {{
                hubOrderVolumes[hubId] = perHub + (i < remainder ? 1 : 0);
            }});
        }} else if (unassigned.length > 0) {{
            // New hubs added - redistribute remaining orders
            const assignedTotal = Object.values(hubOrderVolumes).reduce((s, v) => s + v, 0);
            const remaining = Math.max(0, total - assignedTotal);
            const perNew = unassigned.length > 0 ? Math.floor(remaining / unassigned.length) : 0;
            let rem = remaining - perNew * unassigned.length;
            unassigned.forEach((hubId, i) => {{
                hubOrderVolumes[hubId] = perNew + (i < rem ? 1 : 0);
            }});
        }}
    }}
    
    // =========================================================================
    // COST SIMULATION
    // =========================================================================
    
    // Per-hub order volumes for active WH (hubId -> order count)
    // This is a convenience reference recalculated per active warehouse
    let hubOrderVolumes = {{}};
    
    function getVehicleConfig() {{
        return [
            {{ type: 'Big', capacity: parseInt(document.getElementById('v-big-cap').value), cost: parseInt(document.getElementById('v-big-cost').value) }},
            {{ type: 'Medium', capacity: parseInt(document.getElementById('v-med-cap').value), cost: parseInt(document.getElementById('v-med-cost').value) }},
            {{ type: 'Small', capacity: parseInt(document.getElementById('v-small-cap').value), cost: parseInt(document.getElementById('v-small-cost').value) }},
        ];
    }}
    
    // Vehicle selection: finds cheapest combination that carries all shipments
    // Each vehicle NEVER exceeds its capacity
    function selectVehicles(shipments) {{
        if (shipments <= 0) return [];
        
        const vehicles = getVehicleConfig().sort((a, b) => b.capacity - a.capacity);
        
        // Generate all valid combinations and pick minimum cost
        // For efficiency, limit search: max vehicles of each type = ceil(shipments/capacity)
        let bestCombo = null;
        let bestCost = Infinity;
        
        const maxBig = Math.ceil(shipments / vehicles[0].capacity);
        const maxMed = Math.ceil(shipments / vehicles[1].capacity);
        const maxSmall = Math.ceil(shipments / vehicles[2].capacity);
        
        // Limit search space to reasonable bounds
        const capBig = Math.min(maxBig, 10);
        const capMed = Math.min(maxMed, 10);
        const capSmall = Math.min(maxSmall, 20);
        
        for (let b = 0; b <= capBig; b++) {{
            for (let m = 0; m <= capMed; m++) {{
                const totalCap = b * vehicles[0].capacity + m * vehicles[1].capacity;
                if (totalCap >= shipments) {{
                    // No smalls needed
                    const cost = b * vehicles[0].cost + m * vehicles[1].cost;
                    if (cost < bestCost) {{
                        bestCost = cost;
                        bestCombo = [b, m, 0];
                    }}
                    break; // More medium won't help
                }}
                // How many smalls needed?
                const need = shipments - totalCap;
                const s = Math.ceil(need / vehicles[2].capacity);
                const cost = b * vehicles[0].cost + m * vehicles[1].cost + s * vehicles[2].cost;
                if (cost < bestCost) {{
                    bestCost = cost;
                    bestCombo = [b, m, s];
                }}
            }}
        }}
        
        // Build result
        const result = [];
        const [bCount, mCount, sCount] = bestCombo;
        let assigned = shipments;
        
        if (bCount > 0) {{
            const carried = Math.min(assigned, bCount * vehicles[0].capacity);
            result.push({{ ...vehicles[0], count: bCount, shipmentsCarried: carried }});
            assigned -= carried;
        }}
        if (mCount > 0) {{
            const carried = Math.min(assigned, mCount * vehicles[1].capacity);
            result.push({{ ...vehicles[1], count: mCount, shipmentsCarried: carried }});
            assigned -= carried;
        }}
        if (sCount > 0) {{
            const carried = Math.min(assigned, sCount * vehicles[2].capacity);
            result.push({{ ...vehicles[2], count: sCount, shipmentsCarried: carried }});
        }}
        
        return result;
    }}
    
    function distributeOrdersEvenly() {{
        const state = getActiveWHState();
        if (!state) return;
        const total = state.load || 0;
        const allHubsInTours = getAllHubsInTours();
        if (allHubsInTours.length === 0) return;
        
        const perHub = Math.floor(total / allHubsInTours.length);
        let remainder = total - perHub * allHubsInTours.length;
        
        allHubsInTours.forEach((hubId, i) => {{
            hubOrderVolumes[hubId] = perHub + (i < remainder ? 1 : 0);
        }});
        
        recalculate();
    }}
    
    function distributeOrdersProportional() {{
        const state = getActiveWHState();
        if (!state) return;
        const total = state.load || 0;
        const allHubsInTours = getAllHubsInTours();
        if (allHubsInTours.length === 0 || !selectedWarehouse) return;
        
        let totalInvDist = 0;
        const invDists = {{}};
        allHubsInTours.forEach(hubId => {{
            const hub = HUBS.find(h => h.id === hubId);
            const dist = roadDistance(selectedWarehouse.lat, selectedWarehouse.lng, hub.lat, hub.lng);
            const invD = 1 / Math.max(dist, 0.5);
            invDists[hubId] = invD;
            totalInvDist += invD;
        }});
        
        let assigned = 0;
        allHubsInTours.forEach(hubId => {{
            const share = Math.round((invDists[hubId] / totalInvDist) * total);
            hubOrderVolumes[hubId] = share;
            assigned += share;
        }});
        
        if (assigned !== total && allHubsInTours.length > 0) {{
            hubOrderVolumes[allHubsInTours[0]] += (total - assigned);
        }}
        
        recalculate();
    }}
    
    function getAllHubsInTours() {{
        const all = new Set();
        Object.values(tours).forEach(t => t.forEach(id => all.add(id)));
        return [...all];
    }}
    
    function renderHubVolumeList() {{
        const container = document.getElementById('hub-volume-list');
        container.innerHTML = '';
        
        const allHubs = getAllHubsInTours();
        if (allHubs.length === 0) {{
            container.innerHTML = '<p style="color:#666;font-size:10px;">Add hubs to tours to set order volumes</p>';
            return;
        }}
        
        allHubs.forEach(hubId => {{
            const hub = HUBS.find(h => h.id === hubId);
            if (!hub) return;
            if (!(hubId in hubOrderVolumes)) hubOrderVolumes[hubId] = 0;
            
            const div = document.createElement('div');
            div.className = 'hub-vol-row';
            const typeIcon = hub.type === 'franchise_hub' ? '▪' : '●';
            div.innerHTML = `
                <span style="color:${{hub.type === 'franchise_hub' ? '#888' : '#00d4aa'}};">${{typeIcon}}</span>
                <span class="hub-name">${{hub.display_name}}</span>
                <input type="number" value="${{hubOrderVolumes[hubId]}}" 
                       onchange="updateHubVolume(${{hubId}}, this.value)" min="0">
            `;
            container.appendChild(div);
        }});
        
        // Show total assigned
        const totalAssigned = Object.values(hubOrderVolumes).reduce((s, v) => s + v, 0);
        const state = getActiveWHState();
        const totalOrders = state ? state.load : 0;
        const footer = document.createElement('div');
        footer.style.cssText = 'margin-top:4px;padding-top:4px;border-top:1px solid #333;color:#888;font-size:10px;';
        footer.innerHTML = `Assigned: <b style="color:${{totalAssigned === totalOrders ? '#00d4aa' : '#ffc107'}}">${{totalAssigned}}</b> / ${{totalOrders}} orders (active WH)`;
        container.appendChild(footer);
    }}
    
    function updateHubVolume(hubId, value) {{
        hubOrderVolumes[hubId] = parseInt(value) || 0;
        calculateCost();
    }}
    
    function calculateCost() {{
        if (Object.keys(networkWarehouses).length === 0) {{
            document.getElementById('cost-breakdown').innerHTML = '<p style="color:#666;">Add warehouses to network first</p>';
            return;
        }}
        
        const cfg = getConfig();
        const lmCostPerOrder = parseInt(document.getElementById('cfg-lm-cost').value);
        
        let fmCost = 0;
        let mmCost = 0;
        let lmCost = 0;
        let breakdownHTML = '';
        
        let totalShipmentsInNetwork = 0;
        // Sum all hub volumes across all warehouses
        Object.values(networkWarehouses).forEach(state => {{
            Object.values(state.tours).forEach(hubIds => {{
                hubIds.forEach(hId => {{ totalShipmentsInNetwork += (hubOrderVolumes[hId] || 0); }});
            }});
        }});
        // Simple fallback: use active WH volumes
        if (totalShipmentsInNetwork === 0) {{
            Object.values(hubOrderVolumes).forEach(v => totalShipmentsInNetwork += v);
        }}
        
        // Iterate over ALL warehouses for cost
        const allVehiclesUsed = [];
        let totalGreenTours = 0, totalYellowTours = 0, totalRedTours = 0;
        
        Object.keys(networkWarehouses).forEach(id => {{
            const whId = parseInt(id);
            const state = networkWarehouses[whId];
            
            breakdownHTML += `<div class="row section-header"><span class="label">📦 ${{state.wh.name}} (${{state.load}} orders)</span><span class="val"></span></div>`;
            
            // Temporarily switch context for calculations
            const prevActiveWhId = activeWhId;
            activeWhId = whId;
            
            // Direct tours
            const directKeys = Object.keys(state.tours).filter(k => k.startsWith('direct_'));
            directKeys.forEach(key => {{
                const tourHubs = state.tours[key] || [];
                if (tourHubs.length === 0) return;
                let tourShipments = 0;
                tourHubs.forEach(hId => {{ tourShipments += (hubOrderVolumes[hId] || 0); }});
                
                if (tourShipments > 0) {{
                    const chosen = selectTourVehicle(key);
                    const tourFMCost = chosen.totalCost;
                    fmCost += tourFMCost;
                    
                    const idx = key.split('_')[1];
                    const statusIcon = chosen.isGreen ? '🟢' : (chosen.isYellow ? '🟡' : '🔴');
                    const vDesc = `${{chosen.count}}x${{chosen.type}} @ ${{chosen.speed}}km/h`;
                    breakdownHTML += `<div class="row"><span class="label">${{statusIcon}} Tour ${{idx}}: ${{tourShipments}} ord</span><span class="val">₹${{tourFMCost}}</span></div>`;
                    breakdownHTML += `<div class="row"><span class="label" style="padding-left:10px;font-size:9px;color:#666;">${{vDesc}} | ${{formatTime(chosen.lastArrival)}}</span><span class="val"></span></div>`;
                    
                    allVehiclesUsed.push({{ type: chosen.type, count: chosen.count, capacity: chosen.capacity, cost: chosen.cost, shipmentsCarried: tourShipments }});
                    if (chosen.isGreen) totalGreenTours++; else if (chosen.isYellow) totalYellowTours++; else totalRedTours++;
                }}
            }});
            
            // CD sub-tours
            state.crossDockHubs.forEach(cdId => {{
                Object.keys(state.tours).filter(k => k.startsWith(`cd_${{cdId}}_`)).forEach(key => {{
                    const subTour = state.tours[key];
                    if (!subTour || subTour.length === 0) return;
                    let subShipments = 0;
                    subTour.forEach(hId => {{ subShipments += (hubOrderVolumes[hId] || 0); }});
                    
                    if (subShipments > 0) {{
                        const chosen = selectTourVehicle(key);
                        const subMMCost = chosen.totalCost;
                        mmCost += subMMCost;
                        
                        const cdHub = HUBS.find(h => h.id === cdId);
                        const statusIcon = chosen.isGreen ? '🟢' : (chosen.isYellow ? '🟡' : '🔴');
                        breakdownHTML += `<div class="row"><span class="label">${{statusIcon}} CD→${{cdHub ? cdHub.display_name : cdId}}: ${{subShipments}} ord</span><span class="val">₹${{subMMCost}}</span></div>`;
                        
                        allVehiclesUsed.push({{ type: chosen.type, count: chosen.count, capacity: chosen.capacity, cost: chosen.cost, shipmentsCarried: subShipments }});
                        if (chosen.isGreen) totalGreenTours++; else if (chosen.isYellow) totalYellowTours++; else totalRedTours++;
                    }}
                }});
            }});
            
            activeWhId = prevActiveWhId;
        }});
        
        // === LAST MILE (fixed per order) ===
        lmCost = totalShipmentsInNetwork * lmCostPerOrder;
        breakdownHTML += '<div class="row section-header"><span class="label">LAST MILE (Delivery)</span><span class="val"></span></div>';
        breakdownHTML += `<div class="row"><span class="label">${{totalShipmentsInNetwork}} orders × ₹${{lmCostPerOrder}}</span><span class="val">₹${{lmCost}}</span></div>`;
        
        // === TOTAL ===
        const totalCost = fmCost + mmCost + lmCost;
        const costPerOrder = totalShipmentsInNetwork > 0 ? (totalCost / totalShipmentsInNetwork).toFixed(1) : 0;
        
        breakdownHTML += '<div class="row section-header"><span class="label">NETWORK TOTAL</span><span class="val"></span></div>';
        breakdownHTML += `<div class="row"><span class="label">FM + MM + LM</span><span class="val" style="color:#00d4aa;font-size:13px;">₹${{totalCost}}</span></div>`;
        breakdownHTML += `<div class="row"><span class="label">Cost per Order</span><span class="val" style="color:#ffc107;">₹${{costPerOrder}}</span></div>`;
        breakdownHTML += `<div class="row"><span class="label">Warehouses in Network</span><span class="val">${{Object.keys(networkWarehouses).length}}</span></div>`;
        
        // Vehicle summary
        const totalVehicles = allVehiclesUsed.reduce((s, v) => s + v.count, 0);
        const totalCapacity = allVehiclesUsed.reduce((s, v) => s + v.count * v.capacity, 0);
        const utilPct = totalCapacity > 0 ? ((totalShipmentsInNetwork / totalCapacity) * 100).toFixed(0) : 0;
        breakdownHTML += '<div class="row section-header"><span class="label">VEHICLE SUMMARY</span><span class="val"></span></div>';
        breakdownHTML += `<div class="row"><span class="label">Total Vehicles</span><span class="val">${{totalVehicles}}</span></div>`;
        breakdownHTML += `<div class="row"><span class="label">Utilization</span><span class="val" style="color:${{utilPct > 70 ? '#00d4aa' : '#ffc107'}}">${{utilPct}}%</span></div>`;
        
        // Tour feasibility summary
        const totalTours = totalGreenTours + totalYellowTours + totalRedTours;
        if (totalTours > 0) {{
            breakdownHTML += '<div class="row section-header"><span class="label">TOUR FEASIBILITY</span><span class="val"></span></div>';
            breakdownHTML += `<div class="row"><span class="label">🟢 Green</span><span class="val">${{totalGreenTours}} / ${{totalTours}}</span></div>`;
            if (totalYellowTours > 0) breakdownHTML += `<div class="row"><span class="label">🟡 Yellow</span><span class="val">${{totalYellowTours}} / ${{totalTours}}</span></div>`;
            if (totalRedTours > 0) breakdownHTML += `<div class="row"><span class="label">🔴 Red</span><span class="val">${{totalRedTours}} / ${{totalTours}}</span></div>`;
        }}
        
        // Update cards
        document.getElementById('cost-fm').textContent = `₹${{fmCost}}`;
        document.getElementById('cost-mm').textContent = `₹${{mmCost}}`;
        document.getElementById('cost-lm').textContent = `₹${{lmCost}}`;
        document.getElementById('cost-per-order').textContent = `₹${{costPerOrder}}`;
        document.getElementById('cost-total').textContent = `₹${{totalCost}}`;
        document.getElementById('cost-breakdown').innerHTML = breakdownHTML;
    }}
    
    // =========================================================================
    // INIT
    // =========================================================================
    function init() {{
        // Populate warehouse add dropdown
        const sel = document.getElementById('warehouse-add-select');
        const sorted = [...WAREHOUSES].sort((a, b) => a.name.localeCompare(b.name));
        sorted.forEach(wh => {{
            const opt = document.createElement('option');
            opt.value = wh.id;
            opt.textContent = `${{wh.name}}`;
            sel.appendChild(opt);
        }});
        
        // Attach event handlers
        document.getElementById('active-tour-select').addEventListener('change', _tourSelectHandler);
        document.getElementById('active-wh-select').addEventListener('change', onActiveWHChange);
        
        renderHubs();
        renderNetworkWHList();
        renderActiveWHDropdown();
    }}
    
    init();
    </script>
</body>
</html>
"""

with open(OUTPUT_PATH, 'w') as f:
    f.write(html_content)

print(f"\n✅ Interactive tour builder saved to:\n   {OUTPUT_PATH}")
print(f"\nFeatures:")
print(f"  - Select any of {len(wh_data)} warehouses")
print(f"  - See {len(hubs_data)} hubs colored Green/Yellow/Red")
print(f"  - Build tours by clicking hubs on map")
print(f"  - Auto-optimize tour order (nearest neighbor)")
print(f"  - All timing parameters are configurable")
print(f"  - Cross-docking center placeholder ready")
