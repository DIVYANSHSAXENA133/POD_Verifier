"""
Bangalore 5PM Same-Day Delivery Network Simulation
===================================================
This script simulates a delivery network for 5PM same-day delivery in Bangalore.
It maps warehouses, mother hubs, last-mile hubs, and quick-commerce hubs,
then designs network connections and validates timing feasibility.

Data Sources:
- City Config.xlsx: Cutoff timings (first mile → mother hub → last mile)
- query_result: Node locations with addresses and types
"""

import pandas as pd
import folium
from folium import plugins
import json
import re
from datetime import datetime, timedelta
import math

# =============================================================================
# SECTION 1: DATA LOADING
# =============================================================================

CITY_CONFIG_PATH = '/Users/divyanshsaxena/Downloads/City Config.xlsx'
QUERY_RESULT_PATH = '/Users/divyanshsaxena/Downloads/query_result_2026-05-25T13_38_36.614839528Z.xlsx'
OUTPUT_DIR = '/Users/divyanshsaxena/Desktop/POD_Verifier/network_simulation'

# Load data
nodes_df = pd.read_excel(QUERY_RESULT_PATH)
warehouses_df = pd.read_excel(CITY_CONFIG_PATH, sheet_name='Sheet4')
prod_config_df = pd.read_excel(CITY_CONFIG_PATH, sheet_name='Production Intercity Config')
test_config_df = pd.read_excel(CITY_CONFIG_PATH, sheet_name='Testing Intercity Config')

# Filter Bangalore
blr_nodes = nodes_df[nodes_df['City Name'] == 'Bangalore'].copy()
blr_warehouses = warehouses_df[warehouses_df['warehouse_city'] == 'Bangalore'].copy()
blr_prod_config = prod_config_df[prod_config_df['city'] == 'BLR'].copy()
blr_test_config = test_config_df[test_config_df['city'] == 'BLR'].copy()

print(f"Bangalore Nodes: {len(blr_nodes)}")
print(f"  - LM Hubs: {len(blr_nodes[blr_nodes['Node Type'] == 'lm_hub'])}")
print(f"  - Quick Hubs: {len(blr_nodes[blr_nodes['Node Type'] == 'quick_hub'])}")
print(f"Bangalore Warehouses: {len(blr_warehouses)}")

# =============================================================================
# SECTION 2: GEOCODING (Approximate coordinates from Bangalore areas/pincodes)
# =============================================================================

# Known Bangalore area coordinates (lat, lng) based on pincodes and area names
BANGALORE_AREA_COORDS = {
    # Central/East
    'marathahalli': (12.9591, 77.6974),
    'whitefield': (12.9698, 77.7500),
    'domlur': (12.9610, 77.6387),
    'hsr layout': (12.9116, 77.6389),
    'hsr': (12.9116, 77.6389),
    'koramangala': (12.9352, 77.6245),
    'indiranagar': (12.9784, 77.6408),
    
    # North
    'hebbal': (13.0358, 77.5970),
    'nagasandra': (13.0485, 77.5170),
    'mathikere': (13.0200, 77.5700),
    'chokkanahalli': (13.0700, 77.5900),
    'yelahanka': (13.1007, 77.5963),
    
    # South
    'btm layout': (12.9166, 77.6101),
    'jayanagar': (12.9250, 77.5838),
    'jp nagar': (12.9063, 77.5857),
    'konanakunte': (12.8878, 77.5737),
    'bannerghatta': (12.8876, 77.5973),
    'bommasandra': (12.8160, 77.6940),
    'chandapura': (12.8015, 77.7070),
    'begur': (12.8720, 77.6340),
    'electronic city': (12.8456, 77.6603),
    
    # West
    'rajarajeshwari nagar': (12.9200, 77.5190),
    'vijayanagar': (12.9716, 77.5366),
    'chamrajpet': (12.9600, 77.5650),
    'peenya': (13.0300, 77.5200),
    
    # Area specific by pincode
    '560037': (12.9591, 77.6974),  # Marathahalli
    '560098': (12.9200, 77.5190),  # RR Nagar
    '560073': (13.0485, 77.5170),  # Nagasandra
    '560054': (13.0200, 77.5700),  # Mathikere
    '560102': (12.9116, 77.6389),  # HSR Layout
    '560076': (12.9166, 77.6101),  # BTM Layout
    '560062': (12.8878, 77.5737),  # Konanakunte
    '560041': (12.9250, 77.5838),  # Jayanagar
    '560099': (12.8160, 77.6940),  # Bommasandra
    '560103': (12.9570, 77.7150),  # Devarabeesanahalli
    '560066': (12.9698, 77.7500),  # Whitefield
    '560071': (12.9610, 77.6387),  # Domlur
    '560032': (13.0358, 77.5970),  # Hebbal
    '560018': (12.9600, 77.5650),  # Chamrajpet
    '560064': (13.0700, 77.5900),  # Chokkanahalli
    '560114': (12.8720, 77.6340),  # Begur
}

# Mother Hub coordinates (Marathahalli is the main BLR mother hub)
MOTHER_HUB_COORDS = {
    'BLR_MOTHER_HUB': (12.9591, 77.6974),  # Marathahalli area
}

# Warehouse cluster coordinates (approximate based on known warehouse locations)
WAREHOUSE_CLUSTERS = {
    'koraluru': (13.1100, 77.7200),  # Near airport/rural
    'thattanahalli': (13.1200, 77.6800),  # Hoskote area
    'anugondanahalli': (13.1000, 77.7500),  # Rural east
    'malonagathihalli': (13.0800, 77.7000),  # Near Whitefield
    'marasur': (12.8200, 77.6500),  # South Bangalore
    'jigni': (12.7800, 77.6200),  # Far south
    'bommasandra_wh': (12.8200, 77.6800),  # Industrial area
    'marathahalli_wh': (12.9591, 77.6974),  # Central
    'hoskote': (13.0700, 77.7900),  # East
    'lukkuru': (13.1300, 77.6500),  # North
}


def extract_pincode(address):
    """Extract 6-digit pincode from address string."""
    if pd.isna(address):
        return None
    matches = re.findall(r'56\d{4}', str(address))
    return matches[0] if matches else None


def geocode_node(row):
    """Approximate geocoding based on address keywords and pincodes."""
    address = str(row.get('Address', '')).lower()
    name = str(row.get('Node Name', '')).lower()
    display = str(row.get('Display Name', '')).lower()
    
    # Try pincode first
    pincode = extract_pincode(row.get('Address', ''))
    if pincode and pincode in BANGALORE_AREA_COORDS:
        base = BANGALORE_AREA_COORDS[pincode]
        # Add small random offset to avoid stacking
        import random
        random.seed(hash(str(row.get('Node ID', 0))))
        offset = (random.uniform(-0.005, 0.005), random.uniform(-0.005, 0.005))
        return (base[0] + offset[0], base[1] + offset[1])
    
    # Try area name matching
    for area, coords in BANGALORE_AREA_COORDS.items():
        if area in address or area in name or area in display:
            import random
            random.seed(hash(str(row.get('Node ID', 0))))
            offset = (random.uniform(-0.003, 0.003), random.uniform(-0.003, 0.003))
            return (coords[0] + offset[0], coords[1] + offset[1])
    
    # Default to Bangalore center with offset
    import random
    random.seed(hash(str(row.get('Node ID', 0))))
    return (12.9716 + random.uniform(-0.02, 0.02), 77.5946 + random.uniform(-0.02, 0.02))


def geocode_warehouse(row):
    """Approximate geocoding for warehouses based on name/address keywords."""
    name = str(row.get('warehouse_name', '')).lower()
    
    for area, coords in WAREHOUSE_CLUSTERS.items():
        if area in name:
            import random
            random.seed(hash(str(row.get('warehouse_id', 0))))
            offset = (random.uniform(-0.005, 0.005), random.uniform(-0.005, 0.005))
            return (coords[0] + offset[0], coords[1] + offset[1])
    
    # Check for specific keywords
    if 'koraluru' in name or 'rural' in name:
        base = WAREHOUSE_CLUSTERS['koraluru']
    elif 'thattanahalli' in name or 'hoskote' in name:
        base = WAREHOUSE_CLUSTERS['thattanahalli']
    elif 'anugondanahalli' in name:
        base = WAREHOUSE_CLUSTERS['anugondanahalli']
    elif 'malonagathihalli' in name:
        base = WAREHOUSE_CLUSTERS['malonagathihalli']
    elif 'marasur' in name:
        base = WAREHOUSE_CLUSTERS['marasur']
    elif 'jigni' in name:
        base = WAREHOUSE_CLUSTERS['jigni']
    elif 'bommasandra' in name:
        base = WAREHOUSE_CLUSTERS['bommasandra_wh']
    elif 'marathahalli' in name or 'outer ring' in name:
        base = WAREHOUSE_CLUSTERS['marathahalli_wh']
    elif 'lukkuru' in name:
        base = WAREHOUSE_CLUSTERS['lukkuru']
    else:
        # Distribute around industrial areas (north-east Bangalore)
        import random
        random.seed(hash(str(row.get('warehouse_id', 0))))
        base = (13.0000 + random.uniform(-0.08, 0.08), 77.6500 + random.uniform(-0.08, 0.08))
        return base
    
    import random
    random.seed(hash(str(row.get('warehouse_id', 0))))
    offset = (random.uniform(-0.005, 0.005), random.uniform(-0.005, 0.005))
    return (base[0] + offset[0], base[1] + offset[1])


# Geocode all Bangalore nodes
blr_nodes['coords'] = blr_nodes.apply(geocode_node, axis=1)
blr_nodes['lat'] = blr_nodes['coords'].apply(lambda x: x[0])
blr_nodes['lng'] = blr_nodes['coords'].apply(lambda x: x[1])

# Geocode warehouses (sample top ones with different cutoff times)
blr_warehouses['coords'] = blr_warehouses.apply(geocode_warehouse, axis=1)
blr_warehouses['lat'] = blr_warehouses['coords'].apply(lambda x: x[0])
blr_warehouses['lng'] = blr_warehouses['coords'].apply(lambda x: x[1])

print("\nGeocoding complete.")
print(f"Nodes geocoded: {len(blr_nodes)}")
print(f"Warehouses geocoded: {len(blr_warehouses)}")

# =============================================================================
# SECTION 3: NETWORK DESIGN - 5PM SAME DAY DELIVERY
# =============================================================================

print("\n" + "="*70)
print("5PM SAME-DAY DELIVERY NETWORK DESIGN - BANGALORE")
print("="*70)

# Current timing configuration
print("\n--- CURRENT PRODUCTION CONFIG (BLR Evening Slot) ---")
print(blr_prod_config.to_string())

print("\n--- CURRENT TESTING CONFIG (BLR Evening Slot) ---")
print(blr_test_config.to_string())

# Proposed 5PM SDD timing
print("\n--- PROPOSED 5PM SAME-DAY DELIVERY TIMING ---")
proposed_timing = {
    'first_mile_pickup_start': '12:00',  # Vehicle leaves for warehouse pickup
    'warehouse_pickup_cutoff': '17:00',  # 5PM - NEW CUTOFF
    'vehicle_reach_motherhub': '18:00',  # 1 hour transit to mother hub
    'motherhub_mm_dispatch': '18:30',    # 30 min for sorting at mother hub
    'last_mile_arrival': '19:30',        # 1 hour to reach last mile hub
    'delivery_completion': '21:00',      # 1.5 hours for last mile delivery
}

print(f"\n{'Stage':<40} {'Time':<10} {'Duration':<15}")
print("-" * 65)
stages = [
    ('First Mile: Vehicle leaves for pickup', proposed_timing['first_mile_pickup_start'], '-'),
    ('First Mile: Warehouse Pickup CUTOFF', proposed_timing['warehouse_pickup_cutoff'], '5 hours window'),
    ('Transit: Vehicle reaches Mother Hub', proposed_timing['vehicle_reach_motherhub'], '1 hour'),
    ('Mother Hub: MM Dispatch Cutoff', proposed_timing['motherhub_mm_dispatch'], '30 minutes'),
    ('Last Mile: Arrival at LM Hub', proposed_timing['last_mile_arrival'], '1 hour'),
    ('Last Mile: Delivery Completion', proposed_timing['delivery_completion'], '1.5 hours'),
]
for stage, time, duration in stages:
    print(f"{stage:<40} {time:<10} {duration:<15}")

# =============================================================================
# SECTION 4: HAVERSINE DISTANCE CALCULATION
# =============================================================================

def haversine_distance(lat1, lon1, lat2, lon2):
    """Calculate distance between two points in km."""
    R = 6371
    lat1, lon1, lat2, lon2 = map(math.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = math.sin(dlat/2)**2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon/2)**2
    c = 2 * math.asin(math.sqrt(a))
    return R * c


def estimate_transit_time(distance_km, avg_speed_kmph=25):
    """Estimate transit time in minutes given distance and avg city speed."""
    return (distance_km / avg_speed_kmph) * 60

# =============================================================================
# SECTION 5: BUILD NETWORK CONNECTIONS
# =============================================================================

# Mother Hub (Marathahalli is the main sorting center)
mother_hub = {
    'name': 'BLR Mother Hub (Marathahalli)',
    'lat': 12.9591,
    'lng': 77.6974,
    'type': 'mother_hub'
}

# LM Hubs
lm_hubs = blr_nodes[blr_nodes['Node Type'] == 'lm_hub'].copy()
quick_hubs = blr_nodes[blr_nodes['Node Type'] == 'quick_hub'].copy()

# Calculate distances from mother hub to each LM hub
lm_hubs['dist_from_mh'] = lm_hubs.apply(
    lambda r: haversine_distance(mother_hub['lat'], mother_hub['lng'], r['lat'], r['lng']), axis=1
)
lm_hubs['transit_time_min'] = lm_hubs['dist_from_mh'].apply(estimate_transit_time)

# Calculate distances from warehouses to mother hub
blr_warehouses['dist_to_mh'] = blr_warehouses.apply(
    lambda r: haversine_distance(r['lat'], r['lng'], mother_hub['lat'], mother_hub['lng']), axis=1
)
blr_warehouses['transit_to_mh_min'] = blr_warehouses['dist_to_mh'].apply(estimate_transit_time)

print("\n--- LAST MILE HUBS - Distance from Mother Hub ---")
for _, hub in lm_hubs.iterrows():
    print(f"  {hub['Display Name']:<35} {hub['dist_from_mh']:.1f} km  |  {hub['transit_time_min']:.0f} min transit")

# Feasibility analysis
print("\n--- 5PM SDD FEASIBILITY BY WAREHOUSE CLUSTER ---")
print(f"\n{'Warehouse Cluster':<30} {'Dist to MH':<12} {'Transit':<10} {'Feasible?':<10}")
print("-" * 62)

# Group warehouses by pickup cutoff
cutoff_groups = blr_warehouses.groupby('pickup_cutoff').agg({
    'warehouse_id': 'count',
    'dist_to_mh': 'mean',
    'transit_to_mh_min': 'mean'
}).reset_index()

for _, group in cutoff_groups.iterrows():
    cutoff = group['pickup_cutoff']
    avg_dist = group['dist_to_mh']
    avg_transit = group['transit_to_mh_min']
    count = group['warehouse_id']
    # With 5PM cutoff, vehicle needs to reach MH before 6PM
    feasible = avg_transit <= 60  # 1 hour window
    print(f"  Cutoff {cutoff} ({int(count)} WHs) {avg_dist:>8.1f} km  {avg_transit:>6.0f} min  {'YES' if feasible else 'NO - needs route optimization'}")

# =============================================================================
# SECTION 6: GENERATE INTERACTIVE MAP
# =============================================================================

print("\n\nGenerating interactive map...")

# Create base map centered on Bangalore
m = folium.Map(
    location=[12.9716, 77.5946],
    zoom_start=11,
    tiles='CartoDB positron'
)

# Add title
title_html = '''
<div style="position: fixed; top: 10px; left: 50%; transform: translateX(-50%); 
     z-index: 1000; background: white; padding: 10px 20px; border-radius: 8px;
     box-shadow: 0 2px 10px rgba(0,0,0,0.2); font-family: Arial;">
    <h3 style="margin:0; color: #333;">Bangalore 5PM Same-Day Delivery Network</h3>
    <p style="margin:2px 0 0 0; font-size: 12px; color: #666;">
        First Mile Pickup Cutoff: 5:00 PM | Target Delivery: 9:00 PM
    </p>
</div>
'''
m.get_root().html.add_child(folium.Element(title_html))

# --- Layer: Mother Hub ---
mother_hub_group = folium.FeatureGroup(name='🏭 Mother Hub (Sorting Center)', show=True)
folium.Marker(
    location=[mother_hub['lat'], mother_hub['lng']],
    popup=folium.Popup(
        f"<b>{mother_hub['name']}</b><br>"
        f"Type: Central Sorting Hub<br>"
        f"MM Dispatch Cutoff: 6:30 PM (proposed)",
        max_width=300
    ),
    icon=folium.Icon(color='black', icon='industry', prefix='fa'),
    tooltip=mother_hub['name']
).add_to(mother_hub_group)
# Add coverage circle around mother hub
folium.Circle(
    location=[mother_hub['lat'], mother_hub['lng']],
    radius=3000,
    color='black',
    fill=True,
    fill_opacity=0.05,
    weight=2,
    dash_array='5,5'
).add_to(mother_hub_group)
mother_hub_group.add_to(m)

# --- Layer: Last Mile Hubs ---
lm_hub_group = folium.FeatureGroup(name='📦 Last Mile Hubs', show=True)
for _, hub in lm_hubs.iterrows():
    color = 'blue'
    folium.Marker(
        location=[hub['lat'], hub['lng']],
        popup=folium.Popup(
            f"<b>{hub['Display Name']}</b><br>"
            f"Type: Last Mile Hub<br>"
            f"Sort Code: {hub['Sort Codes']}<br>"
            f"Clusters: {hub['Clusters']}<br>"
            f"Dist from MH: {hub['dist_from_mh']:.1f} km<br>"
            f"Transit Time: {hub['transit_time_min']:.0f} min",
            max_width=300
        ),
        icon=folium.Icon(color=color, icon='truck', prefix='fa'),
        tooltip=f"LM: {hub['Display Name']}"
    ).add_to(lm_hub_group)
    # Draw connection to mother hub
    folium.PolyLine(
        locations=[
            [mother_hub['lat'], mother_hub['lng']],
            [hub['lat'], hub['lng']]
        ],
        color='blue',
        weight=2,
        opacity=0.6,
        dash_array='10,5',
        tooltip=f"MH → {hub['Display Name']}: {hub['dist_from_mh']:.1f}km, {hub['transit_time_min']:.0f}min"
    ).add_to(lm_hub_group)
lm_hub_group.add_to(m)

# --- Layer: Quick Hubs (Franchises) ---
qh_group = folium.FeatureGroup(name='⚡ Quick Hubs (Franchises)', show=True)
for _, hub in quick_hubs.iterrows():
    folium.CircleMarker(
        location=[hub['lat'], hub['lng']],
        radius=6,
        color='green',
        fill=True,
        fill_color='green',
        fill_opacity=0.7,
        popup=folium.Popup(
            f"<b>{hub['Display Name']}</b><br>"
            f"Type: Quick Hub<br>"
            f"Sort Code: {hub['Sort Codes']}<br>"
            f"Address: {hub['Address']}",
            max_width=300
        ),
        tooltip=f"QH: {hub['Display Name']}"
    ).add_to(qh_group)
qh_group.add_to(m)

# --- Layer: Warehouses (color-coded by cutoff time) ---
# Get unique pickup cutoffs and assign colors
wh_group = folium.FeatureGroup(name='🏬 Pickup Warehouses', show=True)

# Color warehouses by their current cutoff relative to 5PM feasibility
for _, wh in blr_warehouses.iterrows():
    cutoff = wh['pickup_cutoff']
    # Parse cutoff time
    try:
        if pd.notna(cutoff):
            cutoff_str = str(cutoff)
            if ':' in cutoff_str:
                parts = cutoff_str.split(':')
                hour = int(parts[0])
            else:
                hour = 12
        else:
            hour = 12
    except:
        hour = 12
    
    # Color based on whether warehouse can meet 5PM cutoff
    if hour <= 12:
        color = 'red'  # Currently early cutoff - needs extension to 5PM
        status = f'Current cutoff: {cutoff} (needs extension)'
    elif hour <= 15:
        color = 'orange'  # Afternoon cutoff
        status = f'Current cutoff: {cutoff} (moderate extension needed)'
    else:
        color = 'darkgreen'  # Already close to or past 5PM
        status = f'Current cutoff: {cutoff} (already compatible)'
    
    folium.CircleMarker(
        location=[wh['lat'], wh['lng']],
        radius=4,
        color=color,
        fill=True,
        fill_color=color,
        fill_opacity=0.5,
        popup=folium.Popup(
            f"<b>{wh['warehouse_name']}</b><br>"
            f"User: {wh['User ']}<br>"
            f"ID: {wh['warehouse_id']}<br>"
            f"Current Pickup Cutoff: {cutoff}<br>"
            f"Status: {status}<br>"
            f"Dist to Mother Hub: {wh['dist_to_mh']:.1f} km<br>"
            f"Transit to MH: {wh['transit_to_mh_min']:.0f} min",
            max_width=350
        ),
        tooltip=f"WH: {wh['warehouse_name'][:30]}"
    ).add_to(wh_group)
wh_group.add_to(m)

# --- Layer: Network Routes (Warehouse → Mother Hub) ---
route_group = folium.FeatureGroup(name='🔗 First Mile Routes (WH → MH)', show=False)

# Draw routes for top warehouses by volume (unique users)
top_users = blr_warehouses.groupby('User ').agg({
    'warehouse_id': 'count',
    'lat': 'first',
    'lng': 'first'
}).nlargest(20, 'warehouse_id')

for user, data in top_users.iterrows():
    folium.PolyLine(
        locations=[
            [data['lat'], data['lng']],
            [mother_hub['lat'], mother_hub['lng']]
        ],
        color='red',
        weight=1.5,
        opacity=0.4,
        tooltip=f"FM Route: {user} → Mother Hub"
    ).add_to(route_group)
route_group.add_to(m)

# --- Layer: Proposed 5PM Network Coverage ---
coverage_group = folium.FeatureGroup(name='📍 5PM Delivery Coverage Zone', show=True)

# Draw coverage circles around LM hubs (delivery radius)
for _, hub in lm_hubs.iterrows():
    folium.Circle(
        location=[hub['lat'], hub['lng']],
        radius=5000,  # 5km delivery radius
        color='blue',
        fill=True,
        fill_opacity=0.03,
        weight=1,
        tooltip=f"Coverage: {hub['Display Name']} (5km radius)"
    ).add_to(coverage_group)

for _, hub in quick_hubs.iterrows():
    folium.Circle(
        location=[hub['lat'], hub['lng']],
        radius=3000,  # 3km quick delivery radius
        color='green',
        fill=True,
        fill_opacity=0.03,
        weight=1,
    ).add_to(coverage_group)
coverage_group.add_to(m)

# Add layer control
folium.LayerControl(collapsed=False).add_to(m)

# Add legend
legend_html = '''
<div style="position: fixed; bottom: 30px; left: 30px; z-index: 1000;
     background: white; padding: 12px; border-radius: 8px;
     box-shadow: 0 2px 10px rgba(0,0,0,0.2); font-family: Arial; font-size: 12px;">
    <b style="font-size: 13px;">Legend</b><br><br>
    <i style="color: black;">■</i> Mother Hub (Sorting Center)<br>
    <i style="color: blue;">●</i> Last Mile Hubs<br>
    <i style="color: green;">●</i> Quick Hubs (Franchises)<br>
    <br><b>Warehouses by 5PM Feasibility:</b><br>
    <i style="color: red;">●</i> Needs cutoff extension (currently < 12PM)<br>
    <i style="color: orange;">●</i> Moderate extension needed (12-3PM)<br>
    <i style="color: darkgreen;">●</i> Already compatible (> 3PM)<br>
    <br><b>Network Lines:</b><br>
    <span style="color: blue;">- - -</span> Mother Hub → LM Hub<br>
    <span style="color: red;">───</span> Warehouse → Mother Hub (FM)<br>
</div>
'''
m.get_root().html.add_child(folium.Element(legend_html))

# Save map
map_path = f"{OUTPUT_DIR}/bangalore_5pm_sdd_network_map.html"
m.save(map_path)
print(f"\nMap saved to: {map_path}")

# =============================================================================
# SECTION 7: TIMING ANALYSIS SUMMARY
# =============================================================================

print("\n" + "="*70)
print("TIMING ANALYSIS: 5PM SAME-DAY DELIVERY FEASIBILITY")
print("="*70)

print("""
┌─────────────────────────────────────────────────────────────────────┐
│                    CURRENT vs PROPOSED TIMING                         │
├─────────────────────────────────────────────────────────────────────┤
│                                                                       │
│  CURRENT (Evening Slot):                                              │
│  ├─ Warehouse Pickup Cutoff:    1:00 PM                              │
│  ├─ Vehicle Reaches Mother Hub: 2:00 PM                              │
│  ├─ MM Dispatch Cutoff:         3:00 PM                              │
│  ├─ Last Mile Arrival:          5:00 PM (4:00 PM testing)            │
│  └─ Delivery Window:            Until 8:00 PM                        │
│                                                                       │
│  PROPOSED (5PM SDD Slot):                                             │
│  ├─ First Mile Vehicle Departs: 12:00 PM (staggered pickups)         │
│  ├─ Warehouse Pickup Cutoff:    5:00 PM  ← NEW                      │
│  ├─ Vehicle Reaches Mother Hub: 6:00 PM  (1hr transit)               │
│  ├─ MM Dispatch Cutoff:         6:30 PM  (30min sorting)             │
│  ├─ Last Mile Arrival:          7:30 PM  (1hr distribution)          │
│  └─ Delivery Completion:        9:00 PM  (1.5hr last mile)           │
│                                                                       │
├─────────────────────────────────────────────────────────────────────┤
│  KEY CHANGES REQUIRED:                                                │
│  1. Extend warehouse pickup cutoff from 1PM → 5PM                    │
│  2. Add afternoon pickup run (staggered at 12PM, 2PM, 4PM)           │
│  3. Mother hub sorting capacity for 6-6:30PM window                  │
│  4. Last mile riders available for 7:30-9PM delivery window           │
│  5. Extended hub operating hours until 9:30 PM                       │
└─────────────────────────────────────────────────────────────────────┘
""")

# Warehouse impact analysis
print("\n--- WAREHOUSE IMPACT ANALYSIS ---")
print(f"Total Bangalore warehouses: {len(blr_warehouses)}")

# Parse cutoff times properly
def parse_cutoff_hour(cutoff):
    try:
        if pd.isna(cutoff):
            return None
        s = str(cutoff)
        if ':' in s:
            return int(s.split(':')[0])
        return None
    except:
        return None

blr_warehouses['cutoff_hour'] = blr_warehouses['pickup_cutoff'].apply(parse_cutoff_hour)

early = blr_warehouses[blr_warehouses['cutoff_hour'] <= 11]
mid = blr_warehouses[(blr_warehouses['cutoff_hour'] > 11) & (blr_warehouses['cutoff_hour'] <= 14)]
late = blr_warehouses[blr_warehouses['cutoff_hour'] > 14]
no_cutoff = blr_warehouses[blr_warehouses['cutoff_hour'].isna()]

print(f"\n  Warehouses with cutoff <= 11 AM:        {len(early):>4} (need major extension)")
print(f"  Warehouses with cutoff 11AM - 2PM:      {len(mid):>4} (moderate extension)")
print(f"  Warehouses with cutoff > 2PM:           {len(late):>4} (minimal/no change)")
print(f"  Warehouses with no cutoff set:          {len(no_cutoff):>4}")

print(f"\n  Impact: {len(early) + len(mid)} warehouses need cutoff updates for 5PM SDD")

# Network capacity requirements
print("\n--- NETWORK CAPACITY REQUIREMENTS ---")
print(f"""
  Mother Hub:
    - Current capacity window: 2PM - 3PM (1 hour)
    - Proposed capacity window: 6PM - 6:30PM (30 min, needs 2x throughput)
    - Recommendation: Add parallel sorting lines or pre-sort at warehouse

  First Mile Vehicles:
    - Current: 1 run (morning pickup)
    - Proposed: 3 staggered runs (12PM, 2PM, 4PM pickup + 5PM final sweep)
    - Additional vehicles needed: ~2x current fleet

  Last Mile:
    - Current evening delivery: 5PM - 8PM  
    - Proposed window: 7:30PM - 9PM
    - Rider availability: Need to ensure evening shift coverage
    - {len(lm_hubs)} LM hubs + {len(quick_hubs)} Quick hubs = {len(lm_hubs) + len(quick_hubs)} delivery points
""")

# Distance matrix summary
print("--- DISTANCE MATRIX SUMMARY ---")
print(f"\n  Mother Hub → LM Hubs:")
print(f"    Min distance: {lm_hubs['dist_from_mh'].min():.1f} km ({lm_hubs['transit_time_min'].min():.0f} min)")
print(f"    Max distance: {lm_hubs['dist_from_mh'].max():.1f} km ({lm_hubs['transit_time_min'].max():.0f} min)")
print(f"    Avg distance: {lm_hubs['dist_from_mh'].mean():.1f} km ({lm_hubs['transit_time_min'].mean():.0f} min)")

print(f"\n  Warehouses → Mother Hub:")
print(f"    Min distance: {blr_warehouses['dist_to_mh'].min():.1f} km ({blr_warehouses['transit_to_mh_min'].min():.0f} min)")
print(f"    Max distance: {blr_warehouses['dist_to_mh'].max():.1f} km ({blr_warehouses['transit_to_mh_min'].max():.0f} min)")
print(f"    Avg distance: {blr_warehouses['dist_to_mh'].mean():.1f} km ({blr_warehouses['transit_to_mh_min'].mean():.0f} min)")

# Feasibility verdict
max_transit = blr_warehouses['transit_to_mh_min'].max()
avg_transit = blr_warehouses['transit_to_mh_min'].mean()

print(f"\n{'='*70}")
print("FEASIBILITY VERDICT")
print(f"{'='*70}")
print(f"""
  With 5PM warehouse pickup cutoff:
  ✓ Average warehouse-to-MH transit: {avg_transit:.0f} minutes 
  ✓ MH sorting + dispatch: 30 minutes
  ✓ MH-to-LM hub transit (avg): {lm_hubs['transit_time_min'].mean():.0f} minutes
  ✓ Last mile delivery: 90 minutes
  
  Total estimated pipeline: {avg_transit + 30 + lm_hubs['transit_time_min'].mean() + 90:.0f} minutes
  5PM + {(avg_transit + 30 + lm_hubs['transit_time_min'].mean() + 90)/60:.1f} hours = ~{17 + (avg_transit + 30 + lm_hubs['transit_time_min'].mean() + 90)/60:.0f}:00 delivery

  VERDICT: {'FEASIBLE - Delivery by 9PM' if (avg_transit + 30 + lm_hubs['transit_time_min'].mean() + 90) <= 240 else 'TIGHT - Needs optimization for outlier warehouses'}
  
  CRITICAL PATH RISKS:
  - Far-flung warehouses (Hoskote, Jigni) may need 5PM → direct-to-hub routing
  - Evening traffic on ORR can add 20-30 min to transit
  - Mother hub needs rapid sort capability (30 min window is tight)
  
  RECOMMENDATIONS:
  1. Stagger first-mile pickups (12PM/2PM/4PM) to smooth MH inflow
  2. Pre-sort at warehouse level for high-volume sellers
  3. Deploy direct WH→LM routes for nearby warehouse-hub pairs
  4. Add buffer: set operational cutoff at 4:45PM for safety margin
""")

print(f"\n✅ Analysis complete! Open the map at:\n   {map_path}")
