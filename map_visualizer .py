#!/usr/bin/env python3
import json
import os
import time
import folium
from folium import plugins
import requests
import html

class LightweightBuildingMap:
    def __init__(self, results_folder="results/neighborhoodD"):
        self.results_folder = results_folder
        self.master_results_file = os.path.join(results_folder, "all_buildings_to_135310103_results.json")

    def load_master_results(self):
        """Load the master results file to get building locations and basic info"""
        with open(self.master_results_file, 'r', encoding='utf-8') as f:
            return json.load(f)

    def get_building_centroid(self, building_data):
        """
        Prefer querying Overpass (center of OSM way).
        Fallback: first coordinate of first local path.
        """
        bid = building_data.get('building_info', {}).get('building_id')
        try:
            osm_way_id = int(bid)
        except (TypeError, ValueError):
            osm_way_id = None

        def fallback_from_paths():
            paths = building_data.get('paths') or []
            if paths and paths[0].get('path_coordinates'):
                first = paths[0]['path_coordinates'][0]
                return [first['lat'], first['lon']]
            return None

        if not osm_way_id:
            return fallback_from_paths()

        query = f"""
        [out:json][timeout:60];
        way({osm_way_id});
        out center tags;
        """

        mirrors = [
            "https://overpass-api.de/api/interpreter",
            "https://overpass.kumi.systems/api/interpreter",
            "https://overpass.openstreetmap.ru/api/interpreter",
        ]
        headers = {"User-Agent": "TalWalks/1.0 (contact: example@example.com)"}

        for attempt in range(3):
            for url in mirrors:
                try:
                    resp = requests.get(url, params={"data": query}, headers=headers, timeout=90)
                    if not resp.ok or "application/json" not in resp.headers.get("Content-Type", ""):
                        continue
                    data = resp.json()
                    for el in data.get("elements", []):
                        if el.get("type") == "way" and el.get("id") == osm_way_id:
                            if "center" in el:
                                return [el["center"]["lat"], el["center"]["lon"]]
                            if "lat" in el and "lon" in el:
                                return [el["lat"], el["lon"]]
                except requests.RequestException:
                    pass
            time.sleep(1.5 * (attempt + 1))

        return fallback_from_paths()

    def create_lightweight_map(self, output_file="lightweight_building_map.html"):
        """Create a map with markers and a refined sidebar UI for viewing paths"""
        master_data = self.load_master_results()

        # Center on landmark area
        center_lat = 31.26171
        center_lon = 34.80048
        map_obj = folium.Map(
            location=[center_lat, center_lon],
            zoom_start=15,
            tiles='OpenStreetMap'
        )

        # Add building markers (NO popups). Store id + name in tooltip data-*.
        buildings_added = 0
        len_data = len(master_data) - 1
        for building_id, building_data in master_data.items():
            if building_id == "_target":
                continue

            centroid = self.get_building_centroid(building_data)
            if not centroid:
                continue

            target_tags = building_data.get('target_tags', {})
            building_name = target_tags.get('name', target_tags.get('name:en', f'Building {building_id}'))
            safe_name_attr = html.escape(building_name or "", quote=True)

            tooltip_html = f'<span data-bid="{building_id}" data-bname="{safe_name_attr}">{safe_name_attr}</span>'
            folium.Marker(
                location=centroid,
                tooltip=folium.Tooltip(tooltip_html, sticky=True),
                icon=folium.Icon(color='blue', icon='home', prefix='fa')
            ).add_to(map_obj)

            buildings_added += 1
            print(f"Added building {building_id} to map, added {buildings_added} buildings so far from {len_data} buildings.")
            # if buildings_added >= 1:
            #     break
        target_data = master_data.get("_target")
        if target_data and "coordinates" in target_data:
            tlat = target_data["coordinates"]["lat"]
            tlon = target_data["coordinates"]["lon"]
            tname = target_data.get("id", "Target")
            folium.Marker(
                location=[tlat, tlon],
                tooltip=f"Target: {tname}",
                icon=folium.Icon(color='red', icon='star', prefix='fa')
            ).add_to(map_obj)

        # Sidebar (legend) container with refined styling
        legend_sidebar_html = f'''
        <style>
          .tw-card {{ background:#fff; border:1px solid #e6e6e6; border-radius:12px; box-shadow:0 6px 24px rgba(0,0,0,0.08); }}
          .tw-btn {{ border:none; border-radius:10px; padding:8px 12px; cursor:pointer; font-weight:600; }}
          .tw-btn-ghost {{ background:#f6f7fb; color:#1f2a44; }}
          .tw-btn-primary {{ background:#2c5aa0; color:#fff; }}
          .tw-badge {{ display:inline-flex; align-items:center; gap:6px; background:#f6f7fb; border:1px solid #e9eaf2; border-radius:999px; padding:3px 10px; font-size:12px; color:#344055; }}
          .tw-row {{ display:flex; gap:10px; align-items:center; justify-content:space-between; }}
          .tw-muted {{ color:#667086; }}
          .tw-small {{ font-size:12px; }}
          .tw-metric {{ font-size:12px; color:#3b455a; }}
          .tw-h1 {{ margin:0; font-size:18px; color:#1f2a44; }}
          .tw-h2 {{ margin:0; font-size:14px; color:#1f2a44; font-weight:700; }}
          .tw-pathcard {{ margin:10px 0; padding:10px; border-left:5px solid var(--clr); background:#fafbff; border-radius:10px; }}
          .tw-pathline {{ width:16px; height:4px; border-radius:2px; display:inline-block; background:var(--clr); }}
          .tw-checkbox {{ transform: translateY(1px); }}
          .tw-hr {{ border:0; border-top:1px solid #edf0f6; margin:10px 0; }}
          .tw-legend-chip {{ display:inline-flex; align-items:center; gap:6px; margin:4px 8px 0 0; }}
        </style>

        <div id="paths_legend" class="tw-card" style="
          position: fixed; top: 10px; right: 10px; width: 380px; max-height: 76vh;
          overflow:auto; z-index: 9999; padding: 14px;">
          <div class="tw-row">
            <h4 class="tw-h1">Building Paths</h4>
            <div style="display:flex; gap:8px;">
              <button class="tw-btn tw-btn-ghost" onclick="clearSelectedPath()" title="Clear drawn">Clear</button>
            </div>
          </div>

          <div class="tw-small tw-muted" style="margin-top:6px;">Total buildings indexed: <b>{buildings_added}</b></div>

          <div id="legend_header" style="margin-top:10px;">
            <span class="tw-muted">Click a <b>building</b> to load its paths.</span>
          </div>

          <div class="tw-hr"></div>

          <div id="legend_body">
            <!-- paths list goes here -->
          </div>

          <div class="tw-hr"></div>

          <div>
            <div class="tw-h2">Legend</div>
            <div style="display:flex; flex-wrap:wrap; margin-top:8px;">
              <span class="tw-legend-chip"><span class="tw-pathline" style="background:#E31837;"></span> Shortest</span>
              <span class="tw-legend-chip"><span class="tw-pathline" style="background:#002B5C;"></span> Few lights</span>
              <span class="tw-legend-chip"><span class="tw-pathline" style="background:#4CAF50;"></span> Safest</span>
              <span class="tw-legend-chip"><span class="tw-pathline" style="background:#FFD700;"></span> Fastest</span>
            </div>
          </div>

          <div class="tw-small tw-muted" style="margin-top:10px;">
            Tip: Use checkboxes to show/hide multiple paths at once; or “Show All”.
          </div>
        </div>
        '''
        map_obj.get_root().html.add_child(folium.Element(legend_sidebar_html))

        # Inject JavaScript
        map_var = map_obj.get_name()
        js_code = f"""
        <script>
// Color + dash patterns for clarity
const PATH_STYLES = {{
  'shortest path': {{ color: '#E31837', dashArray: '0', weight: 5 }},
  'few traffic lights': {{ color: '#002B5C', dashArray: '6 6', weight: 5 }},
  'safest path': {{ color: '#4CAF50', dashArray: '10 6', weight: 5 }},
  'fastest path': {{ color: '#FFD700', dashArray: '2 8', weight: 6 }}
}};

let currentPathLayer = null;                      // last "single-show" layer (for Show on Map)
let currentBuildingLayers = {{}};                 // per-building: array of polylines
let lastLoadedBuildingId = null;                  // remember which building is in the panel

function getLeafletMap() {{
  return window['{map_var}'];
}}

function getDataAttr(layer, attr) {{
  const tt = layer.getTooltip && layer.getTooltip();
  if (!tt) return null;
  const html = tt.getContent && tt.getContent();
  if (!html || typeof html !== 'string') return null;
  const re = new RegExp(attr + '="([^"]+)"');
  const m = html.match(re);
  return m ? m[1] : null;
}}

function wireMarkerClicks() {{
  const map = getLeafletMap();
  if (!map) return;
  map.eachLayer(function(layer) {{
    if (layer && typeof layer.on === 'function' && layer.getTooltip && layer.getTooltip()) {{
      const bid = getDataAttr(layer, 'data-bid');
      const bname = getDataAttr(layer, 'data-bname') || bid;
      if (!bid) return;
      if (layer.__wiredPathsClick) return;
      layer.__wiredPathsClick = true;

      layer.on('click', function() {{
        loadBuildingPathsLegend(bid, bname);
      }});
    }}
  }});
}}

function clearSelectedPath() {{
  const map = getLeafletMap();
  if (currentPathLayer && map) {{
    map.removeLayer(currentPathLayer);
    currentPathLayer = null;
  }}
}}

// Remove all existing per-building layers from map (for previous building)
function clearCurrentBuildingLayers() {{
  const map = getLeafletMap();
  if (!map) return;
  if (!lastLoadedBuildingId) return;
  const arr = currentBuildingLayers[lastLoadedBuildingId] || [];
  arr.forEach(pl => map.removeLayer(pl));
  currentBuildingLayers[lastLoadedBuildingId] = [];
}}

async function loadBuildingPathsLegend(buildingId, buildingName) {{
  const header = document.getElementById('legend_header');
  const body = document.getElementById('legend_body');

  // Clear previous building layers
  clearCurrentBuildingLayers();

  lastLoadedBuildingId = buildingId;
  header.innerHTML = `
    <div class="tw-row" style="align-items:flex-start;">
      <div style="min-width:0;">
        <div class="tw-h2" style="font-size:16px;">${{buildingName || buildingId}}</div>
      </div>
      <div style="display:flex; gap:8px; flex-wrap:wrap;">
        <button class="tw-btn tw-btn-ghost" onclick="toggleAllPaths(true)">Show All</button>
        <button class="tw-btn tw-btn-ghost" onclick="toggleAllPaths(false)">Hide All</button>
        <button class="tw-btn tw-btn-primary" onclick="fitAllVisible()">Fit</button>
      </div>
    </div>
  `;

  body.innerHTML = '<p class="tw-muted">Loading paths…</p>';

  try {{
    const url = `results/neighborhoodD/building_${{buildingId}}_to_135310103_results.json`;
    const resp = await fetch(url);
    if (!resp.ok) throw new Error('Failed to load building data');
    const data = await resp.json();

    const info = data[buildingId];
    const paths = (info && info.paths) ? info.paths : [];

    if (!paths.length) {{
      body.innerHTML = '<p style="color:#aa0000;">No paths found for this building.</p>';
      return;
    }}

    // Build UI + prepare layer objects (but do not add to map yet)
    const map = getLeafletMap();
    currentBuildingLayers[buildingId] = [];
    let html = '';

    paths.forEach((p, idx) => {{
      const style = PATH_STYLES[p.algorithm] || {{ color:'#666', dashArray:'0', weight: 5 }};
      const dist = Math.round(p.distance_meters || 0);
      const mins = ((p.time_seconds || 0) / 60).toFixed(2); // two decimals
      const coords = (p.path_coordinates || []).map(c => [c.lat, c.lon]);

      // Create polyline layer but don't add yet
      const pl = L.polyline(coords, {{
        color: style.color,
        weight: style.weight,
        opacity: 0.95,
        dashArray: style.dashArray
      }});
      pl.__visible = false;
      currentBuildingLayers[buildingId].push(pl);

      html += `
        <div class="tw-pathcard" style="--clr: ${{style.color}}">
          <div class="tw-row">
            <div style="min-width:0;">
              <div class="tw-h2" style="color:${{style.color}};">${{(p.algorithm || 'PATH').toUpperCase()}}</div>
              <div class="tw-metric">Distance: <b>${{dist}}</b> m &nbsp;·&nbsp; Time: <b>${{mins}}</b> min</div>
            </div>
            <div style="display:flex; align-items:center; gap:8px;">
              <label class="tw-small tw-muted" title="Toggle visibility">
                <input class="tw-checkbox" type="checkbox" data-path-index="${{idx}}" onchange="togglePath('${{buildingId}}', ${{idx}}, this.checked)"> show
              </label>
              <button class="tw-btn tw-btn-ghost" onclick="showOnlyPath('${{buildingId}}', ${{idx}})">Only</button>
            </div>
          </div>
        </div>
      `;
    }});

    body.innerHTML = html;
  }} catch (e) {{
    body.innerHTML = '<p style="color:red;">Error: ' + e.message + '</p>';
  }}
}}

// Toggle a single path’s visibility
function togglePath(buildingId, index, visible) {{
  const map = getLeafletMap();
  const layers = (currentBuildingLayers[buildingId] || []);
  const pl = layers[index];
  if (!pl || !map) return;

  if (visible) {{
    if (!pl.__visible) {{
      pl.addTo(map);
      pl.__visible = true;
    }}
  }} else {{
    if (pl.__visible) {{
      map.removeLayer(pl);
      pl.__visible = false;
    }}
  }}
}}

// Show only one path (removes others), and fit to it
function showOnlyPath(buildingId, index) {{
  const map = getLeafletMap();
  if (!map) return;

  // Hide all checkboxes first
  const body = document.getElementById('legend_body');
  body.querySelectorAll('input[type="checkbox"][data-path-index]').forEach(cb => cb.checked = false);

  // Hide all lines
  (currentBuildingLayers[buildingId] || []).forEach(pl => {{
    if (pl.__visible) map.removeLayer(pl);
    pl.__visible = false;
  }});

  // Show selected
  const layers = (currentBuildingLayers[buildingId] || []);
  const pl = layers[index];
  if (!pl) return;
  pl.addTo(map);
  pl.__visible = true;

  // Check its checkbox
  const cb = body.querySelector('input[type="checkbox"][data-path-index="' + index + '"]');
  if (cb) cb.checked = true;

  fitAllVisible();
}}

// Show/Hide all paths for current building
function toggleAllPaths(visible) {{
  const map = getLeafletMap();
  if (!map || !lastLoadedBuildingId) return;
  const layers = (currentBuildingLayers[lastLoadedBuildingId] || []);
  const body = document.getElementById('legend_body');

  layers.forEach((pl, i) => {{
    const cb = body.querySelector('input[type="checkbox"][data-path-index="' + i + '"]');
    if (cb) cb.checked = visible;
    if (visible) {{
      if (!pl.__visible) {{
        pl.addTo(map); pl.__visible = true;
      }}
    }} else {{
      if (pl.__visible) {{
        map.removeLayer(pl); pl.__visible = false;
      }}
    }}
  }});

  if (visible) fitAllVisible();
}}

// Fit map to all currently visible paths (for the loaded building)
function fitAllVisible() {{
  const map = getLeafletMap();
  if (!map || !lastLoadedBuildingId) return;
  const layers = (currentBuildingLayers[lastLoadedBuildingId] || []);
  const visible = layers.filter(pl => pl.__visible);
  if (!visible.length) return;
  const group = L.featureGroup(visible);
  map.fitBounds(group.getBounds(), {{ padding: [30, 30] }});
}}

// Legacy single "Show on Map" (kept for API completeness; unused in new UI)
function showPathOnMap(buildingId, pathIndex) {{
  const map = getLeafletMap();
  if (!map) return;
  if (currentPathLayer) {{
    map.removeLayer(currentPathLayer);
    currentPathLayer = null;
  }}
  fetch(`results/neighborhoodD/building_${{buildingId}}_to_135310103_results.json`)
    .then(r => r.json())
    .then(data => {{
      const info = data[buildingId];
      const p = (info && info.paths) ? info.paths[pathIndex] : null;
      if (!p || !p.path_coordinates) return;

      const style = PATH_STYLES[p.algorithm] || {{ color:'#666', dashArray:'0', weight:5 }};
      const coords = p.path_coordinates.map(c => [c.lat, c.lon]);
      currentPathLayer = L.polyline(coords, {{ color: style.color, weight: style.weight, opacity: 0.95, dashArray: style.dashArray }});
      currentPathLayer.addTo(map);
      map.fitBounds(currentPathLayer.getBounds());
    }})
    .catch(e => alert('Error loading path: ' + e.message));
}}

// Initialize after Folium map is ready
(function init() {{
  const tryWire = () => {{
    const map = getLeafletMap();
    if (!map) {{ setTimeout(tryWire, 150); return; }}
    wireMarkerClicks();
  }};
  tryWire();
}})();
        </script>
        """
        map_obj.get_root().html.add_child(folium.Element(js_code))

        # Save the map
        map_obj.save(output_file)
        print(f"Lightweight building map created: {output_file}")
        print(f"Added {buildings_added} building markers")
        print("Serve via HTTP (e.g., `python -m http.server 8000`) so fetch() can read /results/*.json")
        return map_obj

if __name__ == "__main__":
    visualizer = LightweightBuildingMap()
    visualizer.create_lightweight_map("neighborhoodD.html")
