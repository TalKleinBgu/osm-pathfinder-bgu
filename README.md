# OSM Pathfinding (Ben-Gurion University) – Bidirectional A*

This project computes pedestrian routes around Ben-Gurion University using OpenStreetMap (OSM) data and a front-to-front Bidirectional A* search. It generates per-building routes to a chosen target and exports them for interactive Folium maps.

## What’s included

- Bidirectional A* (forward from building, backward from target) with a reverse graph
- Four profiles: shortest, few traffic lights, safest, fastest
- JSON results per building + a master file (used by the map visualizers)
- Two visualizers: a multi-building map and a lightweight WIP sidebar map

## Quick start (Windows PowerShell)

1) Install deps

```powershell
pip install -r requirements.txt
```

2) Run the pathfinder (uses `map.osm` in the repo and writes results under `results/`)

```powershell
python osm_pathfinder.py
```

By default the script runs `main_multi()` for an example area and target (OSM id `135310103`) and saves files like:

```
results/
   neighborhoodB/
      building_<id>_to_135310103_results.json
      all_buildings_to_135310103_results.json   <-- master file
```

3) Visualize (multi-building)

Edit `multi_building_visualizer.py` if needed to point `results_file` to your master file, then run:

```powershell
python multi_building_visualizer.py
```

This creates `multi_building_pathfinding_map.html` with layers you can toggle per building and algorithm.

## Profiles (costs and heuristics)

1) Shortest distance
- Cost: meters on each edge
- Heuristic: straight-line distance (haversine)
- Search: Bidirectional A*

2) Few traffic lights
- Cost: distance scaled by a factor + node penalties when entering intersections
   - distance_factor = 0.35
   - traffic signal: +220 m
   - crossing zebra: +90 m, uncontrolled: +160 m, island: +70 m, other: +120 m
   - tactile paving: −15 m, lowered kerb: −10 m (reduce penalty)
- Heuristic: 0.95 × distance_factor × straight-line distance (admissible)

3) Safest
- Cost: distance + safety penalties (lighting, sidewalks, road class, crossings, surface, width, tunnels/bridges). Public places provide a small bonus.
- Important: final cost is clamped to ≥ distance to keep the distance heuristic admissible.

4) Fastest (time)
- Cost: seconds with walking speed by facility, surface slowdowns, steps multiplier, and waiting at signals/crossings
- Heuristic: straight-line distance / 1.6 m/s (upper-bound speed)

## Bidirectional A* details

- Forward g-values from start; backward g-values from target on the reverse graph
- Heuristics: hF(n)=dist(n, goal), hB(n)=dist(n, start) (scaled appropriately per profile)
- Upper bound (UB) tightened from settled meet nodes and from cross-frontier edge relaxations
- Termination: stop when min_f_forward + min_f_backward ≥ UB
- Reconstruction: via stored meet node/edge and the forward/backward predecessor chains

This setup returns optimal paths per profile given the admissible heuristics above. Distances shown in the UI are summed from the graph edges on the returned path.

## Repository layout

```
map.osm                      # Input OSM XML
osm_pathfinder.py            # Parser, graph builder, cost profiles, Bidirectional A*
multi_building_visualizer.py # Folium map for many buildings (recommended)
lightweight_building_map.py  # WIP sidebar visualizer
requirements.txt             # Python deps
results/                     # Output JSONs (per-building + master)
README.md
```

## Usage tips

- Target: `osm_pathfinder.py` defaults to target id `135310103`. Use `set_target()` to change (supports nodes or building ways).
- Areas: `main_multi()` shows how to route for multiple bounding boxes and write to a subfolder (e.g., `results/neighborhoodB`).
- Visual colors: the visualizer normalizes algorithm names so styles apply even if labels differ slightly.

## Troubleshooting

- “Shortest” distance looks longer than another profile
   - The search is bidirectional A* with an admissible distance heuristic. If another profile appears shorter, it’s usually a UI/label mix-up or a stats mismatch. The code sums distances directly from the graph edges for the returned path to avoid drift.
- No paths for a building
   - The nearest “walkable” node may be far or disconnected. Check OSM tags (foot access) or expand the area.
- Performance
   - For larger maps, consider indexing nodes for snapping (grid/KD-tree). Current O(N) scan is fine for the included areas.

## License

Open source. OSM data © OpenStreetMap contributors, ODbL.

## References

- OpenStreetMap Wiki – Routing
- A* Search Algorithm
- OSM Pedestrian routing and highway tags
