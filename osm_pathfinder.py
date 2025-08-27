#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
OSM Pathfinding (Pedestrian) for BGU area
- Proper Front-to-Front Bidirectional A* (with reverse graph)
- Gate snapping to nearest walkable node
- Four cost profiles: shortest / few_traffic_lights / safest(day|night) / fastest
- Traffic signals & crossings counted on NODEs (OSM convention)
Outputs JSON compatible with your folium visualizer.
"""

import xml.etree.ElementTree as ET
import math, heapq, json, time
from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional
import os
# --------------------------- Data classes ---------------------------

@dataclass
class Node:
    id: str
    lat: float
    lon: float
    tags: Dict[str, str]

@dataclass
class Way:
    id: str
    nodes: List[str]
    tags: Dict[str, str]

@dataclass
class PathResult:
    path: List[str]
    total_cost: float
    distance_meters: float
    num_traffic_signals: int
    num_crossings: int
    safety_penalties: float
    time_seconds: float
    description: str

# --------------------------- Pathfinder -----------------------------

class OSMPathfinder:
    def __init__(self, osm_file: str):
        self.nodes: Dict[str, Node] = {}
        self.ways: Dict[str, Way] = {}
        self.buildings: Dict[str, Way] = {}
        self.graph = defaultdict(list)   # forward graph: u -> [{to_node, distance, way_id, way_tags}]
        self.rgraph = defaultdict(list)  # reverse graph: u -> [{to_node, distance, way_id, way_tags}]
        # Default target is the Ben Gurion gate node (for backward compatibility)
        self.ben_gurion_gate_id = "135310103"  # שער העלייה (raw OSM node id)
        self.target_id = self.ben_gurion_gate_id

        self._parse_osm(osm_file)
        self._build_graph()

        # Index public places for safety bonuses (playgrounds, sports, worship, schools, etc.)
        self._index_public_places()

        # Tunable weights for profiles
        self._few_traffic_lights_cfg = {
            'distance_factor': 0.35,      # lower means we care less about distance vs signals
            'signal_penalty_m': 220.0,    # base equivalent meters for traffic light
            'zebra_penalty_m': 90.0,
            'uncontrolled_penalty_m': 160.0,
            'island_penalty_m': 70.0,
            'other_crossing_penalty_m': 120.0,
            'tactile_bonus_m': -15.0,     # subtract if tactile_paving=yes
            'lowered_kerb_bonus_m': -10.0 # subtract if kerb=lowered
        }

    # -------------------- Parse & Graph build --------------------

    def _parse_osm(self, osm_file: str):
        print("Parsing OSM file…")
        root = ET.parse(osm_file).getroot()

        for n in root.findall('node'):
            nid = n.get('id')
            lat = float(n.get('lat')); lon = float(n.get('lon'))
            tags = {t.get('k'): t.get('v') for t in n.findall('tag')}
            self.nodes[nid] = Node(nid, lat, lon, tags)

        for w in root.findall('way'):
            wid = w.get('id')
            nrefs = [nd.get('ref') for nd in w.findall('nd')]
            tags = {t.get('k'): t.get('v') for t in w.findall('tag')}

            if 'building' in tags:
                self.buildings[wid] = Way(wid, nrefs, tags)

            if self._is_walkable(tags):
                self.ways[wid] = Way(wid, nrefs, tags)

        print(f"Loaded {len(self.nodes)} nodes, {len(self.ways)} walkable ways, {len(self.buildings)} buildings.")

    def _is_walkable(self, tags: Dict[str, str]) -> bool:
        highway = tags.get('highway', '')
        if highway in {'motorway', 'motorway_link', 'trunk', 'trunk_link'}:
            return False
        foot = tags.get('foot', '')
        if foot == 'no':
            return False
        if foot == 'yes':
            return True
        return highway in {
            'footway','pedestrian','path','steps','sidewalk',
            'primary','secondary','tertiary','residential',
            'living_street','service','unclassified','track'
        }

    def _build_graph(self):
        print("Building graphs (forward & reverse)…")
        for way in self.ways.values():
            tags = way.tags
            # treat one-way for pedestrians only if explicitly tagged
            oneway_car = tags.get('oneway') == 'yes'
            oneway_foot = tags.get('oneway:foot') == 'yes'
            is_steps = tags.get('highway') == 'steps'
            # steps are bidirectional for walking unless explicitly tagged one-way:foot
            is_oneway_for_ped = oneway_foot  # ignore car oneway for pedestrians

            for i in range(len(way.nodes) - 1):
                a, b = way.nodes[i], way.nodes[i+1]
                if a not in self.nodes or b not in self.nodes:
                    continue
                dist = self._haversine(self.nodes[a], self.nodes[b])

                e_ab = {'to_node': b, 'distance': dist, 'way_id': way.id, 'way_tags': tags}
                e_ba = {'to_node': a, 'distance': dist, 'way_id': way.id, 'way_tags': tags}

                # forward
                self.graph[a].append(e_ab)
                # reverse
                self.rgraph[b].append({'to_node': a, 'distance': dist, 'way_id': way.id, 'way_tags': tags})

                # add opposite direction if not one-way for pedestrians
                if not is_oneway_for_ped:
                    self.graph[b].append(e_ba)
                    self.rgraph[a].append({'to_node': b, 'distance': dist, 'way_id': way.id, 'way_tags': tags})

        print(f"Graph has {len(self.graph)} forward nodes, {len(self.rgraph)} reverse nodes.")

    # Public places index for safety bonuses
    def _index_public_places(self, radius_m: float = 60.0) -> None:
        """Mark nodes in the walking graph that are at or near public places
        such as playgrounds, sports fields, schools, places of worship.
        Stores flags in Node.tags: near_public_place=yes and public_place_kind.
        """
        amenity_whitelist = {
            'school','kindergarten','university','library','community_centre',
            'place_of_worship','synagogue','police','clinic','hospital'
        }
        leisure_whitelist = {
            'playground','pitch','sports_centre','stadium','park','garden'
        }

        # Collect public place nodes
        public_node_ids: List[str] = []
        for nid, node in self.nodes.items():
            a = node.tags.get('amenity','')
            l = node.tags.get('leisure','')
            if (a in amenity_whitelist) or (l in leisure_whitelist):
                public_node_ids.append(nid)

        if not public_node_ids:
            return

        # Mark graph nodes that are at/near public places
        for nid in self.graph.keys():
            n = self.nodes[nid]
            # direct tag on this node counts
            if nid in public_node_ids:
                n.tags['near_public_place'] = 'yes'
                n.tags['public_place_dist'] = '0'
                continue
            # otherwise compute proximity to nearest public node (simple scan; area is small)
            mind = float('inf')
            for pid in public_node_ids:
                d = self._haversine(n, self.nodes[pid])
                if d < mind:
                    mind = d
                    if mind <= radius_m:
                        break
            if mind <= radius_m:
                n.tags['near_public_place'] = 'yes'
                n.tags['public_place_dist'] = f"{mind:.1f}"

    # ---------------------- Geometry & Snap ----------------------

    @staticmethod
    def _haversine(n1: Node, n2: Node) -> float:
        R = 6371000.0
        lat1, lon1 = math.radians(n1.lat), math.radians(n1.lon)
        lat2, lon2 = math.radians(n2.lat), math.radians(n2.lon)
        dlat, dlon = lat2-lat1, lon2-lon1
        a = math.sin(dlat/2)**2 + math.cos(lat1)*math.cos(lat2)*math.sin(dlon/2)**2
        return 2*R*math.asin(math.sqrt(a))

    def _heuristic_distance(self, nid: str, tid: str) -> float:
        return self._haversine(self.nodes[nid], self.nodes[tid])

    def _heuristic_time(self, nid: str, tid: str) -> float:
        return self._heuristic_distance(nid, tid) / 1.6  # m/s (upper-bound speed)

    def find_nearest_walkable_node(self, lat: float, lon: float) -> Optional[str]:
        probe = Node("probe", lat, lon, {})
        best, bestd = None, float('inf')
        for nid in self.graph:  # only nodes present in the walking graph
            d = self._haversine(probe, self.nodes[nid])
            if d < bestd:
                best, bestd = nid, d
        return best

    # ---------------------- Target handling ----------------------

    def set_target(self, target_osm_id: str) -> None:
        """Configure destination target by OSM id. Can be a node (POI) or a building way id."""
        self.target_id = target_osm_id

    def get_buildings_in_area(self, north: float, south: float, west: float, east: float) -> List[str]:
        """Find all buildings within the specified bounding box."""
        buildings_in_area = []
        
        for building_id, building in self.buildings.items():
            # Get building centroid
            pts = [self.nodes[nid] for nid in building.nodes if nid in self.nodes]
            if not pts:
                continue
            
            clat = sum(p.lat for p in pts) / len(pts)
            clon = sum(p.lon for p in pts) / len(pts)
            
            # Check if building is within bounds
            if south <= clat <= north and west <= clon <= east:
                buildings_in_area.append(building_id)
        
        print(f"Found {len(buildings_in_area)} buildings in area ({south:.5f}, {west:.5f}) to ({north:.5f}, {east:.5f})")
        return buildings_in_area

    def target_walkable_node(self) -> Optional[str]:
        """Snap the configured target (node or building way) to nearest walkable graph node."""
        # If target is a node
        if self.target_id in self.nodes:
            g = self.nodes[self.target_id]
            return self.find_nearest_walkable_node(g.lat, g.lon)
        # If target is a building way
        if self.target_id in self.buildings:
            return self.find_nearest_walkable_node_to_building(self.target_id)
        # Fallback to legacy gate if target not found
        g = self.nodes.get(self.ben_gurion_gate_id)
        if not g:
            return None
        return self.find_nearest_walkable_node(g.lat, g.lon)

    def find_nearest_walkable_node_to_building(self, building_id: str) -> Optional[str]:
        if building_id not in self.buildings:
            return None
        pts = [self.nodes[nid] for nid in self.buildings[building_id].nodes if nid in self.nodes]
        if not pts: return None
        clat = sum(p.lat for p in pts) / len(pts)
        clon = sum(p.lon for p in pts) / len(pts)
        return self.find_nearest_walkable_node(clat, clon)

    # --------------------- Cost profiles ------------------------

    def _cost_shortest(self, e, from_id):  # meters
        return e['distance']

    def _cost_few_traffic_lights(self, e, from_id):
        # Distance scaled down so signals dominate, with nuanced crossing penalties
        cfg = self._few_traffic_lights_cfg
        c = e['distance'] * cfg['distance_factor']

        # penalties on the node being entered
        nt = self.nodes[e['to_node']].tags
        crossing = nt.get('crossing')  # e.g., traffic_signals, zebra, uncontrolled, island
        is_signal = (nt.get('highway') == 'traffic_signals') or (crossing == 'traffic_signals')

        if is_signal:
            c += cfg['signal_penalty_m']
        elif nt.get('highway') == 'crossing' or crossing:
            # nuanced crossing types
            if crossing == 'zebra' or nt.get('crossing:markings') == 'zebra':
                c += cfg['zebra_penalty_m']
            elif crossing == 'uncontrolled' or crossing == 'no_signals':
                c += cfg['uncontrolled_penalty_m']
            elif crossing == 'island' or crossing == 'refuge_island':
                c += cfg['island_penalty_m']
            else:
                c += cfg['other_crossing_penalty_m']

            # accessibility features slightly reduce perceived penalty
            if nt.get('tactile_paving') == 'yes':
                c += cfg['tactile_bonus_m']
            if nt.get('kerb') == 'lowered':
                c += cfg['lowered_kerb_bonus_m']

        return c

    def _cost_safest(self, e, from_id, is_night: bool):
        c = e['distance']
        wt = e['way_tags']
        # lighting along way or at node
        way_lit = wt.get('lit')
        if (way_lit == 'no') or (way_lit is None and self.nodes[e['to_node']].tags.get('lit') == 'no'):
            c += 150.0 if is_night else 25.0

        if way_lit == 'yes' or (way_lit is None and self.nodes[e['to_node']].tags.get('lit') == 'yes'):
            c -= 10.0

        # alley/service and road classification when mixing with traffic
        hw = wt.get('highway')
        if hw in {'alley','service'}:
            c += 90.0 if is_night else 25.0
        if hw in {'primary','secondary','tertiary'} and wt.get('sidewalk', '') in {'no','none',''}:
            c += 140.0 if is_night else 50.0

        # sidewalks
        sw = wt.get('sidewalk', '')
        if sw in {'none','no',''} and hw not in {'footway','pedestrian','path'}:
            c += 90.0 if is_night else 25.0

        # shared cycle/foot paths without segregation are a bit less comfortable
        if hw in {'path','footway'} and wt.get('bicycle') in {'yes','designated'} and wt.get('segregated') == 'no':
            c += 30.0 if is_night else 10.0

        # surface: poor surfaces feel less safe or comfortable
        surf = wt.get('surface','')
        if surf in {'gravel','sand','dirt','ground','grass','unpaved'}:
            c += 40.0
        if surf in {'cobblestone','sett'}:
            c += 25.0

        # narrow width
        try:
            width = float(wt.get('width', 'nan'))
            if width and width < 1.5:
                c += 30.0
        except ValueError:
            pass

        # tunnels/bridges at night
        if is_night and (wt.get('tunnel') == 'yes' or wt.get('bridge') == 'yes'):
            c += 50.0

        # node-based crossing safety: uncontrolled crossings are worse; zebra better
        nt = self.nodes[e['to_node']].tags
        if nt.get('highway') == 'crossing' or nt.get('crossing'):
            crossing = nt.get('crossing')
            if crossing == 'uncontrolled' or crossing == 'no_signals':
                c += 60.0
            elif crossing == 'zebra' or nt.get('crossing:markings') == 'zebra':
                c += 10.0  # still a minor risk
            elif crossing == 'island':
                c += 20.0

            # accessibility features reduce perceived risk slightly
            if nt.get('tactile_paving') == 'yes':
                c -= 5.0
            if nt.get('kerb') == 'lowered':
                c -= 5.0

        # public places increase perceived safety; apply a small bonus (reduce penalties)
        # Bonus is larger at night. Keep edge cost non-negative by clamping to base distance.
        bonus = 0.0
        amenity = nt.get('amenity','')
        leisure = nt.get('leisure','')
        if (
            amenity in {'place_of_worship','synagogue','school','kindergarten','university','library','community_centre','police','clinic','hospital','restaurant','cafe'}
            or leisure in {'playground','pitch','sports_centre','stadium','park','garden','fitness_centre','dog_park','swimming_pool'}
            or nt.get('near_public_place') == 'yes'
        ):
            bonus = 40.0 if is_night else 20.0
        if bonus:
            c = max(e['distance'], c - bonus)

        return max(e['distance'], c)

    def _cost_fastest(self, e, from_id):  # seconds
        dist = e['distance']
        wt = e['way_tags']

        # baseline walking speed (m/s) varies by facility
        hw = wt.get('highway')
        if hw in {'footway','pedestrian','sidewalk'}:
            v = 1.45
        elif hw == 'steps':
            v = 0.8
        else:
            v = 1.35

        # surface adjustments
        surf = wt.get('surface','')
        if surf in {'asphalt','concrete','paved'}:
            pass
        elif surf in {'compacted','fine_gravel'}:
            v *= 0.95
        elif surf in {'gravel','sand','dirt','ground','grass','unpaved'}:
            v *= 0.8
        elif surf in {'cobblestone','sett'}:
            v *= 0.9

        # narrow width slows flow slightly
        try:
            width = float(wt.get('width', 'nan'))
            if width and width < 1.5:
                v *= 0.92
        except ValueError:
            pass

        t = dist / max(v, 0.5)

        # steps/elevator handling
        if hw == 'steps':
            t *= 1.5
        # entering elevator nodes/ways incurs wait time; very fast vertical travel
        nt = self.nodes[e['to_node']].tags
        if nt.get('highway') == 'elevator' or wt.get('highway') == 'elevator':
            t += 20.0  # average wait
            # movement within elevator is negligible for distance scales here

        # waiting at crossings/signals when entering node
        crossing = nt.get('crossing')
        if (nt.get('highway') == 'traffic_signals') or (crossing == 'traffic_signals'):
            t += 25.0
        elif nt.get('highway') == 'crossing' or crossing:
            if crossing == 'zebra' or nt.get('crossing:markings') == 'zebra':
                t += 6.0
            elif crossing == 'uncontrolled' or crossing == 'no_signals':
                t += 3.0
            elif crossing == 'island' or crossing == 'refuge_island':
                t += 8.0
            else:
                t += 6.0

        # shared with bicycles and not segregated can slow slightly
        if hw in {'path','footway'} and wt.get('bicycle') in {'yes','designated'} and wt.get('segregated') == 'no':
            t *= 1.03

        return t

    # ---------------- Bidirectional A* (Front-to-Front) ----------------

    def bidir_astar(self, start: str, goal: str, profile: str, is_night=False) -> Optional[PathResult]:
        if start == goal:
            return self._calc_stats([start], profile, is_night)
        if start not in self.graph or goal not in self.graph:
            return None

        # Select edge cost
        def edge_cost(e, u):
            if profile == 'shortest': return self._cost_shortest(e, u)
            if profile == 'few_traffic_lights': return self._cost_few_traffic_lights(e, u)
            if profile == 'safest': return self._cost_safest(e, u, is_night)
            if profile == 'fastest': return self._cost_fastest(e, u)
            return self._cost_shortest(e, u)

        # Heuristics: keep admissible for each profile
        def hF(n):  # forward heuristic to goal
            if profile == 'fastest':
                return self._heuristic_time(n, goal)
            if profile == 'few_traffic_lights':
                return 0.95 * self._few_traffic_lights_cfg['distance_factor'] * self._heuristic_distance(n, goal)
            return self._heuristic_distance(n, goal)

        def hB(n):  # backward heuristic to start
            if profile == 'fastest':
                return self._heuristic_time(n, start)
            if profile == 'few_traffic_lights':
                return 0.95 * self._few_traffic_lights_cfg['distance_factor'] * self._heuristic_distance(n, start)
            return self._heuristic_distance(n, start)

        # State
        gF = {start: 0.0}
        gB = {goal: 0.0}
        predF = {start: None}
        predB = {goal: None}
        openF = [(hF(start), 0.0, start)]  # (f, g, node)
        openB = [(hB(goal), 0.0, goal)]
        closedF = set()
        closedB = set()

        UB = float('inf')
        meet = None  # ('node', m) or ('edge', u, v)

        def top_key(pq):
            # return current min f, skipping stale
            while pq and (pq[0][1] > (gF.get(pq[0][2], float('inf')) if pq is openF else gB.get(pq[0][2], float('inf')))):
                heapq.heappop(pq)
            return pq[0][0] if pq else float('inf')

        def improve_UB_via_node(u):
            nonlocal UB, meet
            if u in gF and u in gB:
                val = gF[u] + gB[u]
                if val < UB:
                    UB = val
                    meet = ('node', u)

        def relax_dir(u, g_u, graph_dir, g_here, g_other, pred_here, h_here, open_here, is_forward: bool):
            nonlocal UB, meet
            for e in graph_dir.get(u, []):
                v = e['to_node']
                c = edge_cost(e, u)
                if c < 0:
                    continue  # safety
                tentative = g_u + c
                if tentative < g_here.get(v, float('inf')):
                    g_here[v] = tentative
                    pred_here[v] = u
                    f = tentative + h_here(v)
                    heapq.heappush(open_here, (f, tentative, v))
                # Try to tighten UB using v if the other search has reached v
                if v in g_other:
                    cand = tentative + g_other[v]
                    if cand < UB:
                        UB = cand
                        meet = ('edge', u, v) if is_forward else ('edge', v, u)  # store as (u->v) in forward direction

        # Main loop
        while openF or openB:
            # Expand side with smaller current f
            if top_key(openF) <= top_key(openB):
                # Forward step
                f_u, g_u, u = heapq.heappop(openF)
                if u in closedF or g_u != gF.get(u, float('inf')):
                    continue
                closedF.add(u)
                improve_UB_via_node(u)
                relax_dir(u, g_u, self.graph, gF, gB, predF, hF, openF, True)
            else:
                # Backward step
                f_u, g_u, u = heapq.heappop(openB)
                if u in closedB or g_u != gB.get(u, float('inf')):
                    continue
                closedB.add(u)
                improve_UB_via_node(u)
                relax_dir(u, g_u, self.rgraph, gB, gF, predB, hB, openB, False)

            # Termination: when best possible connection cannot beat current UB
            if (top_key(openF) + top_key(openB)) >= UB:
                break

        if not meet or not math.isfinite(UB):
            return None

        # Reconstruct forward path
        def build_forward_path_to(u):
            seq = [u]
            while predF[seq[-1]] is not None:
                seq.append(predF[seq[-1]])
            seq.reverse()
            return seq

        def build_forward_path_from_using_B(v):
            # Use predB chain: next forward node after x is predB[x]
            seq = [v]
            cur = v
            while cur in predB and predB[cur] is not None:
                nxt = predB[cur]
                seq.append(nxt)
                cur = nxt
            return seq

        if meet[0] == 'node':
            m = meet[1]
            path = build_forward_path_to(m)
            tail = build_forward_path_from_using_B(m)
            path += tail[1:]  # skip duplicate m
        else:
            u, v = meet[1], meet[2]  # forward edge u->v
            head = build_forward_path_to(u)
            mid_tail = build_forward_path_from_using_B(v)  # starts with v
            path = head + mid_tail  # includes u->v

        return self._calc_stats(path, profile, is_night)

    # -------------------- Stats & JSON export --------------------

    def _calc_stats(self, path: List[str], profile: str, is_night: bool) -> PathResult:
        dist = 0.0; sig = 0; cross = 0; safepen = 0.0; cost = 0.0; tsec = 0.0
        for i in range(len(path)-1):
            u, v = path[i], path[i+1]
            # find matching edge in forward graph
            edge = next((e for e in self.graph[u] if e['to_node'] == v), None)
            if not edge:  # fallback (shouldn't happen)
                continue
            dist += edge['distance']
            # costs
            if profile == 'shortest':
                c = self._cost_shortest(edge, u)
            elif profile == 'few_traffic_lights':
                c = self._cost_few_traffic_lights(edge, u)
            elif profile == 'safest':
                c = self._cost_safest(edge, u, is_night)
            elif profile == 'fastest':
                c = self._cost_fastest(edge, u)
            else:
                c = self._cost_shortest(edge, u)
            cost += c
            safepen += self._cost_safest(edge, u, is_night) - edge['distance']
            tsec += self._cost_fastest(edge, u)

            # node-based counters on v
            nt = self.nodes[v].tags
            if nt.get('highway') == 'traffic_signals' or nt.get('crossing') == 'traffic_signals':
                sig += 1
            elif nt.get('highway') == 'crossing' or ('crossing' in nt):
                cross += 1

        descr = {
            'shortest': 'shortest path',
            'few_traffic_lights': 'few_traffic_lights path',
            'safest': f"safest path",
            'fastest': 'fastest path',
        }[profile]

        return PathResult(
            path=path, total_cost=cost, distance_meters=dist,
            num_traffic_signals=sig, num_crossings=cross,
            safety_penalties=safepen, time_seconds=tsec,
            description=descr
        )

    # -------------------- Public API for your run --------------------

    def find_paths_from_building(self, building_id: str) -> Dict[str, List[PathResult]]:
        goal_walk = self.target_walkable_node()
        if not goal_walk:
            print("❌ Could not snap target to a walkable node.")
            return {}
        if building_id not in self.buildings:
            print(f"❌ Building {building_id} not found.")
            return {}

        start = self.find_nearest_walkable_node_to_building(building_id)
        if not start or start not in self.graph:
            print(f"❌ No walkable node near building {building_id}.")
            return {}

        print(f"Start node (walkable): {start}  |  Goal (target-walkable): {goal_walk}")
        print(f"Straight-line: {self._heuristic_distance(start, goal_walk):.1f} m")

        results: List[PathResult] = []
        for prof in ['shortest','few_traffic_lights','safest','fastest']:
            print(f"  → {prof} … ", end="", flush=True)
            r = self.bidir_astar(start, goal_walk, prof, is_night=False)
            if r:
                results.append(r)
                print(f"ok  ({r.distance_meters:.0f} m, {r.time_seconds/60:.1f} min)")
            else:
                print("no path")

        return {building_id: results} if results else {}

    def export_results_to_json(self, results: Dict[str, List[PathResult]], filename: str):
        out = {}
        # add building entry
        for bid, plist in results.items():
            nearest = self.find_nearest_walkable_node_to_building(bid)
            tgt_node = self.nodes[nearest] if nearest else None
            out[bid] = {
                'target_coordinates': {'lat': tgt_node.lat, 'lon': tgt_node.lon} if tgt_node else None,
                'target_tags': self.buildings[bid].tags,
                'building_info': {'building_id': bid, 'nearest_walkable_node': nearest, 'is_building': True},
                'paths': []
            }
            for pr in plist:
                coords = [{'lat': self.nodes[n].lat, 'lon': self.nodes[n].lon, 'id': n} for n in pr.path if n in self.nodes]
                out[bid]['paths'].append({
                    'algorithm': pr.description,
                    'total_cost': pr.total_cost,
                    'distance_meters': pr.distance_meters,
                    'num_traffic_signals': pr.num_traffic_signals,
                    'num_crossings': pr.num_crossings,
                    'safety_penalties': pr.safety_penalties,
                    'time_seconds': pr.time_seconds,
                    'path_coordinates': coords
                })

        # target metadata (node or building)
        snapped_id = self.target_walkable_node()
        target_entry = {'id': self.target_id, 'snapped_walkable_node': snapped_id}
        if self.target_id in self.nodes:
            tnode = self.nodes[self.target_id]
            target_entry.update({
                'type': 'node',
                'coordinates': {'lat': tnode.lat, 'lon': tnode.lon},
                'tags': tnode.tags
            })
        elif self.target_id in self.buildings:
            way = self.buildings[self.target_id]
            pts = [self.nodes[nid] for nid in way.nodes if nid in self.nodes]
            if pts:
                clat = sum(p.lat for p in pts) / len(pts)
                clon = sum(p.lon for p in pts) / len(pts)
                target_entry.update({'coordinates': {'lat': clat, 'lon': clon}})
            target_entry.update({'type': 'building', 'tags': way.tags})
        else:
            # fallback to legacy gate if unknown target id
            g = self.nodes.get(self.ben_gurion_gate_id)
            if g:
                target_entry.update({
                    'type': 'node',
                    'coordinates': {'lat': g.lat, 'lon': g.lon},
                    'tags': g.tags
                })
        out['_target'] = target_entry

        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(out, f, indent=2, ensure_ascii=False)
        print(f"📄 Results exported → {filename}")

    def process_all_buildings_in_area(self, north: float, south: float, west: float, east: float, output_folder: str = "results"):
        """Process all buildings in the specified area and save results to individual JSON files."""
        import os
        
        # Create output folder if it doesn't exist
        os.makedirs(output_folder, exist_ok=True)
        
        # Get all buildings in the area
        buildings = self.get_buildings_in_area(north, south, west, east)
        
        all_results = {}
        successful_buildings = []
        
        for i, building_id in enumerate(buildings, 1):
            print(f"\n[{i}/{len(buildings)}] Processing building {building_id}...")
            
            results = self.find_paths_from_building(building_id)
            if results:
                # Save individual building result
                individual_filename = os.path.join(output_folder, f"building_{building_id}_to_{self.target_id}_results.json")
                self.export_results_to_json(results, individual_filename)
                
                # Add to master results
                all_results.update(results)
                successful_buildings.append(building_id)
            else:
                print(f"❌ No paths found for building {building_id}")
        
        # Save master results file with all buildings
        if all_results:
            master_filename = os.path.join(output_folder, f"all_buildings_to_{self.target_id}_results.json")
            self.export_results_to_json(all_results, master_filename)
            print(f"\n✅ Successfully processed {len(successful_buildings)}/{len(buildings)} buildings")
            print(f"📁 Results saved in '{output_folder}' folder")
            print(f"📄 Master file: {master_filename}")
        else:
            print("\n❌ No successful path computations")
        
        return all_results
    def get_buildings_in_areas(self, boxes: List[Tuple[float, float, float, float]]) -> List[str]:
        """
        Accepts a list of bounding boxes [(north, south, west, east), ...]
        and returns the UNION of building ids within any of them (deduped).
        """
        seen = set()
        results = []
        for (north, south, west, east) in boxes:
            ids = self.get_buildings_in_area(north, south, west, east)
            for bid in ids:
                if bid not in seen:
                    seen.add(bid)
                    results.append(bid)
        print(f"Union across {len(boxes)} boxes → {len(results)} unique buildings")
        return results

    def process_buildings_in_areas(
        self,
        boxes: List[Tuple[float, float, float, float]],
        output_folder: str = "results"
    ):
        """
        Process all buildings inside the UNION of the provided boxes and save
        per-building JSON + a master JSON, identical structure to your single-area version.
        """
        import os
        os.makedirs(output_folder, exist_ok=True)

        # Collect union of buildings
        buildings = self.get_buildings_in_areas(boxes)

        all_results = {}
        successful = []

        for i, building_id in enumerate(buildings, 1):
            print(f"\n[{i}/{len(buildings)}] Processing building {building_id}...")
            results = self.find_paths_from_building(building_id)
            if results:
                # Save individual building result
                individual_filename = os.path.join(
                    output_folder,
                    f"building_{building_id}_to_{self.target_id}_results.json"
                )
                self.export_results_to_json(results, individual_filename)

                # Add to master
                all_results.update(results)
                successful.append(building_id)
            else:
                print(f"❌ No paths found for building {building_id}")

        # Save master results
        if all_results:
            master_filename = os.path.join(
                output_folder,
                f"all_buildings_to_{self.target_id}_results.json"
            )
            self.export_results_to_json(all_results, master_filename)
            print(f"\n✅ Successfully processed {len(successful)}/{len(buildings)} buildings")
            print(f"📁 Results saved in '{output_folder}'")
            print(f"📄 Master file: {master_filename}")
        else:
            print("\n❌ No successful path computations")

        return all_results

# ------------------------------ main -------------------------------

def main():
    osm_file = "map.osm"
    target_id = "135310103"      # University Center (node or way id)
    
    # Landmarks defining the area
    north = 31.25865
    south = 31.25179
    west = 34.78705
    east = 34.79859
    
    pf = OSMPathfinder(osm_file)
    pf.set_target(target_id)
    
    # Process all buildings in the specified area
    all_results = pf.process_all_buildings_in_area(north, south, west, east, "results")
    
    if all_results:
        print(f"\n🎉 Completed processing for {len(all_results)} buildings!")
    else:
        print("\n❌ No buildings processed successfully")

def main_multi():
    osm_file = "map.osm"

    # List of AOIs (north, south, west, east)
    areas = [
        # (31.25865, 31.25179, 34.78705, 34.79859),   # neighborhoodB
        # (31.27371, 31.25850, 34.78765, 34.79808),   # neighborhoodD
        (31.27167, 31.26512, 34.79806, 34.80381),   # neighborhoodOldV
        # add more boxes as needed...
    ]

    # One or many target ids (nodes or building ways)
    target_ids = ["135310103"]  # example: university gate + another landmark

    pf = OSMPathfinder(osm_file)

    for tid in target_ids:
        print("\n" + "="*70)
        print(f"Routing to target {tid}")
        pf.set_target(tid)
        # Optional: separate folder per target to keep outputs organized
        out_dir = os.path.join("results/neighborhoodOldV")
        pf.process_buildings_in_areas(areas, output_folder=out_dir)

if __name__ == "__main__":
    # main()
    main_multi()
