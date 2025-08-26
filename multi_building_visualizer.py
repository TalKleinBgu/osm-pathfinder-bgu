#!/usr/bin/env python3
import json
import folium
from folium.plugins import AntPath, BeautifyIcon
from typing import Dict, List
import os

class MultiBuildingPathfindingVisualizer:
    def __init__(self, results_file: str):
        with open(results_file, 'r', encoding='utf-8') as f:
            self.data = json.load(f)

        self.target_info = self.data['_target']
        self.target_coords = [
            self.target_info['coordinates']['lat'],
            self.target_info['coordinates']['lon']
        ]
        self.results = {k: v for k, v in self.data.items() if not k.startswith('_')}

        # Algorithm colors and styles
        self.colors = {
            'shortest path (day profile)': '#1976D2',
            'few_traffic_lights path (day profile)': '#7B1FA2',
            'safest path (day profile)': '#F57C00',
            'fastest path (day profile)': '#388E3C',
            'other': '#455A64',
        }
        self.dash_styles = {
            'shortest path (day profile)': None,
            'few_traffic_lights path (day profile)': '10,5',
            'safest path (day profile)': '5,10',
            'fastest path (day profile)': '20,10',
            'other': '4,6'
        }

    def get_building_centroid(self, building_data):
        """Calculate building centroid from path start coordinates"""
        # Use the first coordinate of the first path as building location
        paths = building_data.get('paths', [])
        if paths:
            first_path = paths[0]
            path_coords = first_path.get('path_coordinates', [])
            if path_coords:
                first_coord = path_coords[0]
                if isinstance(first_coord, dict):
                    return [first_coord.get('lat'), first_coord.get('lon')]
                else:
                    return [first_coord[0], first_coord[1]]
        
        # Fallback to target_coordinates if no paths
        return [building_data['target_coordinates']['lat'], building_data['target_coordinates']['lon']]

    def _canonical_alg(self, name: str) -> str:
        if not name:
            return 'other'
        key = name.strip().lower()
        
        alias_map = {
            'shortest': 'shortest path (day profile)',
            'shortest path': 'shortest path (day profile)',
            'few_traffic_lights': 'few_traffic_lights path (day profile)',
            'few lights': 'few_traffic_lights path (day profile)',
            'safest': 'safest path (day profile)',
            'fastest': 'fastest path (day profile)',
        }
        
        return alias_map.get(key, name) if name in self.colors else alias_map.get(key, 'other')

    def create_interactive_map(self, output_file: str = "multi_building_pathfinding_map.html"):
        # Center map on target
        m = folium.Map(location=self.target_coords, zoom_start=16, tiles='OpenStreetMap')

        # Target marker
        target_name = "University Center" if self.target_info.get('type') == 'building' else "Target"
        if self.target_info.get('tags', {}).get('name'):
            target_name = self.target_info['tags']['name']
        
        folium.Marker(
            location=self.target_coords,
            popup=(f"<b>🎯 {target_name}</b><br/>"
                   f"📍 {self.target_coords[0]:.5f}, {self.target_coords[1]:.5f}<br/>"
                   f"🎯 Destination for all paths"),
            icon=folium.Icon(color='red', icon='university', prefix='fa'),
            tooltip=f"🎯 {target_name}"
        ).add_to(m)

        # Building markers group (always visible)
        building_group = folium.FeatureGroup(name="Buildings", show=True)
        building_group.add_to(m)

        all_bounds = [self.target_coords]

        # Process each building
        for building_id, building_data in self.results.items():
            # Get building centroid from the actual building location (first path coordinate)
            building_coords = self.get_building_centroid(building_data)
            all_bounds.append(building_coords)
            
            building_tags = building_data.get('target_tags', {})
            building_name = building_tags.get('name', f"Building {building_id}")

            # Create simplified popup with path information
            num_paths = len(building_data.get('paths', []))
            lat, lon = building_coords[0], building_coords[1]
            
            building_popup_html = f"""
            <div style="min-width: 280px; font-family: Arial, sans-serif;">
                <h3 style="margin: 5px 0; color: #2E7D32;">🏠 {building_name}</h3>
                <p style="margin: 5px 0;"><b>Building ID:</b> {building_id}</p>
                <p style="margin: 5px 0;"><b>Location:</b> {lat:.5f}, {lon:.5f}</p>
                <p style="margin: 5px 0;"><b>Available Routes:</b> {num_paths}</p>
                
                <div style="margin-top: 15px;">
                    <h4 style="margin: 10px 0 5px 0; color: #1976D2;">Route Options:</h4>
            """
            
            # Create a unique group for each building's paths
            building_path_groups = {}
            
            # Add path information to popup and create path groups
            for i, path_data in enumerate(building_data.get('paths', [])):
                alg = self._canonical_alg(path_data.get('algorithm', ''))
                display_alg = alg.replace(' path (day profile)', '').replace('_', ' ').title()
                color = self.colors.get(alg, self.colors['other'])
                
                dist = float(path_data.get('distance_meters', 0.0))
                time_min = float(path_data.get('time_seconds', 0.0)) / 60.0
                sig = path_data.get('num_traffic_signals', 0)
                cross = path_data.get('num_crossings', 0)
                
                # Create path group for this algorithm (hidden by default)
                group_name = f"{building_name} - {display_alg}"
                path_group = folium.FeatureGroup(name=group_name, show=False)
                path_group.add_to(m)
                building_path_groups[alg] = path_group
                
                # Add detailed path info to popup
                building_popup_html += f"""
                    <div style="margin: 8px 0; padding: 8px; background-color: #f8f9fa; border-radius: 5px; border-left: 4px solid {color};">
                        <b style="color: {color};">{display_alg}</b><br/>
                        <small>
                            📏 {dist:.0f}m • ⏱️ {time_min:.1f}min<br/>
                            🚦 {sig} signals • 🚶 {cross} crossings<br/>
                            <i>Toggle this path in the layer control →</i>
                        </small>
                    </div>
                """
                
                # Create the path
                coords = path_data.get('path_coordinates', [])
                if coords and len(coords) >= 2:
                    def _pair(c):
                        if isinstance(c, dict):
                            return [c.get('lat'), c.get('lon')]
                        return [c[0], c[1]]
                    
                    coordinates = [[*map(float, _pair(c))] for c in coords]
                    all_bounds.extend(coordinates)

                    # Path stats for popup
                    sig = path_data.get('num_traffic_signals', 0)
                    cross = path_data.get('num_crossings', 0)
                    saf = path_data.get('safety_penalties', 0)
                    dash = self.dash_styles.get(alg, self.dash_styles['other'])

                    path_popup = (
                        f"<b>{building_name} → {target_name}</b><br/>"
                        f"<b>Algorithm:</b> {display_alg}<br/>"
                        f"<b>Distance:</b> {dist:.0f} m<br/>"
                        f"<b>Time:</b> {time_min:.1f} min<br/>"
                        f"<b>Signals:</b> {sig} • <b>Crossings:</b> {cross}<br/>"
                        f"<b>Safety penalty:</b> {float(saf):.0f} m"
                    )

                    # Path line
                    folium.PolyLine(
                        locations=coordinates,
                        color=color,
                        weight=4,
                        opacity=0.8,
                        dash_array=dash,
                        popup=path_popup,
                        tooltip=f"{building_name} → {display_alg}"
                    ).add_to(path_group)
                    
                    # Animated path (lighter weight)
                    AntPath(
                        locations=coordinates,
                        delay=1200,
                        dash_array=[8, 16],
                        weight=2,
                        opacity=0.6,
                        color=color,
                        pulse_color='#ffffff'
                    ).add_to(path_group)

            building_popup_html += """
                </div>
                <p style="margin-top: 15px; font-size: 11px; color: #666; font-style: italic;">
                    💡 Use the layer control panel (top right) to show/hide specific paths for this building
                </p>
            </div>
            """

            # Building marker at actual building location
            folium.CircleMarker(
                location=building_coords,
                radius=8,
                popup=folium.Popup(building_popup_html, max_width=300),
                tooltip=f"🏠 {building_name} (Click for paths)",
                color='#2E7D32',
                fill=True,
                fillColor='#4CAF50',
                fillOpacity=0.8,
                weight=2
            ).add_to(building_group)

        # Add layer control
        folium.LayerControl(position='topright', collapsed=False, autoZIndex=True).add_to(m)

        # Custom CSS and JavaScript for better styling and path toggling
        custom_js_css = f'''
        <style>
        .leaflet-control-layers {{
            background: rgba(255, 255, 255, 0.98) !important;
            border-radius: 12px !important;
            box-shadow: 0 8px 24px rgba(0,0,0,0.18) !important;
            border: 1px solid #e0e0e0 !important;
            min-width: 220px !important;
        }}
        .leaflet-control-layers::before {{
            content: 'Map Layers';
            display: block;
            font-family: Arial, sans-serif;
            font-weight: 700;
            color: #1976D2;
            background: linear-gradient(90deg, rgba(25,118,210,0.10), rgba(25,118,210,0.04));
            padding: 10px 12px;
            border-bottom: 1px solid #e0e0e0;
        }}
        .leaflet-control-layers-overlays {{
            padding: 8px 10px !important;
        }}
        .leaflet-control-layers-overlays label {{
            font-family: Arial, sans-serif !important;
            display: flex !important;
            align-items: center !important;
            gap: 8px !important;
            padding: 6px 4px !important;
            margin: 4px 0 !important;
            border-radius: 6px !important;
            transition: background 0.2s ease !important;
        }}
        .leaflet-control-layers-overlays label:hover {{
            background: #f6f9fc !important;
        }}
        </style>
        
        <script>
        // Store active path groups per building
        window.activePaths = {{}};
        
        // Function to toggle path visibility for a specific building and algorithm
        function toggleBuildingPath(buildingId, algIndex, groupName) {{
            // Find the layer by name
            const layerControl = window[Object.keys(window).find(key => key.includes('map_') && window[key].layerControl)];
            if (!layerControl) return;
            
            // Toggle the specific path group
            Object.values(layerControl.layerControl._layers).forEach(layer => {{
                if (layer.name === groupName) {{
                    const map = layerControl.layerControl._map;
                    if (map.hasLayer(layer.layer)) {{
                        map.removeLayer(layer.layer);
                    }} else {{
                        map.addLayer(layer.layer);
                    }}
                }}
            }});
        }}
        </script>
        '''
        m.get_root().html.add_child(folium.Element(custom_js_css))

        # Fit map to all bounds
        if all_bounds:
            m.fit_bounds(all_bounds, padding=[30, 30])

        m.save(output_file)
        print(f"✅ Interactive map saved → {output_file}")
        print(f"📊 Map contains {len(self.results)} buildings with paths to {target_name}")
        return m

def main():
    results_file = "results/all_buildings_to_135310103_results.json"
    
    if not os.path.exists(results_file):
        print(f"❌ Results file not found: {results_file}")
        print("Please run the pathfinder first to generate results.")
        return
    
    vis = MultiBuildingPathfindingVisualizer(results_file)
    vis.create_interactive_map("multi_building_pathfinding_map.html")

if __name__ == "__main__":
    main()
