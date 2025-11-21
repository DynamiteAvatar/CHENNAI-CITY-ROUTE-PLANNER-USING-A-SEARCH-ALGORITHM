import streamlit as st
import folium
import heapq
import time
from math import radians, cos, sin, asin, sqrt
from streamlit.components.v1 import html

# --- 1. Data Setup (Keep the original data) ---
coordinates = {
    "Chennai Central": (13.0827, 80.2707),
    "Egmore": (13.0733, 80.2606),
    "Marina Beach": (13.0500, 80.2824),
    "T Nagar": (13.0422, 80.2337),
    "Guindy": (13.0060, 80.2209),
    "Adyar": (13.0068, 80.2550),
    "Velachery": (12.9792, 80.2184),
    "IIT Madras": (12.9916, 80.2335),
    "Anna Nagar": (13.0860, 80.2209),
    "Airport": (12.9941, 80.1709),
    "Tambaram": (12.9250, 80.1181),
    "Sholinganallur": (12.8996, 80.2278),
    "Ambattur": (13.1075, 80.1602),
    "Mylapore": (13.0338, 80.2676),
    "Perambur": (13.1138, 80.2299),
    "Kodambakkam": (13.0487, 80.2214),
    "Madipakkam": (12.9632, 80.1986),
    "Porur": (13.0330, 80.1588),
    "Pallavaram": (12.9673, 80.1496)
}

roads = {
    "Chennai Central": ["Egmore", "Anna Nagar"],
    "Egmore": ["Chennai Central", "T Nagar", "Perambur", "Marina Beach"],
    "T Nagar": ["Egmore", "Guindy", "Adyar", "Kodambakkam"],
    "Guindy": ["T Nagar", "IIT Madras", "Airport", "Porur"],
    "Adyar": ["T Nagar", "IIT Madras", "Velachery", "Mylapore"],
    "IIT Madras": ["Guindy", "Adyar", "Velachery"],
    "Velachery": ["IIT Madras", "Adyar", "Sholinganallur", "Madipakkam"],
    "Airport": ["Guindy", "Tambaram", "Pallavaram"],
    "Tambaram": ["Airport", "Sholinganallur"],
    "Sholinganallur": ["Velachery", "Tambaram"],
    "Anna Nagar": ["Chennai Central", "Ambattur", "Kodambakkam"],
    "Ambattur": ["Anna Nagar", "Perambur"],
    "Perambur": ["Ambattur", "Egmore"],
    "Mylapore": ["Adyar", "Marina Beach"],
    "Marina Beach": ["Mylapore", "Egmore"],
    "Kodambakkam": ["T Nagar", "Anna Nagar"],
    "Madipakkam": ["Velachery", "Pallavaram", "Porur"],
    "Pallavaram": ["Airport", "Madipakkam"],
    "Porur": ["Guindy", "Madipakkam"]
}


# --- 2. Core Functions (Same logic, Haversine in KM) ---
def haversine(coord1, coord2):
    lat1, lon1 = coord1
    lat2, lon2 = coord2
    R = 6371
    dlat = radians(lat2 - lat1)
    dlon = radians(lon2 - lon1)
    a = sin(dlat / 2) ** 2 + cos(radians(lat1)) * cos(radians(lat2)) * sin(dlon / 2) ** 2
    c = 2 * asin(sqrt(a))
    return R * c


def build_graph():
    graph = {}
    for src in roads:
        graph[src] = []
        for dest in roads[src]:
            dist = haversine(coordinates[src], coordinates[dest])
            graph[src].append((dest, dist))
    return graph


def a_star_search(graph, start, goal):
    start_time = time.time()
    open_set = [(haversine(coordinates[start], coordinates[goal]), 0, start, [start])]
    visited = set()
    nodes_visited = 0

    while open_set:
        est_total_cost, cost_so_far, current, path = heapq.heappop(open_set)

        if current in visited:
            continue

        visited.add(current)
        nodes_visited += 1

        if current == goal:
            end_time = time.time()
            return path, cost_so_far, nodes_visited, (end_time - start_time) * 1000

        for neighbor, dist in graph.get(current, []):
            if neighbor not in visited:
                g = cost_so_far + dist
                h = haversine(coordinates[neighbor], coordinates[goal])
                f = g + h
                heapq.heappush(open_set, (f, g, neighbor, path + [neighbor]))

    end_time = time.time()
    return None, float('inf'), nodes_visited, (end_time - start_time) * 1000


# --- 3. FIX 1: Plot Route with Attribution ---
def plot_route_on_map(start, goal, path, map_style):
    # Dictionary of tile styles and their required attribution
    tile_options = {
        "OpenStreetMap": ("OpenStreetMap", "© OpenStreetMap contributors"),
        "CartoDB Dark Matter": (
            "CartoDB dark_matter",
            "&copy; <a href='https://carto.com/attributions'>CARTO</a>"
        ),
        "Stamen Terrain": (
            "Stamen Terrain",
            "Map tiles by <a href='http://stamen.com'>Stamen Design</a>, under <a href='http://creativecommons.org/licenses/by/3.0'>CC BY 3.0</a>. Data by <a href='http://openstreetmap.org'>OpenStreetMap</a>, under <a href='http://www.openstreetmap.org/copyright'>ODbL</a>."
        ),
        "Stamen Toner": (
            "Stamen Toner",
            "Map tiles by <a href='http://stamen.com'>Stamen Design</a>, under <a href='http://creativecommons.org/licenses/by/3.0'>CC BY 3.0</a>. Data by <a href='http://openstreetmap.org'>OpenStreetMap</a>, under <a href='http://www.openstreetmap.org/copyright'>ODbL</a>."
        )
    }

    # Get the tiles and attribution string
    tiles, attr = tile_options.get(map_style, tile_options["OpenStreetMap"])

    center_lat = (coordinates[start][0] + coordinates[goal][0]) / 2
    center_lon = (coordinates[start][1] + coordinates[goal][1]) / 2

    # Pass both tiles and attr to avoid the ValueError
    m = folium.Map(location=[center_lat, center_lon], zoom_start=12, tiles=tiles, attr=attr)

    # Add start marker
    folium.Marker(
        coordinates[start],
        popup=f"Start: **{start}**",
        tooltip="Start Point",
        icon=folium.Icon(color='green', icon='play', prefix='fa')
    ).add_to(m)

    # Add goal marker
    folium.Marker(
        coordinates[goal],
        popup=f"Goal: **{goal}**",
        tooltip="Destination",
        icon=folium.Icon(color='red', icon='flag', prefix='fa')
    ).add_to(m)

    # Add route line (using gold/orange for high contrast)
    route_coordinates = [coordinates[loc] for loc in path]
    folium.PolyLine(
        route_coordinates,
        color='#FFC300',  # Gold/Orange
        weight=5,
        opacity=0.9,
        tooltip="Best Route"
    ).add_to(m)

    return m._repr_html_()


def swap_locations():
    """Function to swap start and goal points in session state."""
    if 'start_loc' in st.session_state and 'goal_loc' in st.session_state:
        start = st.session_state['start_loc']
        goal = st.session_state['goal_loc']
        st.session_state['start_loc'] = goal
        st.session_state['goal_loc'] = start
        if 'path' in st.session_state: del st.session_state['path']
        st.rerun()


# --- 4. FIX 2: Aesthetic UI Overhaul ---
def run_app():
    st.set_page_config(layout="wide", page_title="Chennai Smart Navigator | A*", initial_sidebar_state="expanded")

    # --- Custom CSS for Teal & Gold Theme ---
    st.markdown("""
    <style>
    /* Main container and background adjustments */
    .stApp {
        background-color: #f7f9fb; /* Very light subtle background */
    }
    .main-header {
        color: #008080; /* Teal color */
        font-size: 3.5em;
        font-weight: 800;
        text-align: center;
        padding: 10px 0;
        border-bottom: 3px solid #FFC300; /* Gold separator */
        margin-bottom: 20px;
    }

    /* Custom primary button style */
    .stButton button, .stDownloadButton button {
        background-color: #008080; /* Teal */
        color: white;
        border-radius: 8px;
        font-weight: bold;
        transition: all 0.2s ease-in-out;
        border: none;
        padding: 10px;
    }
    .stButton button:hover {
        background-color: #00a0a0; /* Lighter Teal on hover */
    }

    /* Result Box Styling */
    .route-info-box {
        background-color: #e6f7ff; /* Light blue/cyan tint */
        border-radius: 12px;
        padding: 20px;
        margin-top: 15px;
        border-left: 6px solid #FFC300; /* Gold accent */
        box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1);
    }

    /* Metric Boxes */
    .metric-container {
        display: flex;
        justify-content: space-around;
        gap: 15px;
        margin-top: 20px;
    }
    .metric-box {
        background-color: white;
        border-radius: 10px;
        padding: 15px;
        text-align: center;
        flex: 1;
        box-shadow: 0 2px 5px rgba(0, 0, 0, 0.05);
        border: 1px solid #ddd;
    }
    .metric-label {
        font-size: 0.9em;
        color: #555;
        margin-bottom: 5px;
        font-weight: 600;
    }
    .metric-value {
        font-size: 1.8em;
        font-weight: 800;
        color: #008080; /* Teal */
    }
    </style>
    """, unsafe_allow_html=True)

    # --- Header ---
    st.markdown('<p class="main-header">🗺️ Chennai Smart Route Planner</p>', unsafe_allow_html=True)

    # Initialize session state for locations
    if 'start_loc' not in st.session_state:
        location_names = sorted(list(coordinates.keys()))
        st.session_state['start_loc'] = location_names[location_names.index("Chennai Central")]
        st.session_state['goal_loc'] = location_names[location_names.index("Airport")]

    # --- Sidebar for Configuration ---
    with st.sidebar:
        st.header("⚙️ Settings & Data")

        st.subheader("Map Tile Style")
        map_style = st.selectbox(
            "Choose Map Appearance:",
            ["CartoDB Dark Matter", "OpenStreetMap", "Stamen Terrain", "Stamen Toner"],
            key='map_style_select'
        )

        st.markdown("---")
        st.subheader("Graph Data Summary")
        st.info(f"**Total Locations:** {len(coordinates)}")
        st.info(f"**Total Connections:** {sum(len(v) for v in roads.values())}")

        with st.expander("Show All Locations (20+)"):
            st.dataframe(
                data=list(coordinates.items()),
                column_config={
                    0: "Location",
                    1: st.column_config.ListColumn("Coordinates (Lat, Lon)")
                },
                hide_index=True,
                height=300
            )

    # --- Main Layout: Columns for Input & Map ---
    col1, col2 = st.columns([1, 2.5])

    with col1:
        st.markdown("## 1. Plan Route 📍")

        # Location Selection
        location_names = sorted(list(coordinates.keys()))

        st.container(border=True).selectbox(
            "**Departure Point** 🟢",
            location_names,
            key='start_loc'
        )

        # Swap Button in a small column for placement
        st_c1, st_c2, st_c3 = st.columns([1, 1, 1])
        with st_c2:
            st.button("⬆️⬇️", on_click=swap_locations, help="Swap Start and Destination", use_container_width=True)

        st.container(border=True).selectbox(
            "**Destination Point** 🔴",
            location_names,
            key='goal_loc'
        )

        st.markdown("---")

        # Location Details Box
        st.subheader("Selected Location Coordinates")
        st.code(f"Start: {coordinates[st.session_state['start_loc']]}", language="python")
        st.code(f"Goal: {coordinates[st.session_state['goal_loc']]}", language="python")

        st.markdown("---")

        # Find Route Button
        if st.button("🌟 Calculate Optimal Route", type="primary"):
            start = st.session_state['start_loc']
            goal = st.session_state['goal_loc']
            if start == goal:
                st.error("Departure and destination can't be the same!")
            else:
                graph = build_graph()
                with st.spinner(f"Running A* search..."):
                    path, distance, nodes_visited, comp_time = a_star_search(graph, start, goal)
                    st.session_state['path'] = path
                    st.session_state['distance'] = distance
                    st.session_state['nodes'] = nodes_visited
                    st.session_state['time'] = comp_time
                    st.session_state['map_style_final'] = st.session_state['map_style_select']
                    st.session_state['start_loc_final'] = start
                    st.session_state['goal_loc_final'] = goal

        # Reset Button
        if st.button("🗑️ Reset All Data", help="Clear the map, results, and reset select boxes"):
            for key in list(st.session_state.keys()):
                if key not in ['start_loc', 'goal_loc', 'map_style_select']:  # Keep select box current values
                    del st.session_state[key]
            st.rerun()

    # 2. Results & Map Column
    with col2:
        st.markdown("## 2. Route Visualization & Metrics 📈")

        if 'path' in st.session_state and st.session_state['path']:
            path = st.session_state['path']
            distance = st.session_state['distance']
            nodes = st.session_state['nodes']
            comp_time = st.session_state['time']
            map_style_final = st.session_state['map_style_final']
            start_final = st.session_state['start_loc_final']
            goal_final = st.session_state['goal_loc_final']

            # --- Key Metric Boxes ---
            st.markdown('<div class="route-info-box">', unsafe_allow_html=True)
            st.success(f"**Optimal Route Found!** 🥳 from **{start_final}** to **{goal_final}**")

            st.markdown('<div class="metric-container">', unsafe_allow_html=True)

            # Distance Metric Box
            st.markdown(f"""
                <div class="metric-box">
                    <div class="metric-label">Total Distance</div>
                    <div class="metric-value">🛣️ {distance:.2f} km</div>
                </div>
            """, unsafe_allow_html=True)

            # Nodes Explored Metric Box
            st.markdown(f"""
                <div class="metric-box">
                    <div class="metric-label">Nodes Explored (A*)</div>
                    <div class="metric-value">🧠 {nodes}</div>
                </div>
            """, unsafe_allow_html=True)

            # Time Metric Box
            st.markdown(f"""
                <div class="metric-box">
                    <div class="metric-label">Computation Time</div>
                    <div class="metric-value">⏱️ {comp_time:.2f} ms</div>
                </div>
            """, unsafe_allow_html=True)

            st.markdown('</div>', unsafe_allow_html=True)  # Close metric-container
            st.markdown('</div>', unsafe_allow_html=True)  # Close route-info-box

            st.markdown("### Interactive Map 📍")

            # Plot the route
            map_html = plot_route_on_map(start_final, goal_final, path, map_style_final)
            html(map_html, height=550)

            # --- Detailed Route & Simulated Chart ---
            st.markdown("---")

            detail_col1, detail_col2 = st.columns([1, 1])

            with detail_col1:
                with st.expander("Detailed Route Steps 🚶", expanded=False):
                    for i, step in enumerate(path):
                        if i == 0:
                            st.markdown(f"**🟢 Start:** {step}")
                        elif i == len(path) - 1:
                            st.markdown(f"**🔴 End:** {step}")
                        else:
                            st.markdown(f"**{i}.** {step}")

            with detail_col2:
                # Simulated Data Chart (for visual appeal)
                import pandas as pd
                import numpy as np
                # Use cummulative distance as index for a more realistic x-axis
                cum_dist = np.cumsum(
                    [0] + [haversine(coordinates[path[i]], coordinates[path[i + 1]]) for i in range(len(path) - 1)])

                chart_data = pd.DataFrame(
                    {
                        'Simulated Road Detail (m)': np.random.rand(len(path)) * 100 + 50,  # Simulated elevation
                        'Distance (km)': cum_dist
                    }
                ).set_index('Distance (km)')

                st.subheader("Simulated Route Profile")
                st.area_chart(chart_data, color="#FFC300")  # Use gold accent color

        else:
            # Initial state message
            st.info(
                "👆 Use the left panel to select your locations and map style, then click 'Calculate Optimal Route' to see the path, metrics, and interactive map here.")


if __name__ == '__main__':
    run_app()
