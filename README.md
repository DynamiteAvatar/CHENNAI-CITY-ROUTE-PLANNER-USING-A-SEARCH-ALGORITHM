Abstract

◉ High-performance, interactive application for optimal pathfinding within Chennai city.

◉ Built using Python, Streamlit for rapid UI deployment, and Folium for real-time map visualization.

◉ Utilizes the A* (A-Star) search algorithm for efficient and optimal route computation.

◉ Serves as both a functional demonstration and a benchmark tool for heuristic search methods in transportation network analysis.

Project Overview & Objective

◉ Provides an efficient solution for computing the shortest route between two geographic points in a simulated Chennai road network.

◉ Implements an intelligent search strategy that reduces computation time while ensuring globally optimal results.

◉ Offers an interactive UI for exploring pathfinding results and evaluating algorithm performance.

**Core Methodology:**

**1. A* Algorithm:**
   
**1.1 Graph Representation**

◉ Chennai is modeled as a weighted graph.

◉ Nodes (vertices) represent key city locations (e.g., Chennai Central, Airport) with fixed latitude–longitude coordinates.

◉ Edges represent valid road segments between nodes.

◉ Weights represent real-world distances, computed using the Haversine formula.

**1.2 Heuristic Function**

◉ A* uses a heuristic function h(n) to estimate the cost from the current node to the destination.

◉ The heuristic improves search performance by directing exploration toward the goal.

◉ Ensures faster computation while maintaining optimality.

**System Capabilities**:

✔ Guaranteed Optimal Pathfinding:
Determines the shortest route using A*, ensuring globally optimal distance computation.

✔ GIS-Based Interactive Map:
Uses Folium to generate an interactive map with pan/zoom capabilities and multiple tile styles.

✔ Performance Benchmarking:
Displays computation time (in milliseconds) and the total number of explored nodes during pathfinding.

✔ Dynamic UI/UX:
Professional and responsive Streamlit interface, including one-click origin–destination swap functionality.
