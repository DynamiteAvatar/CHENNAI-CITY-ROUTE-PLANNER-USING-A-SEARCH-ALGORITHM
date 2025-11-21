Abstract
This repository presents a high-performance, interactive application for optimal pathfinding within the Chennai city. Developed in Python utilizing the Streamlit framework for rapid deployment and Folium for geographical visualization, the core functionality is driven by the A* search algorithm. This project serves as a functional demonstration and performance benchmark for heuristic search methodologies applied to real-world transportation network analysis.
1. Project Overview and Objective
The objective of this project is to provide a robust and efficient solution for calculating the shortest route between two discrete geographic points within a simulated road network. This is achieved by implementing an intelligent search strategy that minimizes the computational resources required while guaranteeing the globally optimal solution. The application offers users a transparent, interactive interface to analyze pathfinding results and algorithmic performance metrics.
2. Core Methodology:
3. The A* Algorithm2.1 Graph RepresentationThe application models the city of Chennai as a weighted graph, where:Nodes (Vertices): Represent key locations (e.g., Chennai Central, Airport) with fixed geographic coordinates.Edges (Connections): Represent available road segments connecting these locations.Weights: The cost of traversing an edge is the actual physical distance between the two connected nodes, calculated using the Haversine formula.2.2 Heuristic FunctionThe A* algorithm's efficiency stems from its use of a heuristic function, $h(n)$, which estimates the cost from the current node to the goal.
3. System Capabilities
This application provides the following functional and analytical features:

Guaranteed Optimal Pathfinding: Leverages the A* search to ensure the calculated route is the shortest path in terms of distance.

Geographic Information System (GIS) Visualization: Generates an interactive map using Folium, allowing users to pan, zoom, and inspect the calculated route overlaid on various map tile layers.

Performance Benchmarking: Quantifies algorithmic efficiency by reporting the precise computation time (in milliseconds) and the total number of nodes explored during the search.

Dynamic UI/UX: Features a professional, responsive user interface built with Streamlit, including a single-click mechanism for swapping origin and destination points.
