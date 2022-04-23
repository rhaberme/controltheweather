# 2022-04-22

import numpy as np
import matplotlib as plt
# import osmnx as ox
from collections import defaultdict
import networkx as nx

# routing will be realized with Dijkstra's algorithm
# Dijkstra is one of the fastest and most simple algorithm
# and therefore suitable for real-time applications

# stops will be given by the user
# cost will be determined from the distance/traveling time and weather conditions
# paths wil be calculated respecting the weather forecast

nodes_name_dict = {1: "A", 2: "B", 3: "C", 4: "D", 5: "E", 6: "F", 7: "G", 8: "I", 9: "J", 10: "K", 11: "L", 12: "X",
                   13: "Y", 14: "H"}

graph_matrix = np.array([
    [1, 12, 1, 7],
    [2, 12, 2, 2],
    [3, 12, 3, 3],
    [4, 12, 5, 4],
    [5, 1, 2, 3],
    [6, 1, 4, 4],
    [7, 2, 4, 4],
    [8, 2, 14, 5],
    [9, 3, 11, 2],
    [10, 4, 6, 1],
    [11, 6, 14, 3],
    [12, 7, 14, 2],
    [13, 7, 13, 2],
    [14, 8, 9, 6],
    [15, 8, 10, 4],
    [16, 8, 11, 4],
    [17, 9, 11, 1],
    [18, 10, 13, 5],
])
graph_matrix.reshape(-1, 4)


def change_edge_costs(graph_matrix, weather_station, condition):
    """
    # If weather conditions are well (weather = 0), costs will remain
    # for weather = 1, costs should be slightly increased for edges of affected nodes
    # for weather = 2, costs should be strongly increased for edges of affected nodes
    # list with all nodes and local wether stations (for local prediction) are given

    :param graph_matrix: matrix with graph information
    :param weather_station: weather station affected from rain
    :param condition: indicator for the extreme weather (intense)
    :return:
    """

    weather_station_dict = {"Station 1": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 13, 14, 15, 16, 17, 18], "Station 2": [11, 12]}

    affected_edges = weather_station_dict[weather_station]

    for e in affected_edges:
        factor_dict = {1: 1.5, 2: 2}
        # edges which nodes are both affected will get exponential increase of costs

        graph_index = np.where(graph_matrix[:, 0] == e)
        current_cost = graph_matrix[graph_index][0, -1]
        new_cost = current_cost * factor_dict[condition]
        graph_matrix[graph_index, -1] = new_cost

    return graph_matrix


#print(change_edge_costs(graph_matrix=graph_matrix, weather_station="Station 1", condition=2))


def dijkstra(graph_matrix, start_node, end_node):
    # Modified from https://benalexkeen.com/implementing-djikstras-shortest-path-algorithm-with-python/
    # shortest paths is a dict of nodes
    # whose value is a tuple of (previous node, cost)
    best_paths = {start_node: (None, 0)}
    current_node = start_node
    visited = set()

    while current_node != end_node:
        visited.add(current_node)
        connected_edges = []
        try:
            for con_edg in graph_matrix[np.where(graph_matrix[:, 1] == current_node)][:, 0]:
                connected_edges.append(con_edg)
        except IndexError:
            pass

        """try:
            for con_edg in graph_matrix[np.where(graph_matrix[:, 2] == current_node)][:, 0]:
                connected_edges.append(con_edg)
        except IndexError:
            pass"""

        cost_to_current_node = best_paths[current_node][1]

        for next_edge in connected_edges:
            graph_index = np.where(graph_matrix[:, 0] == next_edge)
            current_cost = graph_matrix[graph_index][0, -1]
            cost_away = current_cost + cost_to_current_node
            # cost_back = graph.costs_back[(current_node, next_node)] + cost_to_current_node
            next_node = int(graph_matrix[graph_index, 2])
            if next_node not in best_paths.keys():
                best_paths[next_node] = (current_node, cost_away)
            else:
                current_smallest_cost = best_paths[next_node][1]
                if current_smallest_cost > cost_away:
                    best_paths[next_node] = (current_node, cost_away)

        print("best_paths:")

        print(best_paths)
        next_destinations = {node: best_paths[node] for node in best_paths.keys() if node not in visited}
        print("next_destinations:")
        print(next_destinations)
        print("visited: ")
        print(visited)
        if not next_destinations:
            return "Route Not Possible"
        # next node is the destination with the lowest cost
        current_node = min(next_destinations, key=lambda k: next_destinations[k][1])

    # Work back through destinations in shortest path
    path = []
    while current_node is not None:
        path.append(current_node)
        next_node = best_paths[current_node][0]
        current_node = next_node
    # Reverse path
    path = path[::-1]
    return path


print(dijkstra(graph_matrix=graph_matrix, start_node=12, end_node=13))

# Conversion failed â†’ manual replacement; should be created from a tree (startnodes - endnodes - costs)
edges = [
    ('X', 'A', 7),
    ('X', 'B', 2),
    ('X', 'C', 3),
    ('X', 'E', 4),
    ('A', 'B', 3),
    ('A', 'D', 4),
    ('B', 'D', 4),
    ('B', 'H', 5),
    ('C', 'L', 2),
    ('D', 'F', 1),
    ('F', 'H', 3),
    ('G', 'H', 2),
    ('G', 'Y', 2),
    ('I', 'J', 6),
    ('I', 'K', 4),
    ('I', 'L', 4),
    ('J', 'L', 1),
    ('K', 'Y', 5),
]

"""graph = Graph()
for edge in edges:
    graph.add_edge(*edge)
print("edges:\n", edges)

bestway = dijsktra(graph=graph, initial='X', end='Y')
print("bestway:\n", bestway)

# create directed graph from edges with weights (= costs)
DG = nx.DiGraph()
DG.add_weighted_edges_from(edges)
# DG.add_edges_from(bestway)
print("nodes: ", DG.number_of_nodes(), "\nedges: ", DG.number_of_edges())

# epaths = [(u, v) for (u, v, d) in DG.edges(data=True) if d["weight"] > 1]
# ebest = [(u, v) for (u, v, d) in DG.edges(data=True) if d["weight"] <= 1]

pos = nx.spring_layout(DG, seed=7)  # positions for all nodes - seed for reproducibility

# nodes
nx.draw_networkx_nodes(DG, pos, node_size=500)

# edges
nx.draw_networkx_edges(DG, pos, edgelist=DG.edges, width=3, alpha=0.5, edge_color="b", style="dashed")
# nx.draw_networkx_edges(DG, pos, edgelist=ebest, width=4)

# labels
nx.draw_networkx_labels(DG, pos, font_size=10, font_family="sans-serif")

ax = plt.pyplot.gca()
ax.margins(0.08)
plt.pyplot.axis("off")
plt.pyplot.tight_layout()
plt.pyplot.show()

"""
"""
class Edge:
    def __init__(self, from_node, to_node, cost_away):
        self.from_node = from_node
        self.to_node = to_node
        self.cost_away = cost_away

    def __repr__(self):
        return "(%s,%s,%s)" % (str(self.from_node), str(self.to_node), self.cost_away)


class Node:
    def __init__(self, location):
        self.x, self.y = location


class WeatherStation:
    def __init__(self, nodes_inside):
        self.nodes_inside = nodes_inside


class Graph:
    # Modified from https://benalexkeen.com/implementing-djikstras-shortest-path-algorithm-with-python/
    def __init__(self):
        """"""
        self.edges is a dict of all possible next nodes
        e.g. {'X': ['A', 'B', 'C', 'E'], ...}
        self.cost_away has all the weights between two nodes in one direction,
        self.cost_back opposite direction,
        with the two nodes as a tuple as the key
        e.g. {('X', 'A'): 7, ('X', 'B'): 2, ...}
        """"""
        self.edges = []
        # self.edges = defaultdict(list)

        # self.costs_away = {}
        # self.costs_back = {}

    def add_edge(self, edge):
        self.edges.append(edge)
        # self.edges[from_node].append(to_node)
        # self.edges[to_node].append(from_node)
        # self.costs_away[(from_node, to_node)] = cost_away
        # self.costs_away[(to_node, from_node)] = cost_away
"""
