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

# class edge:
#     def __init__(self, id, start, end, cost):
#         self.id = id
#         self.start = start
#         self.end = end
#         self.cost = cost
#
#     def __repr__(self):
#         return "(%s,%s,%s)" % (str(self.start), str(self.end), self.cost)

class Graph():
    # Modified from https://benalexkeen.com/implementing-djikstras-shortest-path-algorithm-with-python/
    def __init__(self):
        """
        self.edges is a dict of all possible next nodes
        e.g. {'X': ['A', 'B', 'C', 'E'], ...}
        self.cost_away has all the weights between two nodes in one direction,
        self.cost_back opposite direction,
        with the two nodes as a tuple as the key
        e.g. {('X', 'A'): 7, ('X', 'B'): 2, ...}
        """
        self.edges = defaultdict(list)
        self.costs_away = {}
        # self.costs_back = {}

    def add_edge(self, from_node, to_node, cost_away):
        self.edges[from_node].append(to_node)
        self.edges[to_node].append(from_node)
        self.costs_away[(from_node, to_node)] = cost_away
        self.costs_away[(to_node, from_node)] = cost_away

def weathercond(nodes, edges, weather):
    # If weather conditions are well (weather = 0), costs will remain
    # for weather = 1, costs should be slightly increased for edges of affected nodes
    # for weather = 2, costs should be strongly increased for edges of affected nodes
    # list with all nodes and local wether stations (for local prediction) are given
    affected_nodes = nodes # affected by the prediction for the currently slected weather station
    all_edges = edges
    condition = weather
    if condition == 0:
        pass
    elif condition == 1:
        # edges which nodes are both affected will get exponential increase of costs
        factor = 1.5
        all_edges.costs[all_edges.node_from in affected_nodes] *= factor
        all_edges.costs[all_edges.node_to in affected_nodes] *= factor
    elif condition > 1:
        # edges which nodes are both affected will get exponential increase of costs
        factor = 2
        all_edges.costs[all_edges.node_from in affected_nodes] *= factor
        all_edges.costs[all_edges.node_to in affected_nodes] *= factor
    else:
        errormessage = "weather index not interpretable"
        return errormessage
    return edges

def dijsktra(graph, initial, end):
    # Modified from https://benalexkeen.com/implementing-djikstras-shortest-path-algorithm-with-python/
    # shortest paths is a dict of nodes
    # whose value is a tuple of (previous node, cost)
    best_paths = {initial: (None, 0)}
    current_node = initial
    visited = set()

    while current_node != end:
        visited.add(current_node)
        destinations = graph.edges[current_node]
        cost_to_current_node = best_paths[current_node][1]

        for next_node in destinations:
            cost_away = graph.costs_away[(current_node, next_node)] + cost_to_current_node
            # cost_back = graph.costs_back[(current_node, next_node)] + cost_to_current_node
            if next_node not in best_paths:
                best_paths[next_node] = (current_node, cost_away)
            else:
                current_smallest_cost = best_paths[next_node][1]
                if current_smallest_cost > cost_away:
                    best_paths[next_node] = (current_node, cost_away)

        next_destinations = {node: best_paths[node] for node in best_paths if node not in visited}
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

graph = Graph()
for edge in edges:
    graph.add_edge(*edge)
print("edges:\n",edges)

bestway = dijsktra(graph = graph, initial='X', end='Y')
print("bestway:\n",bestway)

# create directed graph from edges with weights (= costs)
DG = nx.DiGraph()
DG.add_weighted_edges_from(edges)
# DG.add_edges_from(bestway)
print("nodes: ",DG.number_of_nodes(), "\nedges: ", DG.number_of_edges())

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