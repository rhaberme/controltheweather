# 2022-04-22

import numpy as np
import random
import time


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

weather_station_dict = {"Station 1": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 13, 14, 15, 16, 17, 18], "Station 2": [11, 12]}


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

    affected_edges = weather_station_dict[weather_station]

    for e in affected_edges:
        factor_dict = {1: 1.5, 2: 2}
        # edges which nodes are both affected will get exponential increase of costs

        graph_index = np.where(graph_matrix[:, 0] == e)
        current_cost = graph_matrix[graph_index][0, -1]
        new_cost = current_cost * factor_dict[condition]
        graph_matrix[graph_index, -1] = new_cost

    return graph_matrix


# print(change_edge_costs(graph_matrix=graph_matrix, weather_station="Station 1", condition=2))


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

        try:
            for con_edg in graph_matrix[np.where(graph_matrix[:, 2] == current_node)][:, 0]:
                connected_edges.append(con_edg)
        except IndexError:
            pass

        cost_to_current_node = best_paths[current_node][1]
        for next_edge in connected_edges:
            graph_index = np.where(graph_matrix[:, 0] == next_edge)
            current_cost = graph_matrix[graph_index][0, -1]
            cost_away = current_cost + cost_to_current_node
            # cost_back = graph.costs_back[(current_node, next_node)] + cost_to_current_node
            next_node = int(graph_matrix[graph_index, 2])
            if next_node == current_node:
                next_node = int(graph_matrix[graph_index, 1])
            if next_node not in best_paths.keys():
                best_paths[next_node] = (current_node, cost_away)
            else:
                current_smallest_cost = best_paths[next_node][1]
                if current_smallest_cost > cost_away:
                    best_paths[next_node] = (current_node, cost_away)

        next_destinations = {node: best_paths[node] for node in best_paths.keys() if node not in visited}

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


def check_weather_prediction_api():
     weather_mat = np.array([["Station 1", random.randint(0, 2)],["Station 2", random.randint(0, 2)]])
     weather_mat.reshape(-1, 2)
     return weather_mat


if __name__ == "__main__":
    running = True
    while running:
        print("CURRENT BEST ROUTE:")
        print(dijkstra(graph_matrix, 12, 13))

        weather_matrix = check_weather_prediction_api()
        current_matrix = graph_matrix
        try:
            raining_rows_in_weather_matrix = weather_matrix[np.where(weather_matrix[:, -1]!="0")]
            for rriwm_row in raining_rows_in_weather_matrix:
                station = rriwm_row[0]
                extr_weather_indic = int(rriwm_row[1])

                current_matrix = change_edge_costs(current_matrix, station, extr_weather_indic)

            print("CURRENT BEST ROUTE:")
            print(dijkstra(current_matrix, 12, 13))

        except IndexError:
            pass

        time.sleep(1)


